from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, Linear, to_hetero
from torch_geometric.data import HeteroData
from sklearn.metrics import precision_recall_curve, auc
from tqdm import tqdm


# -------------------- 数据准备 --------------------
def load_and_prepare_data():
    """加载异构图数据并应用预划分的掩码"""
    data = torch.load('hetero_graph_1.pt')

    # 为每个节点类型创建掩码（使用之前保存的split字段）
    for node_type in data.node_types:
        split = data[node_type].split
        data[node_type].train_mask = torch.tensor([s == 'train' for s in split], dtype=torch.bool)
        data[node_type].val_mask = torch.tensor([s == 'val' for s in split], dtype=torch.bool)
        data[node_type].test_mask = torch.tensor([s == 'test' for s in split], dtype=torch.bool)

    return data

def check_masks(data):
    for node_type in data.node_types:
        train_count = data[node_type].train_mask.sum().item()
        val_count = data[node_type].val_mask.sum().item()
        test_count = data[node_type].test_mask.sum().item()
        total_count = data[node_type].x.size(0)
        print(f"【{node_type}】节点总数: {total_count}, train: {train_count}, val: {val_count}, test: {test_count}")

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=2, num_heads=4, dropout=0.3):
        super().__init__()
        self.dropout = dropout

        # 第一层异构卷积
        self.conv1 = HeteroConv({
            ('institution', 'to', 'institution'): GATv2Conv(-1, hidden_channels, heads=num_heads, dropout=dropout),
            ('person', 'to', 'person'): GATv2Conv(-1, hidden_channels, heads=num_heads, dropout=dropout),
            ('product', 'to', 'product'): GATv2Conv(-1, hidden_channels, heads=num_heads, dropout=dropout)
        }, aggr='mean')

        # 第二层异构卷积
        self.conv2 = HeteroConv({
            ('institution', 'to', 'institution'): GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads,
                                                             dropout=dropout),
            ('person', 'to', 'person'): GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads,
                                                  dropout=dropout),
            ('product', 'to', 'product'): GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads,
                                                    dropout=dropout)
        }, aggr='mean')

        # 异常分类头（三类节点共享）
        self.classifier = Linear(hidden_channels * num_heads, out_channels)

        # 节点类型特定的批归一化
        self.bn_dict = torch.nn.ModuleDict({
            node_type: torch.nn.BatchNorm1d(hidden_channels * num_heads)
            for node_type in ['institution', 'person', 'product']
        })

    def forward(self, x_dict, edge_index_dict):
        # 第一层卷积
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(self.bn_dict[key](x)) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        # 第二层卷积
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {key: F.leaky_relu(self.bn_dict[key](x)) for key, x in x_dict.items()}
        x_dict = {key: F.dropout(x, p=self.dropout, training=self.training) for key, x in x_dict.items()}

        # 分类预测
        out_dict = {}
        for node_type, x in x_dict.items():
            out_dict[node_type] = self.classifier(x)

        return out_dict


from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    out_dict = model(data.x_dict, data.edge_index_dict)

    # 计算每种节点类型的损失
    losses = []
    for node_type in data.node_types:
        mask = data[node_type].train_mask
        out = out_dict[node_type]
        labels = torch.tensor(data[node_type].label, device=out.device)  # 将标签移到和输出相同的设备
        loss = criterion(out[mask], labels[mask])
        losses.append(loss)

    # 总损失
    total_loss = sum(losses)
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


def evaluate(model, data, mask_type='val'):
    """评估模型性能"""
    model.eval()
    out_dict = model(data.x_dict, data.edge_index_dict)
    metrics = {}

    with torch.no_grad():
        for node_type in data.node_types:
            mask = getattr(data[node_type], f"{mask_type}_mask")
            if mask.sum() == 0: continue

            pred = out_dict[node_type][mask].argmax(dim=1)
            true = torch.tensor(data[node_type].label, device=pred.device)[mask]  # 将标签移到和预测相同的设备
            probs = F.softmax(out_dict[node_type][mask], dim=1)[:, 1]

            # 计算指标
            metrics[f'{node_type}/acc'] = accuracy_score(true.cpu(), pred.cpu())
            metrics[f'{node_type}/f1'] = f1_score(true.cpu(), pred.cpu(), average='macro')

            # AUPRC
            precision, recall, _ = precision_recall_curve(true.cpu(), probs.cpu())
            metrics[f'{node_type}/auprc'] = auc(recall, precision)

    return metrics


def test(model, data):
    model.eval()
    out_dict = model(data.x_dict, data.edge_index_dict)

    metrics = {}
    for node_type in data.node_types:
        mask = data[node_type].test_mask
        pred = out_dict[node_type][mask].argmax(dim=1)
        labels = torch.tensor(data[node_type].label, device=pred.device)  # 将标签移到和预测相同的设备
        true = labels[mask]

        # 计算准确率
        metrics[f'{node_type}_acc'] = accuracy_score(true.cpu(), pred.cpu())

        # 计算F1分数（macro平均）
        metrics[f'{node_type}_f1'] = f1_score(true.cpu(), pred.cpu(), average='macro')

        # 计算AUPRC
        # 计算精确率和召回率
        precision, recall, _ = precision_recall_curve(true.cpu(), out_dict[node_type][mask][:, 1].detach().cpu())
        # 计算AUPRC
        auprc = auc(recall, precision)
        metrics[f'{node_type}_auprc'] = auprc

    return metrics


def train_gnn(data, epochs=100, patience=10):
    """完整的训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HeteroGNN().to(device)
    data = data.to(device)  # 将数据移到设备上
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_auprc = 0
    patience_counter = 0
    history = []

    for epoch in tqdm(range(epochs), desc="Training"):
        # 训练
        loss = train(model, data, optimizer, criterion)

        # 验证
        if epoch % 2 == 0:  # 每2轮验证一次
            val_metrics = evaluate(model, data, 'val')
            history.append({'epoch': epoch, 'loss': loss, **val_metrics})

            # 早停机制
            current_auprc = sum(v for k, v in val_metrics.items() if 'auprc' in k) / len(data.node_types)
            if current_auprc > best_val_auprc:
                best_val_auprc = current_auprc
                patience_counter = 0
                torch.save(model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    # 加载最佳模型并测试
    model.load_state_dict(torch.load('best_model.pt', map_location=device))  # 确保加载到正确的设备
    test_metrics = evaluate(model, data, 'test')
    print("\nTest Results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")

    return model, history


# -------------------- 5. 执行训练 --------------------
data = load_and_prepare_data()
check_masks(data)
model, history = train_gnn(data)

# 保存完整模型
torch.save({
    'model_state': model.state_dict(),
    'metrics': history
}, 'gnn_training_results.pt')
