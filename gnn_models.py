import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, HeteroConv, Linear
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
import numpy as np

# -------------------------------
# 模型定义：异构图 GCN / SAGE / GAT (与您的一致，无需修改)
# -------------------------------
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels, model_type='sage'):
        super().__init__()
        if model_type == 'sage':
            conv_class = SAGEConv
        elif model_type == 'gcn':
            conv_class = GCNConv
        elif model_type == 'gat':
            # GATConv的输入格式与其他不同，需要特别处理
            conv_class = lambda in_channels, out_channels: GATConv(in_channels, out_channels, heads=1, add_self_loops=False)
        else:
            raise ValueError("model_type must be 'sage', 'gcn', or 'gat'")

        self.conv1 = HeteroConv({
            edge_type: conv_class(
                # GATConv需要一个整数，而不是元组
                (in_channels_dict[edge_type[0]] if model_type != 'gat' else in_channels_dict[edge_type[0]], 
                 in_channels_dict[edge_type[2]]),
                hidden_channels
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        self.conv2 = HeteroConv({
            edge_type: conv_class(hidden_channels, hidden_channels)
            for edge_type in metadata[1]
        }, aggr='sum')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return self.lin(x_dict['customer'])


# -------------------------------
# 辅助函数：计算所有评估指标 (新增)
# -------------------------------
def calculate_metrics(y_true, y_pred, y_prob):
    """计算并返回一套完整的分类评估指标"""
    metrics = {}
    metrics['Accuracy']  = accuracy_score(y_true, y_pred)
    # 假设是二分类，并且正类标签为1
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall']    = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1 Score']  = f1_score(y_true, y_pred, zero_division=0)
    metrics['AUPRC']     = average_precision_score(y_true, y_prob)
    return metrics


# -------------------------------
# 训练 + 验证 + 测试 主函数 (已修改)
# -------------------------------
def train_and_evaluate(data, model_type='sage', hidden_channels=64, epochs=30, lr=0.005, verbose_interval=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    in_channels_dict = {node_type: data[node_type].x.size(1) for node_type in data.node_types}
    out_channels = int(data['customer'].y.max().item()) + 1

    model = HeteroGNN(
        metadata=data.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        model_type=model_type
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n🔧 Start training with model: {model_type.upper()}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)

        loss = F.cross_entropy(out[data['customer'].train_mask], data['customer'].y[data['customer'].train_mask])
        loss.backward()
        optimizer.step()

        # --- 在验证集上评估 ---
        if epoch % verbose_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                out = model(data.x_dict, data.edge_index_dict)
                # 使用softmax获取概率
                probs = F.softmax(out, dim=1)
                pred = probs.argmax(dim=1)

                # 提取验证集数据
                val_mask = data['customer'].val_mask
                val_true = data['customer'].y[val_mask].cpu().numpy()
                val_pred = pred[val_mask].cpu().numpy()
                val_prob = probs[val_mask][:, 1].cpu().numpy() # 取正类的概率

                # 计算所有指标
                val_metrics = calculate_metrics(val_true, val_pred, val_prob)
                
                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | Val F1: {val_metrics['F1 Score']:.4f} | Val AUPRC: {val_metrics['AUPRC']:.4f}")

    # --- 最终在测试集上评估 ---
    print("\n===== GNN最终测试集评估指标 =====")
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        probs = F.softmax(out, dim=1)
        pred = probs.argmax(dim=1)

        test_mask = data['customer'].test_mask
        test_true = data['customer'].y[test_mask].cpu().numpy()
        test_pred = pred[test_mask].cpu().numpy()
        test_prob = probs[test_mask][:, 1].cpu().numpy()

        test_metrics = calculate_metrics(test_true, test_pred, test_prob)
    
    # 打印最终结果
    for name, value in test_metrics.items():
        print(f"{name:<12}: {value:.4f}")
        
    return model, test_metrics


# -------------------------------
# 主程序执行部分
# -------------------------------
if __name__ == '__main__':
    # 加载图数据
    try:
        data = torch.load('data_new/hetero_graph.pt')
    except FileNotFoundError:
        print("错误：无法找到 'data_new/hetero_graph.pt'。请确保文件路径正确。")
        exit()

    # --- 训练并评估 GraphSAGE ---
    sage_model, sage_metrics = train_and_evaluate(
        data, model_type='sage', epochs=100, lr=0.005
    )

    # # --- 训练并评估 GAT ---
    # gat_model, gat_metrics = train_and_evaluate(
    #     data, model_type='gat', epochs=200, lr=0.005, verbose_interval=20
    # )