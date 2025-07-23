from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, Linear, to_hetero
from torch_geometric.data import HeteroData
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score
from tqdm import tqdm
import pandas as pd

# -------------------- 数据准备 --------------------
def load_and_prepare_data(save_path=None):
    """加载同构图数据并应用预划分的掩码"""
    graph_path = 'data_new/homogeneous_graph_8.pt'
    id_dir='data'
    data = torch.load(graph_path)
    print(type(data))
    print(len(data))
    return data

def check_masks(data):
    train_count = data.train_mask.sum().item()
    val_count = data.val_mask.sum().item()
    test_count = data.test_mask.sum().item()
    total_count = data.x.size(0)
    print(f"节点总数: {total_count}, train: {train_count}, val: {val_count}, test: {test_count}")

class HomoGNN(torch.nn.Module):
    def __init__(self, hidden_channels=73, out_channels=2, num_heads=4, dropout=0):
        super().__init__()
        self.dropout = dropout

        # 第一层图卷积
        self.conv1 = GATv2Conv(-1, hidden_channels, heads=num_heads, dropout=dropout)
        
        # 第二层图卷积
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        
        # 分类头
        self.classifier = Linear(hidden_channels * num_heads, out_channels)
        
        # 批归一化
        self.bn = torch.nn.BatchNorm1d(hidden_channels * num_heads)

    def forward(self, x, edge_index):
        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(self.bn(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 第二层卷积
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(self.bn(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 分类预测
        out = self.classifier(x)
        
        return out

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, SAGEConv, global_mean_pool, JumpingKnowledge
from torch_geometric.nn import BatchNorm, LayerNorm, GraphNorm

class EnhancedHomoGNN(torch.nn.Module):
    def __init__(self, 
                 in_channels=64, 
                 hidden_channels=128, 
                 out_channels=2, 
                 num_heads=8, 
                 dropout=0.3,
                 num_layers=4):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        
        # 输入投影层
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # 多类型图卷积层
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # 交替使用不同类型的图卷积
            if i % 2 == 0:
                conv = GATv2Conv(
                    hidden_channels if i > 0 else hidden_channels,
                    hidden_channels // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                    edge_dim=None,
                    add_self_loops=True
                )
            else:
                conv = SAGEConv(
                    hidden_channels,
                    hidden_channels,
                    aggr='mean',
                    normalize=True,
                    project=True
                )
            
            self.convs.append(conv)
            
            # 使用LayerNorm代替BatchNorm，因为它对输入维度变化更鲁棒
            norm = LayerNorm(hidden_channels)
            self.norms.append(norm)
        
        # 跳跃连接机制 - 使用最大池化而不是拼接，避免维度问题
        self.jk = JumpingKnowledge(mode='max')
        
        # 注意力池化层
        self.att_pool = nn.Linear(hidden_channels, 1)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        
        # 残差连接用的投影
        self.res_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

    def forward(self, x, edge_index, batch=None):
        # 输入投影
        h = self.input_proj(x)
        x_res = self.res_proj(x)
        h = h + x_res  # 初始残差连接
        
        # 存储各层特征用于跳跃连接
        xs = []
        
        # 图卷积层堆叠
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            
            if i % 2 == 0:
                h = F.elu(h)
            else:
                h = F.leaky_relu(h, negative_slope=0.2)
            
            h = F.dropout(h, p=self.dropout, training=self.training)
            xs.append(h)
        
        # 跳跃连接聚合 - 使用max模式避免维度问题
        h = self.jk(xs)
        
        # 注意力池化 (如果使用批量处理)
        if batch is not None:
            att_weights = torch.sigmoid(self.att_pool(h))
            h = h * att_weights
            h = global_mean_pool(h, batch)
        
        # 分类预测
        out = self.classifier(h)
        
        return out

from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def compute_metrics(out, y, mask):
    """计算所有指标的通用函数"""
    metrics = {}
    if mask.sum() == 0:
        return metrics
    
    pred = out[mask].argmax(dim=1)
    true = y[mask]
    probs = F.softmax(out[mask], dim=1)[:, 1]  # 二分类问题取正类概率
    
    # 转换为numpy前先detach
    true_np = true.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    probs_np = probs.cpu().detach().numpy()
    
    # 分类指标
    metrics['acc'] = accuracy_score(true_np, pred_np)
    metrics['precision'] = precision_score(true_np, pred_np, average='macro')
    metrics['recall'] = recall_score(true_np, pred_np, average='macro')
    metrics['f1'] = f1_score(true_np, pred_np, average='macro')
    
    # 概率指标
    precision_curve, recall_curve, _ = precision_recall_curve(true_np, probs_np)
    metrics['auprc'] = auc(recall_curve, precision_curve)
    
    return metrics

def train(model, data, optimizer, criterion):
    """训练函数，返回损失和训练集指标"""
    model.train()
    optimizer.zero_grad()

    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    # 计算训练集指标
    train_metrics = compute_metrics(out, data.y, data.train_mask)
    return loss.item(), train_metrics

def evaluate(model, data, mask_type='val'):
    """评估函数，返回指定mask的指标"""
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        mask = getattr(data, f"{mask_type}_mask")
        return compute_metrics(out, data.y, mask)

def print_metrics(epoch, loss, train_metrics, val_metrics):
    """格式化打印指标"""
    print(f"\nEpoch {epoch:03d}:")
    print(f"Loss: {loss:.4f}")
    
    print("Train Metrics:", end=" ")
    for k, v in train_metrics.items():
        print(f"{k.upper()}: {v:.4f}", end=" | ")
    
    print("\nVal Metrics:  ", end=" ")
    for k, v in val_metrics.items():
        print(f"{k.upper()}: {v:.4f}", end=" | ")
    print()

def train_gnn(data, epochs=100, patience=100, eval_freq=10):
    """完整的训练流程"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EnhancedHomoGNN(
        in_channels=73,
        hidden_channels=128,
        out_channels=2,
        num_heads=4,
        dropout=0.2,
        num_layers=4
    ).to(device)

    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    patience_counter = 0
    history = []
    best_epoch = 0

    for epoch in tqdm(range(epochs), desc="Training"):
        # 训练并获取训练指标
        loss, train_metrics = train(model, data, optimizer, criterion)
        
        # 定期评估
        if epoch % eval_freq == 0 or epoch == epochs - 1:
            val_metrics = evaluate(model, data, 'val')
            
            # 记录历史
            history.append({
                'epoch': epoch,
                'loss': loss,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            })
            
            # 打印指标
            print_metrics(epoch, loss, train_metrics, val_metrics)
            
            # 早停机制
            current_val_f1 = val_metrics.get('f1', 0)
            if current_val_f1 > best_val_f1:
                best_val_f1 = current_val_f1
                best_epoch = epoch
                torch.save(model.state_dict(), 'best_model.pt')
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}, no improvement after {patience} epochs")
                break
        else:
            # 非评估轮次只记录基础信息
            history.append({
                'epoch': epoch,
                'loss': loss,
                **{f'train_{k}': v for k, v in train_metrics.items()}
            })

    # 最终测试
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    test_metrics = evaluate(model, data, 'test')
    
    print("\nFinal Test Results:")
    for k, v in test_metrics.items():
        print(f"{k.upper()}: {v:.4f}")

    return model, history

# -------------------- 执行训练 --------------------
if __name__ == "__main__":
    data = load_and_prepare_data()
    check_masks(data)
    
    model, history = train_gnn(data)
    
    # 保存完整模型和指标历史
    torch.save({
        'model_state': model.state_dict(),
        'metrics': pd.DataFrame(history)  # 转为DataFrame方便分析
    }, 'gnn_training_results.pt')