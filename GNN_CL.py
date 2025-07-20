from torch_geometric.data import HeteroData
import torch

# 加载图
data = torch.load('data/hetero_graph.pt')
# for node_type in data.node_types:
#     print(f"Node type: {node_type}")
    
#     # 查看前5个特征
#     print("x[:5]:")
#     print(data[node_type].x[:5])

#     # 如果保存了 cust_id，查看前5个 ID
#     if 'cust_id' in data[node_type]:
#         print("cust_id[:5]:")
#         print(data[node_type].cust_id[:5])
#         print(data[node_type].label[:5])

#     print("-" * 40)
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, Linear, to_hetero
from torch_geometric.data import HeteroData
from sklearn.metrics import precision_recall_curve, auc

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
            ('institution', 'to', 'institution'): GATv2Conv(hidden_channels*num_heads, hidden_channels, heads=num_heads, dropout=dropout),
            ('person', 'to', 'person'): GATv2Conv(hidden_channels*num_heads, hidden_channels, heads=num_heads, dropout=dropout),
            ('product', 'to', 'product'): GATv2Conv(hidden_channels*num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        }, aggr='mean')
        
        # 异常分类头（三类节点共享）
        self.classifier = Linear(hidden_channels*num_heads, out_channels)
        
        # 节点类型特定的批归一化
        self.bn_dict = torch.nn.ModuleDict({
            node_type: torch.nn.BatchNorm1d(hidden_channels*num_heads)
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

import torch
from torch_geometric.data import HeteroData

def split_hetero_data(data, split_ratio=(0.7, 0.15, 0.15)):
    """
    为异构图数据划分train/val/test掩码
    参数:
        data: HeteroData对象
        split_ratio: (train_ratio, val_ratio, test_ratio)的元组
    """
    assert sum(split_ratio) == 1.0, "划分比例总和必须为1"
    
    for node_type in data.node_types:
        if hasattr(data[node_type], 'label'):
            num_nodes = data[node_type].num_nodes
            labels = data[node_type].label
            
            # 生成随机排列
            perm = torch.randperm(num_nodes)
            
            # 计算划分点
            train_end = int(split_ratio[0] * num_nodes)
            val_end = train_end + int(split_ratio[1] * num_nodes)
            
            # 创建掩码 - 确保是布尔类型
            data[node_type].train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data[node_type].val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            data[node_type].test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            data[node_type].train_mask[perm[:train_end]] = True
            data[node_type].val_mask[perm[train_end:val_end]] = True
            data[node_type].test_mask[perm[val_end:]] = True
            
            # 检查每个集合中至少每个类别有一个样本
            for split in ['train', 'val', 'test']:
                mask = getattr(data[node_type], f"{split}_mask")
                # 确保labels是tensor且mask是布尔型
                if isinstance(labels, torch.Tensor) and mask.dtype == torch.bool:
                    unique_labels_in_split = torch.unique(labels[mask])
                    unique_labels_total = torch.unique(labels)
                    if len(unique_labels_in_split) < len(unique_labels_total):
                        print(f"警告: {node_type}的{split}集缺少某些类别样本")
    
    return data

hetero_data = split_hetero_data(data)
print(hetero_data)

# 检查划分结果
for node_type in hetero_data.node_types:
    if hasattr(hetero_data[node_type], 'label'):
        print(f"\n{node_type}划分结果:")
        print(f"总节点数: {hetero_data[node_type].num_nodes}")
        print(f"训练集: {hetero_data[node_type].train_mask.sum().item()}")
        print(f"测试集: {hetero_data[node_type].test_mask.sum().item()}")


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
        labels = torch.tensor(data[node_type].label)
        loss = criterion(out[mask], labels[mask])
        losses.append(loss)
    
    # 总损失
    total_loss = sum(losses)
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

def test(model, data):
    model.eval()
    out_dict = model(data.x_dict, data.edge_index_dict)
    
    metrics = {}
    for node_type in data.node_types:
        mask = data[node_type].test_mask  
        pred = out_dict[node_type][mask].argmax(dim=1)
        labels = torch.tensor(data[node_type].label)
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

# 初始化模型
model = HeteroGNN(hidden_channels=64, out_channels=2, num_heads=4)  
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
epochs = 200
# 训练循环
for epoch in range(epochs):
    loss = train(model, hetero_data, optimizer, criterion)
    if epoch % 10 == 0:
        metrics = test(model, hetero_data)
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        print(f'Metrics: {metrics}')



