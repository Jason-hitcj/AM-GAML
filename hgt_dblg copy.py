import os.path as osp

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.datasets import DBLP
from torch_geometric.nn import HGTConv, Linear,HeteroConv, SAGEConv
from torch_geometric.transforms import RandomNodeSplit


# 加载图数据集
data =  torch.load('financial_graph.pt')

print("\n节点数量统计:", {k: v.x.size(0) for k, v in data.node_items()})

print("Node types in graph:", data.node_types)
print("--------------------------------")
print("Edge types in graph:", data.edge_types)
print("--------------------------------")

import torch

num_customers = data['customer'].x.size(0)
indices = torch.randperm(num_customers)  # 随机打乱客户节点索引

# 按比例划分（训练60%，验证20%，测试20%）
train_end = int(num_customers * 0.6)
val_end = train_end + int(num_customers * 0.2)

# 初始化掩码
data['customer'].train_mask = torch.zeros(num_customers, dtype=torch.bool)
data['customer'].val_mask = torch.zeros(num_customers, dtype=torch.bool)
data['customer'].test_mask = torch.zeros(num_customers, dtype=torch.bool)

# 分配掩码
data['customer'].train_mask[indices[:train_end]] = True      # 训练集
data['customer'].val_mask[indices[train_end:val_end]] = True # 验证集
data['customer'].test_mask[indices[val_end:]] = True        # 测试集

print("data.x_dict 的键:", data.x_dict.keys())

print("构建的图数据结构:")
print(data.node_types)

def add_reverse_edge(data, src_type, rel_type, dst_type, edge_attr_transform=None):
    src, dst = data[(src_type, rel_type, dst_type)].edge_index
    edge_attr = data[(src_type, rel_type, dst_type)].edge_attr

    # 默认直接复制属性
    reversed_attr = edge_attr.clone()
    if edge_attr_transform:
        reversed_attr = edge_attr_transform(edge_attr)

    # 添加反向边
    rev_rel_type = f'rev_{rel_type}'
    data[(dst_type, rev_rel_type, src_type)].edge_index = torch.stack([dst, src], dim=0)
    data[(dst_type, rev_rel_type, src_type)].edge_attr = reversed_attr

add_reverse_edge(data, 'customer', 'transaction', 'fund')

print("构建的图数据结构:")
print(data)

class HGT(torch.nn.Module):
    def __init__(self, hidden_channels, metadata, edge_dim=5):
        super().__init__()

        # 输入线性映射
        self.lin_dict = nn.ModuleDict()
        for node_type in metadata[0]:
            if node_type == 'customer':
                self.lin_dict[node_type] = nn.Linear(101, hidden_channels)
            elif node_type == 'fund':
                self.lin_dict[node_type] = nn.Linear(1, hidden_channels)

        # HeteroConv 层（1层为例）
        self.conv = HeteroConv({
            ('customer', 'transaction', 'fund'): SAGEConv((-1, -1), hidden_channels),
            ('fund', 'rev_transaction', 'customer'): SAGEConv((-1, -1), hidden_channels),
        }, aggr='sum')

        self.out_lin = nn.Linear(hidden_channels, 2)  # 假设是2分类

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 预处理特征
        x_dict = {
            node_type: self.lin_dict[node_type](x).relu()
            for node_type, x in x_dict.items()
        }

        # Message Passing
        x_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict)

        # 只输出customer的分类结果
        return self.out_lin(x_dict['customer'])


model = HGT(hidden_channels=64, metadata=data.metadata())
if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch_geometric.is_xpu_available():
    device = torch.device('xpu')
else:
    device = torch.device('cpu')
data, model = data.to(device), model.to(device)

with torch.no_grad():  # Initialize lazy modules.
    out = model(data.x_dict, data.edge_index_dict,data.edge_attr_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)


def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)  # ✅ 修复
    mask = data['customer'].train_mask
    loss = F.cross_entropy(out[mask], data['customer'].y[mask])
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test():
    model.eval()
    pred = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict).argmax(dim=-1)  # ✅ 修复

    accs = []
    for split in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['customer'][split]
        acc = (pred[mask] == data['customer'].y[mask]).sum() / mask.sum()
        accs.append(float(acc))
    return accs


for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')