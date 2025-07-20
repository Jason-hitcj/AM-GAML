import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, Linear, to_hetero
from torch_geometric.data import HeteroData

class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels=64, out_channels=2, num_heads=4, dropout=None):
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