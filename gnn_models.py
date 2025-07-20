import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, HeteroConv, Linear


# -------------------------------
# 模型定义：异构图 GCN / SAGE / GAT
# -------------------------------
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels, model_type='sage'):
        super().__init__()

        # 选择卷积层类型
        if model_type == 'sage':
            conv_class = SAGEConv
        elif model_type == 'gcn':
            conv_class = GCNConv
        elif model_type == 'gat':
            conv_class = lambda in_channels, out_channels: GATConv(in_channels, out_channels, heads=1, add_self_loops=False)
        else:
            raise ValueError("model_type must be 'sage', 'gcn', or 'gat'")

        # 第一层 HeteroConv
        self.conv1 = HeteroConv({
            edge_type: conv_class(
                (in_channels_dict[edge_type[0]], in_channels_dict[edge_type[2]]),
                hidden_channels
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        # 第二层 HeteroConv
        self.conv2 = HeteroConv({
            edge_type: conv_class(
                (hidden_channels, hidden_channels),
                hidden_channels
            )
            for edge_type in metadata[1]
        }, aggr='sum')

        # 输出层（仅对 customer 节点）
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        return self.lin(x_dict['customer'])


# -------------------------------
# 训练 + 验证 + 测试 主函数
# -------------------------------
def train_and_evaluate(data, model_type='sage', hidden_channels=64, epochs=30, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 构造输入维度字典
    in_channels_dict = {
        node_type: data[node_type].x.size(1)
        for node_type in data.node_types
    }

    # 类别数
    out_channels = int(data['customer'].y.max().item()) + 1

    # 初始化模型
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

        # 验证精度
        model.eval()
        with torch.no_grad():
            pred = out.argmax(dim=1)
            val_pred = pred[data['customer'].val_mask]
            val_true = data['customer'].y[data['customer'].val_mask]
            val_acc = (val_pred == val_true).float().mean().item()
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_acc:.4f}")

    # 测试精度
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        pred = out.argmax(dim=1)
        test_pred = pred[data['customer'].test_mask]
        test_true = data['customer'].y[data['customer'].test_mask]
        test_acc = (test_pred == test_true).float().mean().item()

    print(f"\n✅ Final Test Accuracy: {test_acc:.4f}")
    return model, pred.cpu(), test_acc



import torch

# 加载图数据
data = torch.load('data_new/hetero_graph.pt')

# 训练 GraphSAGE
# model, pred, test_acc = train_and_evaluate(data, model_type='sage', epochs=100)

# 训练 GAT：
model, pred, test_acc = train_and_evaluate(data, model_type='gat', epochs=500)
