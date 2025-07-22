import torch
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, SAGEConv, GATConv,
    RGCNConv, HGTConv,
    HeteroConv, Linear
)
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, average_precision_score
)
from torch_geometric.utils import to_undirected
import numpy as np

from torch_sparse import spspmm
from torch_geometric.nn.conv.gcn_conv import gcn_norm


# -------------------------------
# 模型定义：GCN/SAGE/GAT/HGT/RGCN
# -------------------------------
class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels, model_type='sage'):
        super().__init__()
        if model_type == 'sage':
            conv_class = SAGEConv
        elif model_type == 'gcn':
            conv_class = GCNConv
        elif model_type == 'gat':
            conv_class = lambda in_channels, out_channels: GATConv(in_channels, out_channels, heads=1, add_self_loops=False)
        else:
            raise ValueError("model_type must be 'sage', 'gcn', or 'gat'")

        self.conv1 = HeteroConv({
            edge_type: conv_class(
                (in_channels_dict[edge_type[0]], in_channels_dict[edge_type[2]]),
                hidden_channels
            )
            for edge_type in metadata[1] if edge_type[0] in in_channels_dict and edge_type[2] in in_channels_dict
        }, aggr='sum')

        self.conv2 = HeteroConv({
            edge_type: conv_class(hidden_channels, hidden_channels)
            for edge_type in metadata[1] if edge_type[0] in in_channels_dict and edge_type[2] in in_channels_dict
        }, aggr='sum')

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        # 目标节点类型为 'customer'
        return self.lin(x_dict['customer'])


class RGCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations=num_relations)

    def forward(self, x, edge_index, edge_type):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = self.conv2(x, edge_index, edge_type)
        return x
    

class HGT(torch.nn.Module):
    def __init__(self, metadata, in_channels_dict, hidden_channels, out_channels, num_heads=2):
        super().__init__()
        self.hgt1 = HGTConv(in_channels_dict, hidden_channels, metadata, heads=num_heads)
        self.hgt2 = HGTConv(hidden_channels, hidden_channels, metadata, heads=num_heads)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.hgt1(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict = self.hgt2(x_dict, edge_index_dict)
        return self.lin(x_dict['customer'])


class MAGNN(torch.nn.Module):
    def __init__(self, metapaths_dict, in_channels_dict, hidden_channels, out_channels):
        super().__init__()
        self.metapaths_dict = metapaths_dict
        self.lins = torch.nn.ModuleDict()
        for node_type, in_channels in in_channels_dict.items():
            self.lins[node_type] = Linear(in_channels, hidden_channels)
        self.semantic_attention = Linear(hidden_channels, 1, bias=False)
        self.out_lin = Linear(hidden_channels, out_channels)
        self._adj_cache = {}

    def _get_metapath_adj(self, edge_index_dict, metapath, num_nodes_dict, device):
        metapath_key = str(metapath)
        if metapath_key in self._adj_cache:
            return self._adj_cache[metapath_key]

        start_edge_type = metapath[0]
        current_adj = edge_index_dict[start_edge_type]
        current_src_type, _, current_dst_type = start_edge_type

        for i in range(1, len(metapath)):
            next_edge_type = metapath[i]
            next_src_type, _, next_dst_type = next_edge_type
            assert current_dst_type == next_src_type
            next_adj = edge_index_dict[next_edge_type]
            
            val_curr = torch.ones(current_adj.size(1), device=device)
            val_next = torch.ones(next_adj.size(1), device=device)
            
            new_adj_index, _ = spspmm(
                current_adj, val_curr, next_adj, val_next,
                num_nodes_dict[current_src_type],
                num_nodes_dict[current_dst_type],
                num_nodes_dict[next_dst_type]
            )
            current_adj, current_dst_type = new_adj_index, next_dst_type

        self._adj_cache[metapath_key] = current_adj
        return current_adj

    def forward(self, x_dict, edge_index_dict):
        device = list(x_dict.values())[0].device
        num_nodes_dict = {ntype: x.size(0) for ntype, x in x_dict.items()}

        h_dict = {node_type: self.lins[node_type](x).relu() 
                  for node_type, x in x_dict.items()}
        
        target_node_type = 'customer' # 我们的目标节点是 customer
        
        metapath_embs = []
        for metapath in self.metapaths_dict.values():
            adj = self._get_metapath_adj(edge_index_dict, metapath, num_nodes_dict, device)
            adj_norm_index, adj_norm_value = gcn_norm(adj, num_nodes=num_nodes_dict[target_node_type], add_self_loops=True)
            
            end_node_type = metapath[-1][-1]
            h_end_nodes = h_dict[end_node_type]

            h_aggregated = torch.sparse.mm(
                torch.sparse_coo_tensor(adj_norm_index, adj_norm_value), h_end_nodes
            )
            metapath_embs.append(h_aggregated)
        
        stacked_embs = torch.stack(metapath_embs, dim=1)
        m = torch.tanh(stacked_embs)
        alpha = self.semantic_attention(m).squeeze(-1)
        alpha = F.softmax(alpha, dim=1)
        final_emb = (alpha.unsqueeze(-1) * stacked_embs).sum(dim=1)
        
        return self.out_lin(final_emb)

# -------------------------------
# 辅助函数：计算所有评估指标 (保持不变)
# -------------------------------
def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['Accuracy']  = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall']    = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1 Score']  = f1_score(y_true, y_pred, zero_division=0)
    metrics['AUPRC']     = average_precision_score(y_true, y_prob)
    return metrics


# -------------------------------
# 训练 + 验证 + 测试 主函数 (已适配 MAGNN 和您的数据)
# -------------------------------
def train_and_evaluate(data, model_type='sage', hidden_channels=64, epochs=30, lr=0.005, verbose_interval=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)

    # 从数据中动态获取输入维度和输出类别数
    in_channels_dict = {node_type: data[node_type].x.size(1) for node_type in data.node_types}
    out_channels = int(data['customer'].y.max().item()) + 1

    if model_type == 'magnn':
        # --- MAGNN 的元路径定义 ---
        # 根据您的数据结构，定义 'Customer -> Fund -> Customer' 元路径
        metapaths_dict = {
            'CFC': [('customer', 'invests', 'fund'), ('fund', 'rev_invests', 'customer')],
        }
        # 如果有其他节点类型和关系，可以添加更多元路径
        # 例如 'Customer -> buys -> Product <- bought_by <- Customer'
        # 'CPC': [('customer', 'buys', 'product'), ('product', 'rev_buys', 'customer')]
        
        model = MAGNN(metapaths_dict, in_channels_dict, hidden_channels, out_channels).to(device)
    
    elif model_type == 'rgcn':
        # RGCN的特殊处理
        # 注意：RGCN的简单实现可能无法充分利用异构信息
        edge_index = data['customer', 'invests', 'fund'].edge_index
        edge_index = to_undirected(edge_index)
        x = data['customer'].x
        edge_type_tensor = torch.zeros(edge_index.shape[1], dtype=torch.long, device=edge_index.device)
        model = RGCN(x.size(1), hidden_channels, out_channels, num_relations=1).to(device)
    elif model_type == 'hgt':
        model = HGT(data.metadata(), in_channels_dict, hidden_channels, out_channels).to(device)
    else: # GCN, GAT, GraphSAGE
        model = HeteroGNN(data.metadata(), in_channels_dict, hidden_channels, out_channels, model_type).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    print(f"\n🔧 Start training with model: {model_type.upper()}")

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        if model_type == 'rgcn':
            out = model(x, edge_index, edge_type_tensor)
        else:
            out = model(data.x_dict, data.edge_index_dict)
        
        loss = F.cross_entropy(out[data['customer'].train_mask], data['customer'].y[data['customer'].train_mask])
        loss.backward()
        optimizer.step()

        if epoch % verbose_interval == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                if model_type == 'rgcn':
                    out = model(x, edge_index, edge_type_tensor)
                else:
                    out = model(data.x_dict, data.edge_index_dict)

                probs = F.softmax(out, dim=1)
                pred = probs.argmax(dim=1)
                val_mask = data['customer'].val_mask
                val_true = data['customer'].y[val_mask].cpu().numpy()
                val_pred = pred[val_mask].cpu().numpy()
                val_prob = probs[val_mask][:, 1].cpu().numpy() # 假设正类为1
                val_metrics = calculate_metrics(val_true, val_pred, val_prob)

                print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val Acc: {val_metrics['Accuracy']:.4f} | Val F1: {val_metrics['F1 Score']:.4f} | Val AUPRC: {val_metrics['AUPRC']:.4f}")

    # --- 最终在测试集上评估 ---
    model.eval()
    with torch.no_grad():
        if model_type == 'rgcn':
            out = model(x, edge_index, edge_type_tensor)
        else:
            out = model(data.x_dict, data.edge_index_dict)

        probs = F.softmax(out, dim=1)
        pred = probs.argmax(dim=1)

        test_mask = data['customer'].test_mask
        test_true = data['customer'].y[test_mask].cpu().numpy()
        test_pred = pred[test_mask].cpu().numpy()
        test_prob = probs[test_mask][:, 1].cpu().numpy()
        test_metrics = calculate_metrics(test_true, test_pred, test_prob)

    print("\n===== 测试集指标 =====")
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
    
    print("图数据信息:")
    print(data)
    print("\n节点特征维度:")
    print({node_type: node_store.x.shape for node_type, node_store in data.node_items()})
    print("\n边信息:")
    for edge_type, edge_store in data.edge_items():
        print(f"  - {edge_type}: edge_index={edge_store.edge_index.shape}"
              f"{', edge_attr=' + str(edge_store.edge_attr.shape) if hasattr(edge_store, 'edge_attr') else ''}")

    # # --- 训练并评估 GraphSAGE ---
    sage_model, sage_metrics = train_and_evaluate(
        data, model_type='sage', epochs=100, lr=0.001
    )

    # --- 训练并评估 GAT ---
    gat_model, gat_metrics = train_and_evaluate(
        data, model_type='gat', epochs=100, lr=0.001, verbose_interval=10
    )
    print("\n=== HGT ===")
    train_and_evaluate(data, model_type='hgt', epochs=200, lr=0.005, verbose_interval=10)

    print("\n=== RGCN ===")
    train_and_evaluate(data, model_type='rgcn', epochs=200, lr=0.005, verbose_interval=20)

    # --- 训练并评估 MAGNN ---
    print("\n=== MAGNN ===")
    train_and_evaluate(
        data, 
        model_type='magnn', 
        epochs=100, 
        lr=0.001, 
        hidden_channels=64, 
        verbose_interval=10
    )