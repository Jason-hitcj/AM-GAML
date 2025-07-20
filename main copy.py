import torch
from torch_geometric.data import Data, HeteroData
from makegraph import build_heterogeneous_graph
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HGTConv, to_hetero,Linear,GINEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj, EdgeType, OptTensor
from torch_geometric.utils import scatter

# class GNNEncoder(nn.Module):
#     def __init__(self, hidden_channels, out_channels, data, num_heads=4):
#         super().__init__()
        
#         # 1. 首先获取每个节点类型的特征维度
#         self.node_types = data.node_types
#         self.initial_lin_dict = nn.ModuleDict()
#         self.post_lin_dict = nn.ModuleDict()
        
#         # 2. 为每个节点类型创建输入线性层
#         for node_type in self.node_types:
#             # 获取该节点类型的特征维度
#             feat_dim = data[node_type].x.size(1) if hasattr(data[node_type], 'x') else 1
#             self.initial_lin_dict[node_type] = Linear(feat_dim, hidden_channels)
#             self.post_lin_dict[node_type] = Linear(hidden_channels, hidden_channels)
        
#         # 3. 初始化HGT卷积层
#         self.conv1 = HGTConv(
#             in_channels=hidden_channels,
#             out_channels=hidden_channels,
#             metadata=data.metadata(),
#             heads=num_heads
#         )
        
#         self.conv2 = HGTConv(
#             in_channels=hidden_channels,
#             out_channels=out_channels,
#             metadata=data.metadata(),
#             heads=num_heads
#         )
        
#         # 4. 修改投影头以处理所有节点类型
#         self.projection_head = nn.ModuleDict({
#             node_type: nn.Sequential(
#                 nn.Linear(out_channels, out_channels),
#                 nn.ReLU(),
#                 nn.Linear(out_channels, out_channels)
#             ) for node_type in self.node_types
#         })

#     def forward(self, data):
#         # data.x_dict, data.edge_index_dict
#         x_dict = data.x_dict
#         edge_index_dict = data.edge_index_dict

#         # 1. 初始特征转换
#         x_dict = {
#             node_type: self.initial_lin_dict[node_type](x).relu_()
#             for node_type, x in x_dict.items()
#         }
        
#         # 2. 第一层HGT卷积
#         # x_dict = self.conv1(x_dict, edge_index_dict,edge_attr_dict=edge_attr_dict)
#         x_dict = self.conv1(x_dict, edge_index_dict)
#         x_dict = {k: F.leaky_relu(v) for k, v in x_dict.items()}
        
#         # 3. 第二层HGT卷积
#         # x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
#         x_dict = self.conv2(x_dict, edge_index_dict)
#         # 4. 对所有节点类型计算投影
#         projections = {
#             node_type: self.projection_head[node_type](x_dict[node_type])
#             for node_type in self.node_types
#         }
        
#         # 5. 返回所有节点类型的嵌入和投影
#         return x_dict['customer'], projections['customer']


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv, GATConv, SAGEConv
from torch_geometric.nn.dense.linear import Linear

def calculate_edge_weights(hetero_data):
    """
    计算异构图中各边类型的综合权重
    
    参数:
        hetero_data: 异构图数据对象，包含边属性:
            - edge_attr[:,0]: 交易金额
            - edge_attr[:,1]: 前5日交易额
            - edge_attr[:,2]: 是否90日突然启用(1=无,10=有)
    
    返回:
        edge_weight_dict: 字典 {边类型: 对应边的权重tensor}
    """
    edge_weight_dict = {}
    
    for edge_type in hetero_data.edge_types:
        edge_attr = hetero_data[edge_type].edge_attr
        
        # 1. 提取三个特征
        amount = edge_attr[:, 0]          # 交易金额
        last5_amount = edge_attr[:, 1]    # 前5日交易额
        sudden_activate = edge_attr[:, 2] # 是否突然启用
        
        # 2. 对各特征进行归一化处理
        # 金额和前5日交易额取对数后标准化（处理长尾分布）
        log_amount = torch.log(amount + 1e-6)
        norm_amount = (log_amount - log_amount.min()) / (log_amount.max() - log_amount.min() + 1e-6)
        
        log_last5 = torch.log(last5_amount + 1e-6)
        norm_last5 = (log_last5 - log_last5.min()) / (log_last5.max() - log_last5.min() + 1e-6)
        
        # 突然启用特征转换为权重系数（1→0.3, 10→1.0）
        activate_weight = sudden_activate / 10.0  # 映射到[0.1, 1.0]
        
        # 3. 组合权重（可调整各部分的权重系数）
        combined_weights = (
            0.5 * norm_amount +      # 当前金额权重50%
            0.3 * norm_last5 +       # 历史金额权重30% 
            0.2 * activate_weight    # 突然启用权重20%
        )
        
        # 4. 最终归一化到[0.1, 1.0]范围（避免权重为0）
        final_weights = 0.1 + 0.9 * (
            (combined_weights - combined_weights.min()) / 
            (combined_weights.max() - combined_weights.min() + 1e-6))
        
        edge_weight_dict[edge_type] = final_weights
    
    return edge_weight_dict

class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, data, num_heads=4):
        super().__init__()
        
        # 1. 首先获取每个节点类型的特征维度
        self.node_types = data.node_types
        self.initial_lin_dict = nn.ModuleDict()
        self.post_lin_dict = nn.ModuleDict()
        
        # 2. 为每个节点类型创建输入线性层
        for node_type in self.node_types:
            # 获取该节点类型的特征维度
            feat_dim = data[node_type].x.size(1) if hasattr(data[node_type], 'x') else 1
            self.initial_lin_dict[node_type] = Linear(feat_dim, hidden_channels)
            self.post_lin_dict[node_type] = Linear(hidden_channels, hidden_channels)
        
        # 3. 初始化HGT卷积层
        self.conv1 = HGTConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            metadata=data.metadata(),
            heads=num_heads
        )
        
        self.conv2 = HGTConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            metadata=data.metadata(),
            heads=num_heads
        )
        
        # 4. 修改投影头以处理所有节点类型
        self.projection_head = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            ) for node_type in self.node_types
        })

    def forward(self, data):
        # data.x_dict, data.edge_index_dict
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # 1. 初始特征转换
        x_dict = {
            node_type: self.initial_lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        
        # 2. 第一层HGT卷积
        # x_dict = self.conv1(x_dict, edge_index_dict,edge_attr_dict=edge_attr_dict)
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: F.leaky_relu(v) for k, v in x_dict.items()}
        
        # 3. 第二层HGT卷积
        # x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        x_dict = self.conv2(x_dict, edge_index_dict)
        # 4. 对所有节点类型计算投影
        projections = {
            node_type: self.projection_head[node_type](x_dict[node_type])
            for node_type in self.node_types
        }
        
        # 5. 返回所有节点类型的嵌入和投影
        return x_dict['customer'], projections['customer']
    


    
class GraphAnomalyDetectionModel(nn.Module):
    def __init__(self, data, hidden_channels=128, out_channels=64):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, out_channels, data)
        self.classifier = AnomalyClassifier(out_channels)
        self.contrastive_loss = ContrastiveLoss()
        
    def forward(self, data):
        # embeddings, projections = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        embeddings, projections = self.encoder(data)
        anomaly_scores = self.classifier(embeddings)
        return anomaly_scores, projections
    
    def loss(self, projections1, projections2, labels, anomaly_scores):
        # 对比损失
        contrast_loss = self.contrastive_loss(projections1, projections2, labels)
        # 异常分类损失
        class_loss = F.binary_cross_entropy(anomaly_scores.squeeze(), labels.float())
        # 总损失
        total_loss = contrast_loss + class_loss
        
        return total_loss
    

def create_augmented_view(hetero_graph):
    # 这里实现图数据的增强策略
    # 例如：随机边丢弃、特征掩码等
    
    # 示例：随机边丢弃
    augmented_graph = hetero_graph.clone()
    for edge_type in hetero_graph.edge_types:
        edge_index = augmented_graph[edge_type].edge_index
        num_edges = edge_index.size(1)
        keep_prob = torch.rand(num_edges) > 0.1  # 10%的丢弃概率
        augmented_graph[edge_type].edge_index = edge_index[:, keep_prob]
    
    return augmented_graph


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, projections1, projections2, labels):
        # 归一化投影向量
        projections1 = F.normalize(projections1, dim=1)
        projections2 = F.normalize(projections2, dim=1)
        
        # 计算相似度矩阵
        logits = torch.mm(projections1, projections2.T) / self.temperature
        
        # 创建目标：正样本对在对角线上
        batch_size = projections1.size(0)
        targets = torch.arange(batch_size).to(logits.device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, targets)
        
        return loss
    
class AnomalyClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, embeddings):
        return self.mlp(embeddings)



def train(model, optimizer, hetero_graph, epochs=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 将异构图转换为适合mini-batch训练的形式
    # 这里简化处理，实际可能需要更复杂的数据划分
    train_loader = DataLoader([hetero_graph], batch_size=batch_size, shuffle=True)
    
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            data = data.to(device)
            
            # 创建增强视图
            augmented_data = create_augmented_view(data)
            augmented_data = augmented_data.to(device)
            
            optimizer.zero_grad()
            
            # 原始视图的前向传播
            # anomaly_scores1, projections1 = model(data.x_dict, data.edge_index_dict,data.edge_attr_dict)
            anomaly_scores1, projections1 = model(data)
            
            # 增强视图的前向传播
            # _, projections2 = model(augmented_data.x_dict, augmented_data.edge_index_dict,augmented_data.edge_attr_dict)
            _, projections2 = model(augmented_data)
            
            # 假设我们有标签数据
            labels = data['customer'].y  # 根据你的实际标签位置调整
            
            # 计算损失
            loss = model.loss(projections1, projections2, labels, anomaly_scores1)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证过程
        model.eval()
        with torch.no_grad():
            # 这里简化验证过程，实际应该使用独立的验证集
            anomaly_scores, _ = model(data)
            auc = roc_auc_score(labels.cpu().numpy(), anomaly_scores.cpu().numpy())
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), 'best_model.pt')
        
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, AUC: {auc:.4f}')
    
    print(f'Training completed. Best AUC: {best_auc:.4f}')


def main():

    # 加载图数据集
    hetero_graph =  torch.load('financial_graph.pt')
    print("构建的图数据结构:")
    print(hetero_graph)
    print("\n节点数量统计:", {k: v.x.size(0) for k, v in hetero_graph.node_items()})

    print("Node types in graph:", hetero_graph.node_types)
    print("--------------------------------")
    print("Edge types in graph:", hetero_graph.edge_types)
    print("--------------------------------")
    # 初始化模型
    model = GraphAnomalyDetectionModel(
        data=hetero_graph,
        hidden_channels=128,
        out_channels=64
    )
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练模型
    train(model, optimizer, hetero_graph, epochs=100)
    
    # 测试模型
    # test_model(model, hetero_graph)

def test_model(model, test_graph):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()
    
    with torch.no_grad():
        test_graph = test_graph.to(device)
        anomaly_scores, _ = model(test_graph.x_dict, test_graph.edge_index_dict)
        labels = test_graph['node'].y
        
        # 计算评估指标
        auc = roc_auc_score(labels.cpu().numpy(), anomaly_scores.cpu().numpy())
        predictions = (anomaly_scores > 0.5).float()
        f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())
        
        print(f'Test AUC: {auc:.4f}, F1 Score: {f1:.4f}')

if __name__ == '__main__':
    main()