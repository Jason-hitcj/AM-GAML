import torch
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HGTConv, to_hetero, Linear, GINEConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj, EdgeType, OptTensor

class GINEConvWithEdgeAttr(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim):
        super().__init__()
        # GINEConv需要一个MLP来处理节点特征
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.conv = GINEConv(nn=self.mlp, train_eps=True)
        # 边特征编码器
        self.edge_encoder = nn.Linear(edge_dim, out_channels)
        
    def forward(self, x, edge_index, edge_attr):
        # 编码边特征
        edge_emb = self.edge_encoder(edge_attr)
        # 应用GINEConv
        return self.conv(x, edge_index, edge_emb)

class HeteroGINEConv(nn.Module):
    def __init__(self, in_channels, out_channels, edge_dim, metadata):
        super().__init__()
        # 为每种边类型创建GINE层
        self.convs = nn.ModuleDict()
        for edge_type in metadata[1]:
            self.convs['__'.join(edge_type)] = GINEConvWithEdgeAttr(
                in_channels, out_channels, edge_dim)
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        out_dict = {}
        # 收集所有目标节点的类型
        target_nodes = set()
        for edge_type in edge_index_dict.keys():
            target_nodes.add(edge_type[2])
        
        # 对每种目标节点类型初始化输出
        for node_type in target_nodes:
            out_dict[node_type] = []
        
        # 对每种边类型应用对应的GINE层
        for edge_type, edge_index in edge_index_dict.items():
            src_type, rel_type, dst_type = edge_type
            key = '__'.join(edge_type)
            
            # 获取源节点特征
            src_x = x_dict[src_type]
            # 获取边特征
            edge_attr = edge_attr_dict[edge_type]
            
            # 应用GINE卷积
            out = self.convs[key](src_x, edge_index, edge_attr)
            out_dict[dst_type].append(out)
        
        # 对每个目标节点类型的多源输入取平均
        for node_type in out_dict:
            if len(out_dict[node_type]) == 0:
                # 如果没有输入边，保持原始特征
                out_dict[node_type] = x_dict[node_type]
            else:
                # 平均所有来源的表示
                out_dict[node_type] = torch.mean(torch.stack(out_dict[node_type]), dim=0)
        
        return out_dict

class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, data, num_heads=4):
        super().__init__()
        self.node_types = data.node_types
        self.edge_types = data.edge_types
        
        # 1. 节点特征初始化投影
        self.initial_lin_dict = nn.ModuleDict()
        for node_type in self.node_types:
            feat_dim = data[node_type].x.size(1) if hasattr(data[node_type], 'x') else 1
            self.initial_lin_dict[node_type] = Linear(feat_dim, hidden_channels)
        
        # 2. 边特征维度检查（假设所有边类型有相同维度）
        sample_edge_type = self.edge_types[0]
        edge_dim = data[sample_edge_type].edge_attr.size(1) if hasattr(data[sample_edge_type], 'edge_attr') else 1
        
        # 3. 图卷积层
        self.conv1 = HeteroGINEConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            edge_dim=edge_dim,
            metadata=data.metadata()
        )
        
        self.conv2 = HeteroGINEConv(
            in_channels=hidden_channels,
            out_channels=out_channels,
            edge_dim=edge_dim,
            metadata=data.metadata()
        )
        
        # 4. 投影头
        self.projection_head = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(out_channels, out_channels),
                nn.ReLU(),
                nn.Linear(out_channels, out_channels)
            ) for node_type in self.node_types
        })

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 1. 初始特征转换
        x_dict = {
            node_type: self.initial_lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }
        
        # 2. 第一层GINE卷积
        x_dict = self.conv1(x_dict, edge_index_dict, edge_attr_dict)
        x_dict = {k: F.leaky_relu(v) for k, v in x_dict.items()}
        
        # 3. 第二层GINE卷积
        x_dict = self.conv2(x_dict, edge_index_dict, edge_attr_dict)
        
        # 4. 对所有节点类型计算投影
        projections = {
            node_type: self.projection_head[node_type](x_dict[node_type])
            for node_type in self.node_types
        }
        
        return x_dict['customer'], projections['customer']

class GraphAnomalyDetectionModel(nn.Module):
    def __init__(self, hetero_graph, hidden_channels=128, out_channels=64):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, out_channels, hetero_graph)
        self.classifier = AnomalyClassifier(out_channels)
        self.contrastive_loss = ContrastiveLoss()
        
    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        embeddings, projections = self.encoder(x_dict, edge_index_dict, edge_attr_dict)
        anomaly_scores = self.classifier(embeddings)
        return anomaly_scores, projections
    
    def loss(self, projections1, projections2, labels, anomaly_scores):
        contrast_loss = self.contrastive_loss(projections1, projections2, labels)
        class_loss = F.binary_cross_entropy(anomaly_scores.squeeze(), labels.float())
        return contrast_loss + class_loss



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
            # augmented_data = create_augmented_view(data)
            augmented_data = data
            augmented_data = augmented_data.to(device)
            
            optimizer.zero_grad()
            
            # 原始视图的前向传播
            anomaly_scores1, projections1 = model(data.x_dict, data.edge_index_dict,data.edge_attr_dict)
            
            # 增强视图的前向传播
            _, projections2 = model(augmented_data.x_dict, augmented_data.edge_index_dict,augmented_data.edge_attr_dict)
            
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
            anomaly_scores, _ = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
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
        hetero_graph=hetero_graph,
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