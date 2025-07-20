import torch
from torch_geometric.data import Data, HeteroData
from makegraph import build_heterogeneous_graph
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HGTConv, to_hetero,Linear,GINEConv,GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj, EdgeType, OptTensor
from torch_geometric.utils import scatter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Linear, HeteroConv, GATConv, SAGEConv
from torch_geometric.nn.dense.linear import Linear

class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, data, num_heads=4):
        super().__init__()
        
        # 1. 首先获取每个节点类型的特征维度
        self.node_types = data.node_types
        self.user_lin = torch.nn.Linear(101, 32)
        self.item_lin = torch.nn.Linear(1, 32)
        
        self.conv1 = GATv2Conv((32, 32), 32, heads=4,edge_dim=3)
        self.conv2 = GATv2Conv((32*4, 32), 32, heads=4,edge_dim=3)

        # 4. 修改投影头以处理所有节点类型
        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, data):
        # data.x_dict, data.edge_index_dict
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict
        # print(edge_attr_dict.keys())

        # 1. 初始特征转换
        user_x = self.user_lin(data['customer'].x)
        item_x = self.item_lin(data['fund'].x)
        edge_attr = data['customer', 'transaction', 'fund'].edge_attr
        edge_index = data['customer', 'transaction', 'fund'].edge_index

        item_x = torch.relu(self.conv1((user_x, item_x), edge_index,edge_attr=edge_attr))

        # 第二层GAT
        user_x = self.conv2((item_x,user_x), data['customer', 'transaction', 'fund'].edge_index.flip([0]))

        z_customer = self.projection_head(user_x)
        
        # 5. 返回节点类型的嵌入和投影
        return user_x, z_customer


    
class GraphAnomalyDetectionModel(nn.Module):
    def __init__(self, data, hidden_channels=128, out_channels=128):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, out_channels, data)
        self.classifier = AnomalyClassifier(128)
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
    

def create_augmented_view(data):
    # 这里实现图数据的增强策略
    # 例如：随机边丢弃、特征掩码等
    
    # 随机边丢弃（同步处理边属性）
    augmented_graph = data.clone()
    for edge_type in data.edge_types:
        # 获取边索引和边属性
        edge_index = augmented_graph[edge_type].edge_index
        edge_attr = augmented_graph[edge_type].edge_attr  # 假设边属性存在
        
        num_edges = edge_index.size(1)
        keep_prob = torch.rand(num_edges) > 0.1  # 保留90%的边
        
        # 同步丢弃边索引和边属性
        augmented_graph[edge_type].edge_index = edge_index[:, keep_prob]
        if edge_attr is not None:  # 如果该边类型有属性
            augmented_graph[edge_type].edge_attr = edge_attr[keep_prob]
        
    
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



def train(model, optimizer, data, epochs=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 将异构图转换为适合mini-batch训练的形式
    # 这里简化处理，实际可能需要更复杂的数据划分
    train_loader = DataLoader([data], batch_size=batch_size, shuffle=True)
    
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


            
            # 有标签数据
            labels = data['customer'].y
            print(labels[:5])
            print(anomaly_scores1[:5])
            
            # 计算损失
            loss = model.loss(projections1, projections2, labels, anomaly_scores1)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证过程
        model.eval()
        with torch.no_grad():
            anomaly_scores, _ = model(data)
            auc = roc_auc_score(labels.cpu().numpy(), anomaly_scores.cpu().numpy())
            
            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), 'best_model_gnn.pt')
        
        print(f'Epoch: {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, AUC: {auc:.4f}')
    
    print(f'Training completed. Best AUC: {best_auc:.4f}')


def main():

    # 加载图数据集
    data =  torch.load('financial_graph.pt')
    # 初始化模型
    model = GraphAnomalyDetectionModel(
        data=data,
        hidden_channels=128,
        out_channels=64
    )
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练模型
    train(model, optimizer, data, epochs=2)
    
    # 测试模型
    test_model(model, data)

def test_model(model, test_graph):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load('best_model_gnn.pt'))
    model.eval()
    
    with torch.no_grad():
        test_graph = test_graph.to(device)
        anomaly_scores, _ = model()
        labels = test_graph['node'].y
        
        # 计算评估指标
        auc = roc_auc_score(labels.cpu().numpy(), anomaly_scores.cpu().numpy())
        predictions = (anomaly_scores > 0.5).float()
        f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy())
        
        print(f'Test AUC: {auc:.4f}, F1 Score: {f1:.4f}')

if __name__ == '__main__':
    main()