import torch
from torch_geometric.data import Data, HeteroData
from makegraph import build_heterogeneous_graph
import pandas as pd
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_auc_score, f1_score
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HGTConv, to_hetero, Linear, GINEConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch_geometric.typing import Adj, EdgeType, OptTensor
from torch_geometric.utils import scatter
from sklearn.model_selection import train_test_split
import numpy as np

class GNNEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels, data, num_heads=4):
        super().__init__()
        
        self.node_types = data.node_types
        self.user_lin = torch.nn.Linear(101, 32)
        self.item_lin = torch.nn.Linear(1, 32)
        
        self.conv1 = GATv2Conv((32, 32), 32, heads=4, edge_dim=3)
        self.conv2 = GATv2Conv((32*4, 32), 32, heads=4, edge_dim=3)

        self.projection_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, data):
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict
        edge_attr_dict = data.edge_attr_dict

        user_x = self.user_lin(data['customer'].x)
        item_x = self.item_lin(data['fund'].x)
        edge_attr = data['customer', 'transaction', 'fund'].edge_attr
        edge_index = data['customer', 'transaction', 'fund'].edge_index

        item_x = torch.relu(self.conv1((user_x, item_x), edge_index, edge_attr=edge_attr))
        user_x = self.conv2((item_x, user_x), data['customer', 'transaction', 'fund'].edge_index.flip([0]))

        z_customer = self.projection_head(user_x)
        
        return user_x, z_customer

class GraphAnomalyDetectionModel(nn.Module):
    def __init__(self, data, hidden_channels=128, out_channels=128):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, out_channels, data)
        self.classifier = AnomalyClassifier(128)
        
    def forward(self, data):
        embeddings, _ = self.encoder(data)
        anomaly_scores = self.classifier(embeddings)
        return anomaly_scores
    
    def loss(self, anomaly_scores, labels):
        return F.binary_cross_entropy(anomaly_scores.squeeze(), labels.float())

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

def split_data(data, test_size=0.2, val_size=0.1, random_state=42):
    """划分数据集为训练、验证和测试集"""
    # 获取所有客户节点的索引
    num_customers = data['customer'].num_nodes
    indices = np.arange(num_customers)
    
    # 首先划分训练+验证和测试集
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state,
        stratify=data['customer'].y.cpu().numpy() if hasattr(data['customer'], 'y') else None
    )
    
    # 然后从训练+验证集中划分验证集
    val_ratio = val_size / (1 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_ratio, random_state=random_state,
        stratify=data['customer'].y[train_val_idx].cpu().numpy() if hasattr(data['customer'], 'y') else None
    )
    
    # 创建掩码
    train_mask = torch.zeros(num_customers, dtype=torch.bool)
    val_mask = torch.zeros(num_customers, dtype=torch.bool)
    test_mask = torch.zeros(num_customers, dtype=torch.bool)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    # 将掩码添加到数据中
    data['customer'].train_mask = train_mask
    data['customer'].val_mask = val_mask
    data['customer'].test_mask = test_mask
    
    return data

def train(model, optimizer, data, epochs=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    
    best_auc = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 只使用训练集节点
        train_mask = data['customer'].train_mask
        anomaly_scores = model(data)
        
        # 计算训练损失
        train_scores = anomaly_scores[train_mask]
        train_labels = data['customer'].y[train_mask]
        loss = model.loss(train_scores, train_labels)
        
        loss.backward()
        optimizer.step()
        
        # 验证集评估
        model.eval()
        with torch.no_grad():
            val_mask = data['customer'].val_mask
            val_scores = anomaly_scores[val_mask]
            val_labels = data['customer'].y[val_mask]
            
            val_auc = roc_auc_score(val_labels.cpu().numpy(), 
                                  val_scores.detach().cpu().numpy())
            
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), 'best_model_gnn.pt')
        
        # 打印训练和验证指标
        train_auc = roc_auc_score(train_labels.detach().cpu().numpy(), 
                                 train_scores.detach().cpu().numpy())
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}, Train AUC: {train_auc:.4f}, Val AUC: {val_auc:.4f}')
    
    print(f'Training completed. Best Val AUC: {best_auc:.4f}')


def test(model, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.load_state_dict(torch.load('best_model_gnn.pt'))
    model.eval()
    
    data = data.to(device)
    with torch.no_grad():
        test_mask = data['customer'].test_mask
        anomaly_scores = model(data)
        test_scores = anomaly_scores[test_mask]
        test_labels = data['customer'].y[test_mask]
        
        # 计算评估指标
        auc = roc_auc_score(test_labels.cpu().numpy(), 
                           test_scores.detach().cpu().numpy())
        predictions = (test_scores > 0.5).float()
        f1 = f1_score(test_labels.cpu().numpy(), 
                     predictions.detach().cpu().numpy())
        
        print(f'Test AUC: {auc:.4f}, F1 Score: {f1:.4f}')

def main():
    # 加载图数据集
    data = torch.load('financial_graph.pt')
    
    # 划分数据集
    data = split_data(data, test_size=0.2, val_size=0.1)
    
    # 初始化模型
    model = GraphAnomalyDetectionModel(
        data=data,
        hidden_channels=128,
        out_channels=64
    )
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # 训练模型
    train(model, optimizer, data, epochs=100)
    
    # 测试模型
    test(model, data)

if __name__ == '__main__':
    main()