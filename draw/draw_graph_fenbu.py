import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import os

# 确保img目录存在
os.makedirs('img', exist_ok=True)

# 加载图数据
data = torch.load('data/homogeneous_graph.pt')

# 将PyG数据转换为NetworkX图
G = to_networkx(data, to_undirected=True)

# 定义颜色映射 (正常节点为蓝色，异常节点为红色)
color_map = {0: 'blue', 1: 'red'}

# 为每个分割集创建可视化
masks = {
    'train': data.train_mask,
    'val': data.val_mask,
    'test': data.test_mask
}

for name, mask in masks.items():
    # 获取当前分割集的节点索引
    node_indices = mask.nonzero().flatten().tolist()
    
    # 创建子图 (只包含当前分割集的节点)
    subgraph = G.subgraph(node_indices)
    
    # 获取节点颜色
    node_colors = [color_map[data.y[i].item()] for i in node_indices]
    
    # 绘制图形
    plt.figure(figsize=(12, 12))
    
    # 使用力导向布局 - 调整参数使图更集中
    pos = nx.spring_layout(subgraph, seed=42, k=0.1, iterations=100)  # 减小k值使图更集中
    
    # 绘制边 - 使用黑色
    nx.draw_networkx_edges(
        subgraph, pos, 
        edge_color='black',  # 黑色边线
        width=2,          # 稍宽的边线
        alpha=1           # 适中的透明度
    )
    
    # 绘制节点
    nx.draw_networkx_nodes(
        subgraph, pos, 
        node_color=node_colors, 
        node_size=30,
        alpha=0.8
    )
    
    # 添加图例
    plt.scatter([], [], c='blue', label='Normal', s=100)
    plt.scatter([], [], c='red', label='Anomaly', s=100)
    plt.legend(loc='upper right')
    
    plt.title(f'{name.capitalize()} Set Graph Visualization (Nodes: {len(node_indices)})')
    plt.axis('off')
    
    # 保存图像
    plt.savefig(f'img/{name}_graph_compact_black.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f'Saved {name} set visualization to img/{name}_graph_compact_black.png')