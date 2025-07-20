import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# -------------------- 第一部分：构建异构节点特征 --------------------
# 原始属性数据
raw_df = pd.read_csv('data/preprocessed_data_gnn.csv')
unique_df = raw_df.groupby('CUST_ID', as_index=False).first()
raw_df = unique_df

# LSTM特征
lstm_features_df = pd.read_csv('data/customer_node_features.csv')

# 加载预划分的ID（替换原有数据划分逻辑）
train_ids = pd.read_csv('data/train_ids.csv')['CUST_ID'].tolist()
val_ids = pd.read_csv('data/val_ids.csv')['CUST_ID'].tolist()
test_ids = pd.read_csv('data/test_ids.csv')['CUST_ID'].tolist()

# 创建划分掩码
lstm_features_df['split'] = 'test'
lstm_features_df.loc[lstm_features_df['CUST_ID'].isin(train_ids), 'split'] = 'train'
lstm_features_df.loc[lstm_features_df['CUST_ID'].isin(val_ids), 'split'] = 'val'



# 静态特征处理
static_features = ['AGE', 'GENDER', 'NET_CODE', 'RISK_LEV']
cat_cols = ['GENDER', 'NET_CODE']
numerical_cols = ['AGE', 'GENDER', 'RISK_LEV']

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_cats = ohe.fit_transform(raw_df[cat_cols])
encoded_numerical = raw_df[numerical_cols].values
static_feature_matrix = np.concatenate([encoded_numerical, encoded_cats], axis=1)

# 合并静态 + LSTM特征
lstm_matrix = lstm_features_df[[f'feature_{i}' for i in range(64)]].values
final_feature_matrix = np.concatenate([static_feature_matrix, lstm_matrix], axis=1)

# 创建异构图数据对象
data = HeteroData()
node_types = ['institution', 'person', 'product']
for node_type in node_types:
    mask = (raw_df['CUST_TYPE'] == node_types.index(node_type))
    data[node_type].x = torch.tensor(final_feature_matrix[mask], dtype=torch.float)
    data[node_type].cust_id = raw_df.loc[mask, 'CUST_ID'].tolist()
    cust_ids_masked = raw_df.loc[mask, 'CUST_ID']
    lstm_matched = lstm_features_df.set_index('CUST_ID').loc[cust_ids_masked]
    data[node_type].label = lstm_matched['label'].tolist()
    data[node_type].split = lstm_matched['split'].tolist()


# -------------------- 第二部分：基于SVM概率构建边 --------------------



# 仅使用训练集数据训练SVM和Scaler
train_mask = (lstm_features_df['split'] == 'train')
X_train = lstm_features_df.loc[train_mask, [f'feature_{i}' for i in range(64)]].values
y_train = lstm_features_df.loc[train_mask, 'label'].values

# 标准化器仅拟合训练数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 训练SVM（保持概率输出）
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_train_scaled, y_train)

# 对所有数据应用训练好的scaler（避免数据泄漏）
X_all_scaled = scaler.transform(lstm_features_df[[f'feature_{i}' for i in range(64)]].values)
probs = svm.predict_proba(X_all_scaled)  # [N, 2]


# 计算余弦相似度矩阵
sim_matrix = cosine_similarity(probs)

from sklearn.metrics.pairwise import cosine_similarity

edge_dict = {}
K = 5  # 每个节点保留 top-K
THRESHOLD = 0.7  # 相似度阈值

for node_type in node_types:
    type_mask = (raw_df['CUST_TYPE'] == node_types.index(node_type))
    type_indices = np.where(type_mask)[0]

    if len(type_indices) == 0:
        continue

    type_sim_matrix = cosine_similarity(probs[type_indices])

    edges = []
    for i in range(len(type_indices)):
        sim_scores = type_sim_matrix[i]
        sim_scores[i] = -1  # 排除自身

        # 获取符合阈值条件的索引（不包含自身）
        valid_indices = np.where(sim_scores >= THRESHOLD)[0]
        
        # 如果存在满足条件的，取其相似度 top-K
        if len(valid_indices) > 0:
            top_k = valid_indices[np.argsort(sim_scores[valid_indices])[-K:]]
        else:
            # 没有满足阈值的，取相似度最高的那个
            top_k = [np.argmax(sim_scores)]

        for j in top_k:
            src = type_indices[i]
            dst = type_indices[j]
            edges.append([src, dst])
            edges.append([dst, src])  # 双向添加

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_dict[(node_type, 'to', node_type)] = edge_index


# 添加到异构图
for edge_type, edge_index in edge_dict.items():
    data[edge_type].edge_index = edge_index


# -------------------- 结果验证 --------------------
print("最终异构图信息:")
print(data)
print("\n节点数量:")
for node_type in node_types:
    print(f"{node_type}: {data[node_type].x.shape[0]}")
print("\n边类型和数量:")
for edge_type in data.edge_types:
    print(f"{edge_type}: {data[edge_type].edge_index.shape[1]} edges")

# 可选：保存异构图
torch.save(data, 'hetero_graph_1.pt')

from torch_geometric.data import Data

# # 1. 合并节点特征
# all_x = []
# all_labels = []
# all_types = []
# all_global_ids = []
# type_to_idx = {t: i for i, t in enumerate(node_types)}
# type_node_ranges = {}

# global_idx = 0
# for node_type in node_types:
#     num_nodes = data[node_type].x.shape[0]
#     all_x.append(data[node_type].x)
#     all_labels.extend(data[node_type].label)
#     all_types.extend([type_to_idx[node_type]] * num_nodes)
#     all_global_ids.extend(data[node_type].cust_id)
#     type_node_ranges[node_type] = (global_idx, global_idx + num_nodes)
#     global_idx += num_nodes

# x_homo = torch.cat(all_x, dim=0)
# label_homo = torch.tensor(all_labels, dtype=torch.long)
# cust_type = torch.tensor(all_types, dtype=torch.long)

# # 2. 合并边
# all_edges = []
# for (src_type, _, dst_type), edge_index in data.edge_index_dict.items():
#     src_offset = type_node_ranges[src_type][0]
#     dst_offset = type_node_ranges[dst_type][0]
#     adjusted_edge = edge_index + torch.tensor([[src_offset], [dst_offset]])
#     all_edges.append(adjusted_edge)

# edge_index_homo = torch.cat(all_edges, dim=1)

# # 3. 构建同构图
# homo_data = Data(x=x_homo, edge_index=edge_index_homo)
# homo_data.y = label_homo
# homo_data.cust_type = cust_type  # 可选属性：节点类型
# homo_data.cust_id = all_global_ids  # 可选属性：客户ID（列表）

# # 4. 保存同构图
# torch.save(homo_data, 'homogeneous_graph.pt')

# # 打印信息
# print("\n同构图信息:")
# print(homo_data)
# print("x shape:", homo_data.x.shape)
# print("edge_index shape:", homo_data.edge_index.shape)
# print("cust_type (节点类型) 分布:", torch.bincount(homo_data.cust_type))