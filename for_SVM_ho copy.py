import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- 加载LSTM特征 --------------------
lstm_features_df = pd.read_csv('data/customer_node_features_fill.csv')

# -------------------- 加载划分 ID --------------------
train_ids = pd.read_csv('data/train_ids.csv')['CUST_ID'].tolist()
val_ids = pd.read_csv('data/val_ids.csv')['CUST_ID'].tolist()
test_ids = pd.read_csv('data/test_ids.csv')['CUST_ID'].tolist()

# 创建划分标签列
lstm_features_df['split'] = 'test'
lstm_features_df.loc[lstm_features_df['CUST_ID'].isin(train_ids), 'split'] = 'train'
lstm_features_df.loc[lstm_features_df['CUST_ID'].isin(val_ids), 'split'] = 'val'

# -------------------- 过滤数据 --------------------
df = lstm_features_df[lstm_features_df['split'].isin(['train', 'val', 'test'])].reset_index(drop=True)

# -------------------- 特征处理 --------------------
lstm_feats = df[[f'feature_{i}' for i in range(64)]].values
x = torch.tensor(lstm_feats, dtype=torch.float)
y = torch.tensor(df['label'].values, dtype=torch.long)
cust_ids = df['CUST_ID'].tolist()

# -------------------- 构建边 --------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(lstm_feats)
svm = SVC(kernel='rbf', probability=True, random_state=42)
svm.fit(X_scaled, y)
probs = svm.predict_proba(X_scaled)

sim_matrix = cosine_similarity(probs)
edges = []
K = 5
THRESHOLD = 0.7

for i in range(len(sim_matrix)):
    sim_scores = sim_matrix[i]
    sim_scores[i] = -1
    valid_indices = np.where(sim_scores >= THRESHOLD)[0]
    if len(valid_indices) > 0:
        top_k = valid_indices[np.argsort(sim_scores[valid_indices])[-K:]]
    else:
        top_k = [np.argmax(sim_scores)]
    for j in top_k:
        edges.append([i, j])
        edges.append([j, i])

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# -------------------- 构建掩码 --------------------
split_series = df['split']
train_mask = torch.tensor(split_series == 'train')
val_mask = torch.tensor(split_series == 'val')
test_mask = torch.tensor(split_series == 'test')

# -------------------- 构建图 --------------------
data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask
data.cust_id = cust_ids  # 可选

# -------------------- 保存 --------------------
torch.save(data, 'data/homogeneous_graph_lstm_only.pt')

print("✅ 同构图构建完成（仅使用LSTM特征）")
print("节点数:", data.x.shape[0])
print("特征维度:", data.x.shape[1])
print("训练集节点数:", data.train_mask.sum().item())
print("验证集节点数:", data.val_mask.sum().item())
print("测试集节点数:", data.test_mask.sum().item())