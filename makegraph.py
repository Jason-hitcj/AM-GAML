import pandas as pd
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import LabelEncoder
import numpy as np

# 1. 读取主数据
df = pd.read_csv("data_new/preprocessed_data_gnn_full.csv")

# 2. 编码 CUST_ID 和 FUND_CODE 为图中连续索引
cust_encoder = LabelEncoder()
fund_encoder = LabelEncoder()
df['cust_index'] = cust_encoder.fit_transform(df['CUST_ID'])
df['fund_index'] = fund_encoder.fit_transform(df['FUND_CODE'])

# 3. 构建 customer 节点特征
cust_features = df[['cust_index', 'CUST_ID', 'CUST_TYPE', 'GENDER', 'NET_CODE', 'RISK_LEV', 'AGE']].drop_duplicates('cust_index')
cust_features = cust_features.sort_values('cust_index')
x_cust = torch.tensor(cust_features[['CUST_TYPE', 'GENDER', 'NET_CODE', 'RISK_LEV', 'AGE']].values, dtype=torch.float)

# 4. 构建 fund 节点特征（此处为空）
fund_ids = df[['fund_index']].drop_duplicates().sort_values('fund_index')
x_fund = torch.zeros((len(fund_ids), 1))  # 可添加基金特征

# 5. 构建边和边属性
edge_index = torch.tensor([
    df['cust_index'].values,
    df['fund_index'].values
], dtype=torch.long)

edge_attr_cols = ['CONF_AMTS', 'BUSI_CODE', 'CONF_YEAR', 'CONF_MONTH', 'CONF_DAY', '5D_TOTAL', 'FLAG']
edge_attr = torch.tensor(df[edge_attr_cols].values, dtype=torch.float)

# 6. 构建标签（每个 customer 仅保留一个 target）
cust_label_df = df[['cust_index', 'CUST_ID', 'target']].drop_duplicates('cust_index').sort_values('cust_index')
y = torch.tensor(cust_label_df['target'].values, dtype=torch.long)

# 7. 读取已有的划分文件并创建 mask
train_ids = pd.read_csv('data_new/train_ids.csv')['CUST_ID'].tolist()
val_ids = pd.read_csv('data_new/val_ids.csv')['CUST_ID'].tolist()
test_ids = pd.read_csv('data_new/test_ids.csv')['CUST_ID'].tolist()

# 创建 cust_id -> cust_index 的映射
cust_id_to_index = dict(zip(cust_label_df['CUST_ID'], cust_label_df['cust_index']))

# 初始化 mask
n_customers = len(cust_label_df)
train_mask = torch.zeros(n_customers, dtype=torch.bool)
val_mask = torch.zeros(n_customers, dtype=torch.bool)
test_mask = torch.zeros(n_customers, dtype=torch.bool)

# 标记 mask
for cid in train_ids:
    if cid in cust_id_to_index:
        train_mask[cust_id_to_index[cid]] = True
for cid in val_ids:
    if cid in cust_id_to_index:
        val_mask[cust_id_to_index[cid]] = True
for cid in test_ids:
    if cid in cust_id_to_index:
        test_mask[cust_id_to_index[cid]] = True

# 8. 构建 HeteroData 图对象
data = HeteroData()
data['customer'].x = x_cust
data['customer'].y = y
data['customer'].train_mask = train_mask
data['customer'].val_mask = val_mask
data['customer'].test_mask = test_mask

data['fund'].x = x_fund

data['customer', 'invests', 'fund'].edge_index = edge_index
data['customer', 'invests', 'fund'].edge_attr = edge_attr
data['fund', 'rev_invests', 'customer'].edge_index = edge_index[[1, 0]]
data['fund', 'rev_invests', 'customer'].edge_attr = edge_attr

print(data)

# ✅ 保存图数据
torch.save(data, 'data_new/hetero_graph.pt')
print("图数据已保存到 data_new/hetero_graph.pt")