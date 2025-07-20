import pandas as pd
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 加载数据
df = pd.read_csv('data/preprocessed_data_gnn.csv')

# 1. 数据预处理
# 确保所有类别特征都是字符串类型
cat_cols = ['COUNTY_PROV', 'COUNTY_CITY', 'COUNTY_DIST', 'NET_CODE']
df[cat_cols] = df[cat_cols].astype(str)

# 2. 特征编码
# 准备one-hot编码器
categories = {
    col: df[col].dropna().astype(str).unique().tolist()
    for col in cat_cols
}

encoder = OneHotEncoder(
    categories=[categories[col] for col in cat_cols],
    sparse_output=False,
    handle_unknown='ignore'
)
encoder.fit(df[cat_cols])

dummy_df = pd.DataFrame([{col: categories[col][0] for col in cat_cols}])
onehot_dim = encoder.transform(dummy_df).shape[1]
print(f"每个客户的one-hot特征维度: {onehot_dim}")

# 3. 创建异构图数据对象
data = HeteroData()

# 4. 添加节点
# 获取所有唯一ID
all_customers = df['CUST_ID'].unique()

# 创建映射字典
customer_to_idx = {cust_id: idx for idx, cust_id in enumerate(all_customers)}


# 准备客户节点特征
customer_features = []
for cust_id in all_customers:
    cust_data = df[df['CUST_ID'] == cust_id].iloc[0]
    cust_type = cust_data['CUST_TYPE']
    
    # 基础特征
    features = [
        float(cust_type),
        float(cust_data['RISK_LEV']),
        float(cust_data['TELL_PREFIX'])
    ]
    
    # one-hot特征 (保持DataFrame格式)
    try:
        onehot = encoder.transform(
            pd.DataFrame([cust_data[cat_cols]], columns=cat_cols)
        )[0]
    except ValueError as e:
        print(f"处理客户 {cust_id} 时出错:")
        print(cust_data[cat_cols])
        raise e
        
    features.extend(onehot)
    
    # 个人特有特征
    features.append(float(cust_data['AGE'])) if cust_type == 0 else features.append(0.0)
    
    customer_features.append(features)

# 转换为tensor并添加
customer_features = np.array(customer_features, dtype=np.float32)
data['customer'].x = torch.from_numpy(customer_features)












# 5. 添加边
# 收集所有边信息
edge_indices = []  # 存储边的索引
edge_attrs = []   # 存储边属性
edge_y = []
for _, row in df.iterrows():
    # 源节点(客户)和目标节点(基金)的索引
    src = customer_to_idx[row['CUST_ID']]
    dst = fund_to_idx[row['FUND_CODE']]
    
    edge_indices.append([src, dst])
    edge_y.append(row['target'])
    # 边属性
    edge_attrs.append([
        row['CONF_AMTS'],
        row['5D_TOTAL'],
        row['FLAG']
    ])

# 转换为tensor并添加
edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

data['customer', 'transaction', 'fund'].edge_index = edge_index
data['customer', 'transaction', 'fund'].edge_attr = edge_attr
data['customer', 'transaction', 'fund'].y = edge_y

# 6. 添加目标标签 (如果有监督学习需求)
if 'target' in df.columns:
    # 假设每个客户有一个目标标签
    targets = df.groupby('CUST_ID')['target'].any().astype(int).values
    data['customer'].y = torch.from_numpy(targets).long()

def add_reverse_edge(data, src_type, rel_type, dst_type, edge_attr_transform=None):
    src, dst = data[(src_type, rel_type, dst_type)].edge_index
    edge_attr = data[(src_type, rel_type, dst_type)].edge_attr

    # 默认直接复制属性
    reversed_attr = edge_attr.clone()
    if edge_attr_transform:
        reversed_attr = edge_attr_transform(edge_attr)

    # 添加反向边
    rev_rel_type = f'rev_{rel_type}'
    data[(dst_type, rev_rel_type, src_type)].edge_index = torch.stack([dst, src], dim=0)
    data[(dst_type, rev_rel_type, src_type)].edge_attr = reversed_attr

add_reverse_edge(data, 'customer', 'transaction', 'fund')


# 7. 验证图数据
print(data)
print("\n节点类型:", data.node_types)
print("\n边类型:", data.edge_types)
print("\n客户节点特征维度:", data['customer'].x.shape)
print("产品节点特征维度:", data['fund'].x.shape)
print("边特征维度:", data['customer', 'transaction', 'fund'].edge_attr.shape)
print('y=',data['customer'].y)
# 8. 保存图数据 (可选)
torch.save(data, 'financial_graph.pt')