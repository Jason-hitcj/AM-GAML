import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
# 加载数据
df = pd.read_csv('data/preprocessed_data_gnn.csv')

# 选择要使用的特征列
feature_columns = ['CUST_TYPE', 'BUSI_CODE', 'FUND_CODE', 'CONF_AMTS', 
                  'GENDER', 'NET_CODE', 'RISK_LEV', 'AGE', 
                  'TELL_PREFIX', 'COUNTY_PROV', 'COUNTY_CITY', 
                  'COUNTY_DIST', '5D_TOTAL', 'FLAG']

# 分类变量编码
cat_cols = ['CUST_TYPE', 'BUSI_CODE', 'FUND_CODE', 'GENDER', 
            'NET_CODE', 'COUNTY_PROV', 'COUNTY_CITY', 'COUNTY_DIST', 'FLAG']

# 数值列
num_cols = [col for col in feature_columns if col not in cat_cols]

# 特殊处理：对金额取对数
df['CONF_AMTS'] = np.log1p(df['CONF_AMTS'].clip(1e-5, None))  # 避免log(0)
df['5D_TOTAL'] = np.log1p(df['5D_TOTAL'].clip(1e-5, None))

# 1. 数据预处理
# 对分类变量进行标签编码
label_encoders = {}
for col in cat_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# 对数值列进行归一化
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 2. 按客户账号分组并创建序列
customer_groups = df.groupby('CUST_ID')

from sklearn.model_selection import train_test_split

# 获取所有客户的 ID 和类型
cust_id_type_df = df.groupby('CUST_ID')['CUST_TYPE'].first().reset_index()

train_ids, val_ids, test_ids = [], [], []

# 遍历每种客户类型，单独划分
for cust_type in cust_id_type_df['CUST_TYPE'].unique():
    ids_of_type = cust_id_type_df[cust_id_type_df['CUST_TYPE'] == cust_type]['CUST_ID'].tolist()
    
    # 先划分出训练集（70%）
    ids_train, ids_temp = train_test_split(
        ids_of_type, test_size=0.3, random_state=42
    )
    
    # 剩下的 30% 再划分为验证集（15%）和测试集（15%）
    ids_val, ids_test = train_test_split(
        ids_temp, test_size=0.5, random_state=42
    )
    
    train_ids.extend(ids_train)
    val_ids.extend(ids_val)
    test_ids.extend(ids_test)

# 保存为 CSV 方便后续使用
pd.DataFrame({'CUST_ID': train_ids}).to_csv('data/train_ids.csv', index=False)
pd.DataFrame({'CUST_ID': val_ids}).to_csv('data/val_ids.csv', index=False)
pd.DataFrame({'CUST_ID': test_ids}).to_csv('data/test_ids.csv', index=False)

# 输出统计信息
print(f"训练集客户数: {len(train_ids)}")
print(f"验证集客户数: {len(val_ids)}")
print(f"测试集客户数: {len(test_ids)}")


from collections import defaultdict

cust_type_dist = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})

for cust_type in cust_id_type_df['CUST_TYPE'].unique():
    ids_of_type = cust_id_type_df[cust_id_type_df['CUST_TYPE'] == cust_type]['CUST_ID'].tolist()
    
    train_cnt = len(set(ids_of_type) & set(train_ids))
    val_cnt = len(set(ids_of_type) & set(val_ids))
    test_cnt = len(set(ids_of_type) & set(test_ids))
    
    cust_type_dist[cust_type]['train'] = train_cnt
    cust_type_dist[cust_type]['val'] = val_cnt
    cust_type_dist[cust_type]['test'] = test_cnt

# 打印出来
print("按客户类型的划分分布：")
for cust_type, counts in cust_type_dist.items():
    print(f"{cust_type}: train={counts['train']}, val={counts['val']}, test={counts['test']}")


for label, id_list in [('train', train_ids), ('test', test_ids)]:
    types = cust_id_type_df[cust_id_type_df['CUST_ID'].isin(id_list)]['CUST_TYPE']
    print(f"{label} set CUST_TYPE分布：\n{types.value_counts(normalize=True)}\n")

# 存储每个客户的序列和特征
customer_sequences = []
customer_labels = []
customer_types = []
customer_ids = []



def generate_sequences_for_ids(id_list, df, feature_columns):
    sequences = []
    labels = []
    types = []
    ids = []
    groups = df.groupby('CUST_ID')

    for cust_id in id_list:
        group = groups.get_group(cust_id)
        seq = group[feature_columns].values
        label = 1 if (group['target'].sum() > 0) else 0
        cust_type = group['CUST_TYPE'].iloc[0]
        
        sequences.append(seq)
        labels.append(label)
        types.append(cust_type)
        ids.append(cust_id)
    
    return sequences, labels, types, ids

train_seqs, train_labels, train_types, train_ids = generate_sequences_for_ids(train_ids, df, feature_columns)
val_seqs, val_labels, val_types, val_ids = generate_sequences_for_ids(val_ids, df, feature_columns)
test_seqs, test_labels, test_types, test_ids = generate_sequences_for_ids(test_ids, df, feature_columns)

customer_labels = np.array(customer_labels)

# 3. 准备数据
class CustomerDataset(Dataset):
    def __init__(self, sequences, labels=None):
        self.sequences = sequences
        self.labels = labels
        self.lengths = [len(seq) for seq in sequences]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.FloatTensor(self.sequences[idx]), self.lengths[idx], torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            return torch.FloatTensor(self.sequences[idx]), self.lengths[idx]

def collate_fn(batch):
    if len(batch[0]) == 3:  # 有监督模式
        sequences, lengths, labels = zip(*batch)
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        return sequences_padded, torch.tensor(lengths), torch.stack(labels)
    else:  # 无监督模式
        sequences, lengths = zip(*batch)
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        return sequences_padded, torch.tensor(lengths)

# 创建数据集和数据加载器
# dataset = CustomerDataset(customer_sequences, customer_labels)
# dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)
train_dataset = CustomerDataset(train_seqs, train_labels)
train_dataloader = DataLoader(train_dataset, batch_size=64, collate_fn=collate_fn, shuffle=True)

# 创建验证集DataLoader
val_dataset = CustomerDataset(val_seqs, val_labels)
val_dataloader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn, shuffle=False)

# 4. 定义模型
class CustomerLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True  # 输入输出格式为(batch, seq_len, features)
        )
        self.fc = nn.Linear(hidden_size, num_classes)  # 二分类输出层
        self.dropout = nn.Dropout(0.2)  # 添加dropout防止过拟合

    def forward(self, x, lengths):
        # 动态处理变长序列
        packed_input = pack_padded_sequence(
            input=x,
            lengths=lengths.cpu(),  # 确保lengths在CPU上
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed_input)  # 忽略隐藏状态
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        # 取每个序列的最后一个有效时间步
        last_step = out[torch.arange(out.size(0)), lengths - 1]
        last_step = self.dropout(last_step)
        return self.fc(last_step)


# 5. 训练配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomerLSTM(
    input_size=len(feature_columns),
    hidden_size=64,
    num_classes=2
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
num_epochs = 100

# 在训练配置部分添加：
best_val_loss = float('inf')
early_stop_patience = 10  # 连续10轮验证损失未改善则停止
patience_counter = 0
train_losses, val_losses = [], []  # 记录损失用于绘图

# 修改训练循环：
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    # 训练阶段
    for batch, lengths, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        batch = batch.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(batch, lengths)
        loss = criterion(outputs, labels)
        
        # 计算准确率
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    # 验证阶段
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch, lengths, labels in val_dataloader:
            batch, labels = batch.to(device), labels.to(device)
            outputs = model(batch, lengths)
            val_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    
    # 早停与模型保存
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        # 保存最佳模型
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Saved best model (val_loss={avg_val_loss:.4f})")
    else:
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # 打印训练/验证指标
    print(f"Epoch {epoch+1}: "
          f"Train Loss={total_loss/len(train_dataloader):.4f}, "
          f"Val Loss={avg_val_loss:.4f}, "
          f"Val Acc={val_accuracy:.2f}%")



# 6. 为每个客户生成节点特征和预测概率
@torch.no_grad()
def generate_node_features(model, sequences, lengths, labels, cust_ids, cust_types):
    model.eval()
    customer_features = []
    customer_probs = []
    for seq, length in zip(sequences, lengths):
        seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(device)
        packed = pack_padded_sequence(seq_tensor, torch.tensor([length]), batch_first=True, enforce_sorted=False)
        out_packed, _ = model.lstm(packed)
        out, _ = pad_packed_sequence(out_packed, batch_first=True)
        last_step = out[0, length - 1].cpu().numpy()
        customer_features.append(last_step)
        output = model.fc(torch.FloatTensor(last_step).unsqueeze(0).to(device))
        prob = torch.softmax(output, dim=1)[0, 1].item()
        customer_probs.append(prob)

    return pd.DataFrame({
        'CUST_ID': cust_ids,
        'CUST_TYPE': cust_types,
        'label': labels,
        'anomaly_prob': customer_probs,
        **{f'feature_{i}': [f[i] for f in customer_features] for i in range(len(customer_features[0]))}
    })

# 训练结束后加载最佳模型
model.load_state_dict(torch.load('best_model.pth'))
train_node_df = generate_node_features(model, train_seqs, [len(s) for s in train_seqs], train_labels, train_ids, train_types)
val_node_df = generate_node_features(model, val_seqs, [len(s) for s in val_seqs], val_labels, val_ids, val_types)
test_node_df = generate_node_features(model, test_seqs, [len(s) for s in test_seqs], test_labels, test_ids, test_types)

node_features = pd.concat([train_node_df, val_node_df, test_node_df])
node_features.to_csv('data/customer_node_features.csv', index=False)

print("Supervised node features generated and saved successfully!")