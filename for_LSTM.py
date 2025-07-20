import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import defaultdict

# 固定随机种子
def set_random_seed(seed=42):
    """设置所有相关库的随机种子"""
    # Python原生random
    import random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # 确保PyTorch的确定性行为
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量确保完全可重复
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"已设置随机种子为: {seed}")

# 在程序开始时设置随机种子
set_random_seed(42)

class CustomerSequenceModel:
    def __init__(self, data_path='data_new/preprocessed_data_gnn_full.csv'):
        self.data_path = data_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.feature_columns = [
            'CUST_TYPE', 'BUSI_CODE', 'FUND_CODE', 'CONF_AMTS', 'CONF_YEAR','CONF_MONTH','CONF_DAY',
            'GENDER', 'NET_CODE', 'RISK_LEV', 'AGE', 
            'TELL_PREFIX', 'COUNTY_PROV', 'COUNTY_CITY', 
            'COUNTY_DIST', '5D_TOTAL', 'FLAG'
        ]
        self.cat_cols = [
            'CUST_TYPE', 'BUSI_CODE', 'FUND_CODE', 'GENDER', 
            'NET_CODE', 'COUNTY_PROV', 'COUNTY_CITY', 'COUNTY_DIST', 'FLAG','CONF_YEAR','CONF_MONTH','CONF_DAY'
        ]
        self.num_cols = [col for col in self.feature_columns if col not in self.cat_cols]
        self.label_encoders = {}
        self.scaler = MinMaxScaler()
        self.model = CustomerLSTM(
            input_size=14,
            hidden_size=64,
            num_classes=2
        ).to(self.device)
        
    def load_and_preprocess_data(self):
        """加载并预处理数据"""
        df = pd.read_csv(self.data_path)
        
        # 特殊处理：对金额取对数
        df['CONF_AMTS'] = np.log1p(df['CONF_AMTS'].clip(1e-5, None))
        df['5D_TOTAL'] = np.log1p(df['5D_TOTAL'].clip(1e-5, None))
        
        # 对分类变量进行标签编码
        for col in self.cat_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # 对数值列进行归一化
        df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])
        
        return df
    
    def split_data(self, df):
        """划分训练集、验证集和测试集
        如果存在划分文件(data/train_ids.csv, data/val_ids.csv, data/test_ids.csv)，
        则直接读取；否则进行划分并保存划分结果。
        新增功能：统计并打印训练集、验证集和测试集的交易笔数
        """
        flag = 1
        # 检查划分文件是否存在
        if (os.path.exists('data_new/train_ids.csv') and \
        (os.path.exists('data_new/val_ids.csv')) and \
        (os.path.exists('data_new/test_ids.csv'))) and flag:
            # 直接读取已有的划分
            train_ids = pd.read_csv('data_new/train_ids.csv')['CUST_ID'].tolist()
            val_ids = pd.read_csv('data_new/val_ids.csv')['CUST_ID'].tolist()
            test_ids = pd.read_csv('data_new/test_ids.csv')['CUST_ID'].tolist()
            print("检测到已有划分文件，直接读取划分序列")
        else:
            # 进行新的划分
            print("未检测到划分文件，开始新的划分...")
            cust_id_type_df = df.groupby('CUST_ID')['CUST_TYPE'].first().reset_index()
            train_ids, val_ids, test_ids = [], [], []
            
            for cust_type in cust_id_type_df['CUST_TYPE'].unique():
                ids_of_type = cust_id_type_df[cust_id_type_df['CUST_TYPE'] == cust_type]['CUST_ID'].tolist()
                ids_train, ids_temp = train_test_split(ids_of_type, test_size=0.6, random_state=42)
                ids_val, ids_test = train_test_split(ids_temp, test_size=0.5, random_state=42)

                train_ids.extend(ids_train)
                val_ids.extend(ids_val)
                test_ids.extend(ids_test)
            
            # 确保data目录存在
            os.makedirs('data_new', exist_ok=True)
            
            # 保存ID文件
            pd.DataFrame({'CUST_ID': train_ids}).to_csv('data_new/train_ids.csv', index=False)
            pd.DataFrame({'CUST_ID': val_ids}).to_csv('data_new/val_ids.csv', index=False)
            pd.DataFrame({'CUST_ID': test_ids}).to_csv('data_new/test_ids.csv', index=False)
            print("划分完成并已保存划分文件")
        
        # 统计并打印各数据集的交易笔数
        train_transactions = df[df['CUST_ID'].isin(train_ids)].shape[0]
        val_transactions = df[df['CUST_ID'].isin(val_ids)].shape[0]
        test_transactions = df[df['CUST_ID'].isin(test_ids)].shape[0]
        total_transactions = train_transactions + val_transactions + test_transactions
        
        print("\n数据集交易笔数统计:")
        print(f"训练集交易笔数: {train_transactions} ({(train_transactions/total_transactions)*100:.1f}%)")
        print(f"验证集交易笔数: {val_transactions} ({(val_transactions/total_transactions)*100:.1f}%)")
        print(f"测试集交易笔数: {test_transactions} ({(test_transactions/total_transactions)*100:.1f}%)")
        print(f"总交易笔数: {total_transactions}")
        
        return train_ids, val_ids, test_ids
    
    def generate_sequences(self, df, id_list):
        """为指定ID列表生成序列"""
        sequences = []
        labels = []
        types = []
        ids = []
        groups = df.groupby('CUST_ID')
        
        for cust_id in id_list:
            group = groups.get_group(cust_id)
            seq = group[self.feature_columns].values
            label = 1 if (group['target'].sum() > 0) else 0
            cust_type = group['CUST_TYPE'].iloc[0]
            
            sequences.append(seq)
            labels.append(label)
            types.append(cust_type)
            ids.append(cust_id)
        
        return sequences, labels, types, ids
    
    def calculate_metrics(self, dataloader):
        """计算所有指标"""
        self.model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch, lengths, labels in dataloader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(batch, lengths)
                
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        # 计算所有指标
        metrics = {
            'acc': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, average='macro'),
            'recall': recall_score(all_labels, all_preds, average='macro'),
            'f1': f1_score(all_labels, all_preds, average='macro'),
        }
        
        # 计算AUPRC
        precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
        metrics['auprc'] = auc(recall_curve, precision_curve)
        
        return metrics
    
    def train_model(self, train_seqs, train_labels, val_seqs, val_labels, 
                   input_size=17, hidden_size=64, num_classes=2, 
                   num_epochs=101, batch_size=64, early_stop_patience=100):
        """训练模型"""
        # 为DataLoader设置worker初始化函数，确保多进程加载数据时随机种子固定
        def worker_init_fn(worker_id):
            np.random.seed(42 + worker_id)
        
        train_dataset = CustomerDataset(train_seqs, train_labels)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                     collate_fn=self.collate_fn, shuffle=True,
                                     worker_init_fn=worker_init_fn)
        
        val_dataset = CustomerDataset(val_seqs, val_labels)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                                   collate_fn=self.collate_fn, shuffle=False,
                                   worker_init_fn=worker_init_fn)
        
        self.model = CustomerLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_classes=num_classes
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses, val_losses = [], []
        
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # 训练阶段
            for batch, lengths, labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(batch, lengths)
                loss = criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            # 每10轮计算并打印训练集指标
            if (epoch ) % 10 == 0:
                train_metrics = self.calculate_metrics(train_dataloader)
                print(f"\nEpoch {epoch+1} Training Metrics:")
                print(f"Accuracy: {train_metrics['acc']:.4f}")
                print(f"Precision: {train_metrics['precision']:.4f}")
                print(f"Recall: {train_metrics['recall']:.4f}")
                print(f"F1 Score: {train_metrics['f1']:.4f}")
                print(f"AUPRC: {train_metrics['auprc']:.4f}\n")
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch, lengths, labels in val_dataloader:
                    batch, labels = batch.to(self.device), labels.to(self.device)
                    outputs = self.model(batch, lengths)
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
                torch.save(self.model.state_dict(), 'model/best_model_LSTM.pth')
                print(f"Saved best model (val_loss={avg_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}: "
                  f"Train Loss={total_loss/len(train_dataloader):.4f}, "
                  f"Val Loss={avg_val_loss:.4f}, "
                  f"Val Acc={val_accuracy:.2f}%")
    
    def generate_node_features(self, sequences, lengths, labels, cust_ids, cust_types):
        """生成节点特征"""
        self.model.eval()
        customer_features = []
        customer_probs = []
        
        with torch.no_grad():
            for seq, length in zip(sequences, lengths):
                seq_tensor = torch.FloatTensor(seq).unsqueeze(0).to(self.device)
                packed = pack_padded_sequence(seq_tensor, torch.tensor([length]), 
                                            batch_first=True, enforce_sorted=False)
                out_packed, _ = self.model.lstm(packed)
                out, _ = pad_packed_sequence(out_packed, batch_first=True)
                last_step = out[0, length - 1].cpu().numpy()
                customer_features.append(last_step)
                output = self.model.fc(torch.FloatTensor(last_step).unsqueeze(0).to(self.device))
                prob = torch.softmax(output, dim=1)[0, 1].item()
                customer_probs.append(prob)
        
        return pd.DataFrame({
            'CUST_ID': cust_ids,
            'CUST_TYPE': cust_types,
            'label': labels,
            'anomaly_prob': customer_probs,
            **{f'feature_{i}': [f[i] for f in customer_features] for i in range(len(customer_features[0]))}
        })
    
    @staticmethod
    def collate_fn(batch):
        """自定义批处理函数"""
        if len(batch[0]) == 3:  # 有监督模式
            sequences, lengths, labels = zip(*batch)
            sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
            return sequences_padded, torch.tensor(lengths), torch.stack(labels)
        else:  # 无监督模式
            sequences, lengths = zip(*batch)
            sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
            return sequences_padded, torch.tensor(lengths)


class CustomerDataset(Dataset):
    """自定义数据集类"""
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


class CustomerLSTM(nn.Module):
    """LSTM模型"""
    def __init__(self, input_size, hidden_size, num_classes=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 添加设备属性
        
        # 初始化权重以确保可重复性
        self.init_weights()
        
        self.to(self.device)  # 确保模型参数在正确设备上
    
    def init_weights(self):
        """初始化模型权重"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
    
    def forward(self, x, lengths):
        packed_input = pack_padded_sequence(
            input=x,
            lengths=lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        packed_out, _ = self.lstm(packed_input)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        last_step = out[torch.arange(out.size(0)), lengths - 1]
        last_step = self.dropout(last_step)
        return self.fc(last_step)


def test_model(model, test_seqs, test_labels, batch_size=4):
    """测试模型性能，返回完整评估指标"""
    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
    
    test_dataset = CustomerDataset(test_seqs, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, 
                                collate_fn=CustomerSequenceModel.collate_fn, shuffle=False,
                                worker_init_fn=worker_init_fn)
    
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch, lengths, labels in tqdm(test_dataloader, desc="Testing"):
            batch = batch.to(model.device)  # 使用模型的device属性
            labels = labels.to(model.device)
            outputs = model(batch, lengths)
            
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # 计算所有指标
    metrics = {
        'acc': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1': f1_score(all_labels, all_preds, average='macro'),
    }
    
    # 计算AUPRC
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, all_probs)
    metrics['auprc'] = auc(recall_curve, precision_curve)
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    metrics['confusion_matrix'] = cm
    
    return metrics

def run_customer_sequence_model():
    """运行客户序列模型的完整流程"""
    # 初始化模型
    csm = CustomerSequenceModel()
    
    # 1. 加载并预处理数据
    df = csm.load_and_preprocess_data()
    
    
    # 2. 划分数据集
    train_ids, val_ids, test_ids = csm.split_data(df)
    
    # 3. 生成序列数据
    train_seqs, train_labels, train_types, train_ids = csm.generate_sequences(df, train_ids)
    val_seqs, val_labels, val_types, val_ids = csm.generate_sequences(df, val_ids)
    test_seqs, test_labels, test_types, test_ids = csm.generate_sequences(df, test_ids)

    # 4. 训练模型
    csm.train_model(train_seqs, train_labels, val_seqs, val_labels)

    # 加载最佳模型
    csm.model.load_state_dict(torch.load('model/best_model_LSTM.pth'))
    
    # 5. 生成节点特征
    train_node_df = csm.generate_node_features(train_seqs, [len(s) for s in train_seqs], 
                                             train_labels, train_ids, train_types)
    val_node_df = csm.generate_node_features(val_seqs, [len(s) for s in val_seqs], 
                                           val_labels, val_ids, val_types)
    test_node_df = csm.generate_node_features(test_seqs, [len(s) for s in test_seqs], 
                                            test_labels, test_ids, test_types)
    
    # 6. 保存结果
    node_features = pd.concat([train_node_df, val_node_df, test_node_df])
    node_features.to_csv('data_new/customer_node_features_gnn.csv', index=False)

    # 7. 测试模型性能
    test_metrics = test_model(csm.model, test_seqs, test_labels)
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        if metric != 'confusion_matrix':
            print(f"{metric}: {value:.4f}")
    
    print("\nConfusion Matrix:")
    print(test_metrics['confusion_matrix'])
    
    print("Supervised node features generated and saved successfully!")
    return node_features, test_metrics


# 调用命令
if __name__ == "__main__":
    # 确保在主函数开始时随机种子是固定的
    set_random_seed(42)
    node_features , test_metrics= run_customer_sequence_model()