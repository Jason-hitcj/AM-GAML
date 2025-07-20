from torch_geometric.data import HeteroData
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, GATv2Conv, SAGEConv, Linear, to_hetero
from torch_geometric.data import HeteroData
from sklearn.metrics import precision_recall_curve, auc, precision_score, recall_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from for_LSTM import CustomerDataset, CustomerLSTM, CustomerSequenceModel

class SequenceInputBuilder:
    def __init__(self, feature_columns, cat_cols, data_path='data/preprocessed_data_gnn_fill.csv'):
        self.data_path = data_path
        self.feature_columns = feature_columns
        self.cat_cols = cat_cols
        self.num_cols = [col for col in self.feature_columns if col not in self.cat_cols]
        self.label_encoders = {}
        self.scaler = MinMaxScaler()

    def load_and_preprocess_data(self):
        """加载并预处理数据，只处理序列输入特征"""
        df = pd.read_csv(self.data_path)

        # 对金额等取log，防止极值影响
        for col in ['CONF_AMTS', '5D_TOTAL']:
            if col in df.columns:
                df[col] = np.log1p(df[col].clip(1e-5, None))

        # 分类特征进行Label Encoding
        for col in self.cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # 数值特征归一化
        df[self.num_cols] = self.scaler.fit_transform(df[self.num_cols])

        return df

    def build_sequences(self, df, cust_id_list):
        """构造每个客户的输入序列并返回PyTorch张量
        
        返回:
            node_seqs: 形状为 [num_nodes, seq_len, input_dim] 的张量
            lengths: 包含每个序列长度的张量
        """
        sequences = []
        lengths = []
        grouped = df.groupby('CUST_ID')

        for cust_id in cust_id_list:
            if cust_id not in grouped.groups:
                continue
            group = grouped.get_group(cust_id)
            sequence = group[self.feature_columns].values  # 获取numpy数组
            sequences.append(sequence)
            lengths.append(len(sequence))

        # 转换为PyTorch张量
        if not sequences:  # 如果没有有效序列
            return torch.empty((0, 0, len(self.feature_columns))), torch.tensor(lengths)
        
        # 填充序列使它们长度相同 (seq_len)
        max_len = max(lengths)
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_len:
                # 用零填充短序列
                pad_len = max_len - len(seq)
                pad = np.zeros((pad_len, len(self.feature_columns)))
                padded_seq = np.vstack([seq, pad])
            else:
                padded_seq = seq
            padded_sequences.append(padded_seq)
        
        # 堆叠所有序列形成 [num_nodes, seq_len, input_dim] 张量
        node_seqs = torch.tensor(np.stack(padded_sequences), dtype=torch.float32)
        lengths = torch.tensor(lengths, dtype=torch.long)
        
        return node_seqs, lengths
    


class HomoGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=64, out_channels=2, num_heads=4, dropout=0):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.classifier = Linear(hidden_channels * num_heads, out_channels)
        self.bn = torch.nn.BatchNorm1d(hidden_channels * num_heads)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(self.bn(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.leaky_relu(self.bn(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)


class LSTM_GNN_Model(torch.nn.Module):
    def __init__(self, input_dim=14, lstm_hidden=64, gnn_hidden=64, out_channels=2, num_heads=4):
        super().__init__()
        self.lstm_encoder = LSTMEncoder(input_dim, lstm_hidden)
        self.gnn = HomoGNN(in_channels=lstm_hidden, hidden_channels=gnn_hidden,
                           out_channels=out_channels, num_heads=num_heads)

    def forward(self, node_seqs, edge_index):
        node_features = self.lstm_encoder(node_seqs)  # [num_nodes, 64]
        return self.gnn(node_features, edge_index)
    
class CustomerLSTM(nn.Module):
    """LSTM模型"""
    def __init__(self, input_size, hidden_size, num_classes=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout = nn.Dropout(0.2)
        self.device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")  # 添加设备属性
        # self.device = torch.device("cpu")  # 添加设备属性
        self.to(self.device)  # 确保模型参数在正确设备上
    
    def forward(self, x, lengths=1980):
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
        return last_step

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
    
    def forward(self, x_seq):  # x_seq: [N, T, input_dim]
        _, (h_n, _) = self.lstm(x_seq)
        return h_n[-1]  # shape: [N, hidden_dim]

from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score


def train(model, node_seqs, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(node_seqs, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate(model, node_seqs, data, mask_type='val'):
    model.eval()
    metrics = {}
    with torch.no_grad():
        out = model(node_seqs, data.edge_index)
        mask = getattr(data, f"{mask_type}_mask")
        if mask.sum() == 0:
            return metrics
        pred = out[mask].argmax(dim=1)
        true = data.y[mask]
        probs = F.softmax(out[mask], dim=1)[:, 1]
        metrics['acc'] = accuracy_score(true.cpu(), pred.cpu())
        metrics['f1'] = f1_score(true.cpu(), pred.cpu(), average='macro')
        metrics['precision'] = precision_score(true.cpu(), pred.cpu(), average='macro')
        metrics['recall'] = recall_score(true.cpu(), pred.cpu(), average='macro')
        precision_curve, recall_curve, _ = precision_recall_curve(true.cpu(), probs.cpu())
        metrics['auprc'] = auc(recall_curve, precision_curve)
    return metrics

def train_gnn(data, node_seqs, epochs=100, patience=5):
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    model = LSTM_GNN_Model(input_dim=14).to(device)
    data = data.to(device)
    node_seqs = node_seqs.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in tqdm(range(epochs), desc="Training"):
        loss = train(model, node_seqs, data, optimizer, criterion)

        if epoch % 5 == 0:
            val_metrics = evaluate(model, node_seqs, data, 'val')
            history.append({'epoch': epoch, 'loss': loss, **val_metrics})

            current_f1 = val_metrics.get('f1', 0)
            if current_f1 > best_val_f1:
                best_val_f1 = current_f1
                best_epoch = epoch
                torch.save(model.state_dict(), 'data/best_model_joint.pt')

            if epoch >= best_epoch + patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load('data/best_model_joint.pt', map_location=device))
    test_metrics = evaluate(model, node_seqs, data, 'test')
    print("\nTest Results:")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    return model, history



# -------------------- 5. 执行训练 --------------------
if __name__ == "__main__":
    # 设置特征列
    feature_columns = [
        'CUST_TYPE', 'BUSI_CODE', 'FUND_CODE', 'CONF_AMTS', 
        'GENDER', 'NET_CODE', 'RISK_LEV', 'AGE', 
        'TELL_PREFIX', 'COUNTY_PROV', 'COUNTY_CITY', 
        'COUNTY_DIST', '5D_TOTAL', 'FLAG'
    ]
    cat_cols = [
        'CUST_TYPE', 'BUSI_CODE', 'FUND_CODE', 'GENDER', 
        'NET_CODE', 'COUNTY_PROV', 'COUNTY_CITY', 'COUNTY_DIST', 'FLAG'
    ]

    builder = SequenceInputBuilder(feature_columns, cat_cols)
    df = builder.load_and_preprocess_data()

    graph_path = 'data/homogeneous_graph.pt'
    data = torch.load(graph_path)

    print(data)

    cust_ids = data.cust_id

    # # 假设你有这三类ID列表
    # train_ids = pd.read_csv("data/train_ids.csv")['CUST_ID'].tolist()
    # val_ids = pd.read_csv("data/val_ids.csv")['CUST_ID'].tolist()
    # test_ids = pd.read_csv("data/test_ids.csv")['CUST_ID'].tolist()

    # 构建序列输入
    node_sequences, node_lengths = builder.build_sequences(df, cust_ids)

    print(node_sequences.shape)


    model, history = train_gnn(data,node_sequences)



    
    # model, history = train_gnn(data)
    
    # # 保存完整模型
    # torch.save({
    #     'model_state': model.state_dict(),
    #     'metrics': history
    # }, 'gnn_training_results.pt')
