import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)

# 1. 读取数据
df = pd.read_csv("data_new/customer_node_features_gnn.csv")  

# 2. 提取特征和标签
feature_cols = [f"feature_{i}" for i in range(64)]
X = df[feature_cols].values
y = df["label"].values

# 3. 读取预先划分好的ID列表 (修正部分)
train_ids = pd.read_csv('data_new/train_ids.csv')['CUST_ID'].tolist()
val_ids = pd.read_csv('data_new/val_ids.csv')['CUST_ID'].tolist()
test_ids = pd.read_csv('data_new/test_ids.csv')['CUST_ID'].tolist()

# 4. 创建ID到索引的映射字典 (修正部分)
id_to_index = {cust_id: idx for idx, cust_id in enumerate(df['CUST_ID'])}

# 5. 获取对应的索引位置 (修正部分)
train_indices = [id_to_index[id_] for id_ in train_ids]
val_indices = [id_to_index[id_] for id_ in val_ids]
test_indices = [id_to_index[id_] for id_ in test_ids]

# 6. 按照索引划分数据集 (修正部分)
X_train_raw = X[train_indices]
X_val_raw = X[val_indices]
X_test_raw = X[test_indices]

y_train = y[train_indices]
y_val = y[val_indices]
y_test = y[test_indices]

# 7. 特征归一化 (修正了数据泄露问题)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)

print("===== 数据集划分完成 =====")
print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"验证集形状: X_val={X_val.shape}, y_val={y_val.shape}")
print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")


# 8. 训练标准的SVM模型
print("\n===== 开始训练SVM模型 =====")

# C 和 gamma 是最重要的超参数，可以后续使用验证集进行调优
svm_model = SVC(
    kernel='rbf', 
    C=0.1, 
    gamma='scale', 
    probability=True, 
    class_weight=None,
    random_state=42
)

# 一次性完成训练
svm_model.fit(X_train, y_train)
print("===== SVM模型训练完成 =====")


# 9. 辅助函数：用于评估和打印结果
def evaluate_and_print(model_name, model, X_data, y_data):
    print(f"\n===== 在 {model_name} 上的评估指标 =====")
    
    # 获取预测结果
    y_prob = model.predict_proba(X_data)[:, 1]
    y_pred = model.predict(X_data)
    
    # 计算指标
    acc = accuracy_score(y_data, y_pred)
    prec = precision_score(y_data, y_pred, zero_division=0)
    rec = recall_score(y_data, y_pred, zero_division=0)
    f1 = f1_score(y_data, y_pred, zero_division=0)
    auprc = average_precision_score(y_data, y_prob)

    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUPRC        : {auprc:.4f}")

# 10. 在验证集和测试集上评估模型
evaluate_and_print("验证集", svm_model, X_val, y_val)
evaluate_and_print("最终测试集", svm_model, X_test, y_test)