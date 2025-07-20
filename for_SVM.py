import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    auc
)

# 1. 读取数据
df = pd.read_csv("data_new/customer_node_features_gnn.csv")  

# 2. 特征列
feature_cols = [f"feature_{i}" for i in range(64)]
X = df[feature_cols].values
y = df["label"].values

# 3. 数据归一化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=1, stratify=y
)

# 自定义训练过程类
class CustomSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale', max_iter=100, verbose_interval=10):
        self.svm = SVC(kernel=kernel, C=C, gamma=gamma, probability=True, max_iter=max_iter)
        self.verbose_interval = verbose_interval
        self.max_iter = max_iter
    
    def fit(self, X_train, y_train):
        # 保存中间结果
        self.train_metrics = []
        
        # 自定义训练循环
        for i in range(1, self.max_iter + 1):
            # 使用部分迭代次数训练
            self.svm.max_iter = i
            self.svm.fit(X_train, y_train)
            
            # 每verbose_interval次迭代计算并打印指标
            if i % self.verbose_interval == 0 or i == 1 or i == self.max_iter:
                # 在训练集上计算指标
                y_pred_train = self.svm.predict(X_train)
                y_prob_train = self.svm.predict_proba(X_train)[:, 1]
                
                # 计算所有指标
                acc = accuracy_score(y_train, y_pred_train)
                prec = precision_score(y_train, y_pred_train, zero_division=0)
                rec = recall_score(y_train, y_pred_train, zero_division=0)
                f1 = f1_score(y_train, y_pred_train, zero_division=0)
                
                # 计算AUPRC
                precision_curve, recall_curve, _ = precision_recall_curve(y_train, y_prob_train)
                auprc = auc(recall_curve, precision_curve)
                
                # 保存指标
                self.train_metrics.append({
                    'iteration': i,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'auprc': auprc
                })
                
                # 打印指标
                print(f"\nIteration {i}/{self.max_iter} - Training Metrics:")
                print(f"Accuracy     : {acc:.4f}")
                print(f"Precision    : {prec:.4f}")
                print(f"Recall       : {rec:.4f}")
                print(f"F1 Score     : {f1:.4f}")
                print(f"AUPRC        : {auprc:.4f}")
        
        return self
    
    def predict(self, X):
        return self.svm.predict(X)
    
    def predict_proba(self, X):
        return self.svm.predict_proba(X)

# 5. 训练自定义SVM模型
print("===== 开始训练 =====")
custom_svm = CustomSVM(kernel='rbf', C=1.0, gamma='scale', max_iter=100, verbose_interval=10)
custom_svm.fit(X_train, y_train)

# 6. 在测试集上评估
y_pred = custom_svm.predict(X_test)
y_prob = custom_svm.predict_proba(X_test)[:, 1]  # 获取正类概率

# 7. 计算测试集指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auprc = average_precision_score(y_test, y_prob)

# 8. 打印最终结果
print("\n===== 最终测试集评估指标 =====")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"AUPRC        : {auprc:.4f}")

# 9. 可选：保存训练过程中的指标
train_metrics_df = pd.DataFrame(custom_svm.train_metrics)
print("\n训练过程指标记录:")
print(train_metrics_df)