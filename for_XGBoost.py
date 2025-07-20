import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
    auc
)
import xgboost as xgb
from tqdm import tqdm

# 1. 读取数据
df = pd.read_csv("data/customer_node_features_supervised.csv")  

# 2. 提取特征和标签
feature_cols = [f"feature_{i}" for i in range(64)]
X = df[feature_cols].values
y = df["label"].values

# 3. 特征归一化（可选，XGBoost 不强制需要）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=40, stratify=y
)

# 5. 自定义训练函数，用于监控训练过程中的指标
def custom_train_xgb(X_train, y_train, X_test, y_test, n_estimators=100, 
                    max_depth=5, learning_rate=0.1, eval_interval=10):
    """
    自定义XGBoost训练过程，定期评估训练集指标
    """
    # 转换为DMatrix格式（XGBoost的高效数据格式）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 参数设置
    params = {
        'max_depth': max_depth,
        'eta': learning_rate,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    
    # 用于存储评估结果
    eval_results = []
    
    # 逐步训练模型
    print("\n===== 开始训练XGBoost模型 =====")
    bst = None
    for i in tqdm(range(1, n_estimators + 1)):
        # 每次只训练1棵树
        bst = xgb.train(params, dtrain, num_boost_round=1, xgb_model=bst)
        
        # 每eval_interval轮或在最后一轮评估指标
        if (i == 2) or (i % eval_interval == 0) or (i == n_estimators):
            # 在训练集上预测
            y_pred_train = np.where(bst.predict(dtrain) > 0.5, 1, 0)
            y_prob_train = bst.predict(dtrain)
            
            # 计算所有指标
            acc = accuracy_score(y_train, y_pred_train)
            prec = precision_score(y_train, y_pred_train, zero_division=0)
            rec = recall_score(y_train, y_pred_train, zero_division=0)
            f1 = f1_score(y_train, y_pred_train, zero_division=0)
            
            # 计算AUPRC
            precision_curve, recall_curve, _ = precision_recall_curve(y_train, y_prob_train)
            auprc = auc(recall_curve, precision_curve)
            
            # 保存评估结果
            eval_results.append({
                'n_estimators': i,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1,
                'auprc': auprc
            })
            
            # 打印当前指标
            print(f"\n迭代轮次 {i}/{n_estimators} - 训练集指标:")
            print(f"Accuracy     : {acc:.4f}")
            print(f"Precision    : {prec:.4f}")
            print(f"Recall       : {rec:.4f}")
            print(f"F1 Score     : {f1:.4f}")
            print(f"AUPRC        : {auprc:.4f}")
    
    return bst, eval_results

# 6. 训练模型
n_estimators = 100
eval_interval = 10  # 每10轮评估一次

model, train_metrics = custom_train_xgb(
    X_train, y_train, X_test, y_test,
    n_estimators=n_estimators,
    max_depth=5,
    learning_rate=0.1,
    eval_interval=eval_interval
)

# 7. 在测试集上评估
dtest = xgb.DMatrix(X_test)
y_pred = np.where(model.predict(dtest) > 0.5, 1, 0)
y_prob = model.predict(dtest)

# 8. 计算测试集指标
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auprc = average_precision_score(y_test, y_prob)

# 9. 打印最终结果
print("\n===== 最终测试集评估指标 =====")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"AUPRC        : {auprc:.4f}")

# 10. 可选：保存训练过程中的指标
train_metrics_df = pd.DataFrame(train_metrics)
print("\n训练过程指标记录:")
print(train_metrics_df)