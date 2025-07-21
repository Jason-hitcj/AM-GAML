import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
import xgboost as xgb

# 1. 读取数据
df = pd.read_csv("data_new/customer_node_features_gnn.csv")  

# 2. 提取特征和标签
feature_cols = [f"feature_{i}" for i in range(64)]
X = df[feature_cols].values
y = df["label"].values

# 3. 读取ID列表
train_ids = pd.read_csv('data_new/train_ids.csv')['CUST_ID'].tolist()
val_ids = pd.read_csv('data_new/val_ids.csv')['CUST_ID'].tolist()
test_ids = pd.read_csv('data_new/test_ids.csv')['CUST_ID'].tolist()

# 4. 映射 ID 到索引
id_to_index = {cust_id: idx for idx, cust_id in enumerate(df['CUST_ID'])}
train_indices = [id_to_index[id_] for id_ in train_ids]
val_indices = [id_to_index[id_] for id_ in val_ids]
test_indices = [id_to_index[id_] for id_ in test_ids]

# 5. 合并训练集和验证集
train_full_indices = train_indices + val_indices
X_train_full_raw = X[train_full_indices]
y_train_full = y[train_full_indices]
X_test_raw = X[test_indices]
y_test = y[test_indices]

# 6. 特征归一化
scaler = StandardScaler()
X_train_full = scaler.fit_transform(X_train_full_raw)
X_test = scaler.transform(X_test_raw)

# 7. XGBoost训练（不使用验证集）
def train_xgb_no_val(X_train, y_train, n_estimators=200, max_depth=5, learning_rate=0.1):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

    params = {
        'max_depth': max_depth,
        'eta': learning_rate,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',
        'scale_pos_weight': scale_pos_weight,
        'seed': 42
    }

    print("\n===== 开始训练XGBoost模型（取消验证集）=====")
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train')],
        verbose_eval=10
    )
    return bst

# 8. GBDT训练（不使用验证集，不早停）
def train_gbdt_no_val(X_train, y_train, n_estimators=200, max_depth=5, learning_rate=0.1):
    print("\n===== 开始训练GBDT模型（取消验证集）=====")
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

    gbdt = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    gbdt.fit(X_train, y_train, sample_weight=sample_weights)
    return gbdt

# 9. 模型评估
def evaluate_and_print(model_name, model, X_test, y_test, best_iter=None, is_xgb=False):
    print(f"\n===== {model_name} 测试集评估 =====")
    if best_iter:
        print(f"使用轮数: {best_iter}")

    if is_xgb:
        dtest = xgb.DMatrix(X_test)
        y_prob = model.predict(dtest, iteration_range=(0, best_iter))
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (y_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auprc = average_precision_score(y_test, y_prob)

    print(f"Accuracy     : {acc:.4f}")
    print(f"Precision    : {prec:.4f}")
    print(f"Recall       : {rec:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"AUPRC        : {auprc:.4f}")

# 10. 模型训练与评估
n_estimators_final = 200

# --- XGBoost ---
xgb_model = train_xgb_no_val(X_train_full, y_train_full, n_estimators=n_estimators_final)
evaluate_and_print("XGBoost", xgb_model, X_test, y_test, best_iter=n_estimators_final, is_xgb=True)

# --- GBDT ---
gbdt_model = train_gbdt_no_val(X_train_full, y_train_full, n_estimators=n_estimators_final)
evaluate_and_print("GBDT", gbdt_model, X_test, y_test, best_iter=n_estimators_final)
