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
from tqdm import tqdm

import pandas as pd
from sklearn.preprocessing import StandardScaler

# 1. 读取包含全部节点特征和标签的数据
df = pd.read_csv("data_new/customer_node_features_gnn.csv")  

# 2. 提取特征列
feature_cols = [f"feature_{i}" for i in range(64)]

# 3. 读取划分好的 ID 列表
train_ids = pd.read_csv('data_new/train_ids.csv')['CUST_ID'].tolist()
val_ids = pd.read_csv('data_new/val_ids.csv')['CUST_ID'].tolist()
test_ids = pd.read_csv('data_new/test_ids.csv')['CUST_ID'].tolist()

# 4. 通过 CUST_ID 直接筛选出对应的数据行
train_df = df[df['CUST_ID'].isin(train_ids)]
val_df = df[df['CUST_ID'].isin(val_ids)]
test_df = df[df['CUST_ID'].isin(test_ids)]

# 5. 提取特征和标签
X_train_raw = train_df[feature_cols].values
y_train = train_df["label"].values

X_val_raw = val_df[feature_cols].values
y_val = val_df["label"].values

X_test_raw = test_df[feature_cols].values
y_test = test_df["label"].values

# 6. 特征归一化（只使用训练集进行拟合）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_val = scaler.transform(X_val_raw)
X_test = scaler.transform(X_test_raw)


# 8. 验证划分结果
print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"验证集形状: X_val={X_val.shape}, y_val={y_val.shape}")
print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")


# 9. XGBoost 训练函数 (已优化)
def train_xgb(X_train, y_train, X_val, y_val, 
              n_estimators=100, max_depth=3, learning_rate=0.1, 
              early_stopping_rounds=20, eval_interval=10):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)
    
    params = {
        'max_depth': max_depth,
        'eta': learning_rate,
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr', # 早停默认使用最后一个指标
        'scale_pos_weight': scale_pos_weight,
        'seed': 42
    }
    
    print("\n===== 开始训练XGBoost模型 =====")
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=eval_interval
    )
    return bst

# ==============================================================================
# 10. GBDT 训练函数 (新增)
# ==============================================================================
def train_gbdt(X_train, y_train, X_val, y_val, 
               n_estimators=100, max_depth=3, learning_rate=0.1, 
               early_stopping_rounds=20):
    """
    使用Scikit-learn的GBDT进行训练，并通过验证集手动实现早停。
    """
    print("\n===== 开始训练GBDT模型 (寻找最佳迭代次数) =====")
    
    # 初始化模型，训练完整的轮数
    gbdt = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    
    # Scikit-learn使用sample_weight处理类别不平衡
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    gbdt.fit(X_train, y_train, sample_weight=sample_weights)

    # 手动早停：在验证集上找到最佳迭代次数
    best_iter = 0
    best_val_aucpr = 0
    patience_counter = 0
    
    # 使用staged_predict_proba获取每一轮的预测结果
    val_probs_staged = gbdt.staged_predict_proba(X_val)
    
    print("在验证集上评估每一轮的效果...")
    for i, y_val_prob in tqdm(enumerate(val_probs_staged), total=n_estimators):
        val_aucpr = average_precision_score(y_val, y_val_prob[:, 1])
        if val_aucpr > best_val_aucpr:
            best_val_aucpr = val_aucpr
            best_iter = i + 1
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_rounds:
            print(f"早停触发！在第 {i+1} 轮停止。")
            break
            
    print(f"GBDT最佳迭代次数为: {best_iter} (AUPRC: {best_val_aucpr:.4f})")
    
    # 使用最佳迭代次数重新训练最终模型
    print("使用最佳迭代次数重新训练最终GBDT模型...")
    final_gbdt = GradientBoostingClassifier(
        n_estimators=best_iter, # 使用最佳轮数
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42
    )
    final_gbdt.fit(X_train, y_train, sample_weight=sample_weights)
    
    return final_gbdt, best_iter


# 辅助函数：用于评估和打印结果
def evaluate_and_print(model_name, model, X_test, y_test, best_iter=None, is_xgb=False):
    print(f"\n===== {model_name} 最终测试集评估指标 =====")
    if best_iter:
        print(f"最佳迭代次数: {best_iter}")
    
    # 优化预测代码，只调用一次predict
    if is_xgb:
        dtest = xgb.DMatrix(X_test)
        y_prob = model.predict(dtest, iteration_range=(0, best_iter))
    else:
        y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = np.where(y_prob > 0.5, 1, 0)
    
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


# ==============================================================================
#                             模型训练与评估
# ==============================================================================

# --- XGBoost ---
xgb_model = train_xgb(
    X_train, y_train, X_val, y_val,
    n_estimators=200,          # 可以适当增加总轮数
    max_depth=5,
    learning_rate=0.1,
    early_stopping_rounds=20   # 设置一个合理的早停耐心值
)
evaluate_and_print("XGBoost", xgb_model, X_test, y_test, best_iter=xgb_model.best_iteration, is_xgb=True)


# --- GBDT ---
gbdt_model, gbdt_best_iter = train_gbdt(
    X_train, y_train, X_val, y_val,
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    early_stopping_rounds=20
)
evaluate_and_print("GBDT", gbdt_model, X_test, y_test, best_iter=gbdt_best_iter)