import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


import pandas as pd
from sklearn.model_selection import train_test_split

# 载入数据
# df = pd.read_csv('data/preprocessed_data_ml_full.csv')
# target_column = 'target'

# # 分离特征和目标
# X = df.drop(target_column, axis=1)
# y = df[target_column]

df = pd.read_csv('data/customer_node_features_supervised.csv')
target_column = 'label'


# 分离特征和目标
# 提取特征得到 numpy 数组
features_numpy = df[[f'feature_{i}' for i in range(64)]].values

# 将 numpy 数组转换回 DataFrame
columns = [f'feature_{i}' for i in range(64)]
X = pd.DataFrame(features_numpy, columns=columns)
y = df[target_column]

# 检查特征矩阵
print("\n=== 特征矩阵检查 ===")
print(f"特征矩阵形状: {X.shape}")
print("\n特征数据类型:")
print(X.dtypes)
print("\n缺失值统计:")
print(X.isnull().sum())

# 检查是否有特征与目标变量完全一致
leakage_features = [col for col in X.columns if X[col].equals(y)]
if leakage_features:
    print(f"🚨 数据泄露！以下特征与目标变量完全相同: {leakage_features}")
else:
    print("✅ 未发现特征与目标变量完全一致")

# 计算数值特征与目标的绝对相关性
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_features:
    corr = abs(X[col].corr(y))
    if corr > 0.99:
        print(f"⚠️ 特征 '{col}' 与目标变量的绝对相关性高达: {corr:.4f}")


# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=9,
    stratify=y  # 如果分类不平衡建议添加分层抽样
)

print("\n=== 数据分割结果 ===")
print(f"训练集形状: X_train={X_train.shape}, y_train={y_train.shape}")
print(f"测试集形状: X_test={X_test.shape}, y_test={y_test.shape}")

# 确认X_test和y_test的ID与训练集不同
print(f"X_train ID: {id(X_train)}, X_test ID: {id(X_test)}")
print(f"y_train ID: {id(y_train)}, y_test ID: {id(y_test)}")

# 检查测试集样本是否真的独立
assert len(set(X_test.index) & set(X_train.index)) == 0, "测试集与训练集存在重叠样本！"

# 如果需要合并为完整DataFrame用于分析/可视化
train_df = X_train.copy()
train_df[target_column] = y_train

test_df = X_test.copy()
test_df[target_column] = y_test

print("\n训练集DataFrame预览:")
print(train_df.head())


# 1. Gradient Boosting Decision Tree (GBDT)

gbdt_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gbdt_model.fit(X_train, y_train)
y_pred_gbdt = gbdt_model.predict(X_test)



# 2. LightGBM

lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train, y_train)
y_pred_lgb = lgb_model.predict(X_test)



# 3. XGBoost

xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)



# 4. AdaBoost

ada_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=3), n_estimators=100, random_state=42)
ada_model.fit(X_train, y_train)
y_pred_ada = ada_model.predict(X_test)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
# 如果是二分类任务，可以简化指标计算：
def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    try:
        auc_roc = roc_auc_score(y_true, y_pred)
        print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC-ROC: {auc_roc:.4f}")
    except:
        print(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


print("Performance Comparison:")

evaluate_model(y_test, y_pred_gbdt, "GBDT")
evaluate_model(y_test, y_pred_lgb, "LightGBM")
evaluate_model(y_test, y_pred_xgb, "XGBoost")
evaluate_model(y_test, y_pred_ada, "AdaBoost")


plt.style.use('ggplot')  # 使用有效的样式
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()



def plot_predictions(ax, y_true, y_pred, model_name):
    sns.scatterplot(x=y_true, y=y_pred, ax=ax)
    ax.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', label='Perfect Prediction')
    ax.set_title(f'{model_name} Predictions vs Actual')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.legend()



# 预测与实际值对比

plot_predictions(axes[0], y_test, y_pred_gbdt, 'GBDT')
plot_predictions(axes[1], y_test, y_pred_lgb, 'LightGBM')
plot_predictions(axes[2], y_test, y_pred_xgb, 'XGBoost')
plot_predictions(axes[3], y_test, y_pred_ada, 'AdaBoost')



plt.tight_layout()
# 保存文件（支持PNG/PDF/SVG等格式）
plt.savefig("img/output1.png", dpi=300, bbox_inches="tight")  # bbox_inches防止截断

# 主动关闭图形，避免内存占用
plt.close()  # 关键！阻止图形显示



# 预测值分布图

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()



def plot_prediction_distribution(ax, y_pred, model_name):
    sns.histplot(y_pred, ax=ax, kde=True, bins=30)
    ax.set_title(f'{model_name} Predicted Values Distribution')
    ax.set_xlabel('Predicted Values')
    ax.set_ylabel('Frequency')



# 绘制预测值的分布

plot_prediction_distribution(axes[0], y_pred_gbdt, 'GBDT')
plot_prediction_distribution(axes[1], y_pred_lgb, 'LightGBM')
plot_prediction_distribution(axes[2], y_pred_xgb, 'XGBoost')
plot_prediction_distribution(axes[3], y_pred_ada, 'AdaBoost')


plt.tight_layout()
# 保存文件（支持PNG/PDF/SVG等格式）
plt.savefig("img/output2.png", dpi=300, bbox_inches="tight")  # bbox_inches防止截断

# 主动关闭图形，避免内存占用
plt.close()  # 关键！阻止图形显示

def plot_residuals(y_true, y_pred, model_name):
    plt.figure(figsize=(8, 6))
    residuals = y_true - y_pred
    sns.scatterplot(x=y_pred, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{model_name} Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

# 分别绘制每个模型的残差图
plot_residuals(y_test, y_pred_gbdt, 'GBDT')
plot_residuals(y_test, y_pred_lgb, 'LightGBM')
plot_residuals(y_test, y_pred_xgb, 'XGBoost')
plot_residuals(y_test, y_pred_ada, 'AdaBoost')
plt.tight_layout()
# 保存文件（支持PNG/PDF/SVG等格式）
plt.savefig("img/output3.png", dpi=300, bbox_inches="tight")  # bbox_inches防止截断

# 主动关闭图形，避免内存占用
plt.close()  # 关键！阻止图形显示
