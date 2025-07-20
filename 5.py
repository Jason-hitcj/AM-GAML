import pandas as pd

# 读取 CSV 文件
file_path = "data/customer_node_features_supervised.csv"
df = pd.read_csv(file_path)

# 计算 CUST_TYPE 的百分比（小数形式）
cust_type_percent = df['CUST_TYPE'].value_counts(normalize=True) * 100

print("CUST_TYPE 占比（百分比）：")
print(cust_type_percent)