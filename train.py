import torch
import pandas as pd

try:
    # 加载异构图数据
    data = torch.load('hetero_graph_1.pt')

    all_data = []
    label_stats = {}

    # 遍历所有节点类型
    for node_type in data.node_types:
        node_data = data[node_type]
        if 'cust_id' in node_data and 'split' in node_data and 'label' in node_data:
            cust_id = node_data['cust_id'].tolist() if isinstance(node_data['cust_id'], torch.Tensor) else node_data['cust_id']
            split = node_data['split'].tolist() if isinstance(node_data['split'], torch.Tensor) else node_data['split']
            label = node_data['label'].tolist() if isinstance(node_data['label'], torch.Tensor) else node_data['label']
            df = pd.DataFrame({
                'node_type': [node_type] * len(cust_id),
                'cust_id': cust_id,
                'split': split,
                'label': label
            })
            all_data.append(df)

            # 统计 label 为 0 和 1 的数量
            label_0_count = label.count(0)
            label_1_count = label.count(1)
            label_stats[node_type] = {
                'label_0_count': label_0_count,
                'label_1_count': label_1_count
            }

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    # 保存到 CSV 文件
    combined_df.to_csv('node_data.csv', index=False)
    print("数据已成功保存到 node_data.csv")

    # 打印 label 统计信息
    print("每个节点类型的 label 统计信息：")
    for node_type, stats in label_stats.items():
        print(f"{node_type}: label 0 数量 = {stats['label_0_count']}, label 1 数量 = {stats['label_1_count']}")

except FileNotFoundError:
    print("错误：未找到 hetero_graph_1.pt 文件。")
except Exception as e:
    print(f"发生未知错误：{e}")
    