import matplotlib.pyplot as plt
import os

def plot_train_val_metrics(metrics_data, save_dir='training_metrics'):
    # 定义x轴（假设是0-100，每10epoch一个点）
    epochs = list(range(0, 101, 10))
    x_ticks = [0, 20, 40, 60, 80, 100]

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 美观的配色和标记
    style = {
        'train': {'color': '#4C72B0', 'marker': 'o', 'label': 'Train'},
        'val': {'color': '#DD8452', 'marker': 's', 'label': 'Validation'}
    }

    # 要绘制的指标及其显示名称（不再固定ylim）
    metrics = {
        'acc': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'auprc': 'AUPRC'
    }

    for metric, title in metrics.items():
        plt.figure(figsize=(8, 5))

        # 获取所有值以便设定ylim范围
        all_values = []
        for phase in ['train', 'val']:
            if metric in metrics_data[phase]:
                values = metrics_data[phase][metric]
                all_values.extend(values)
                plt.plot(
                    epochs,
                    values,
                    label=f"{style[phase]['label']} {title}",
                    color=style[phase]['color'],
                    marker=style[phase]['marker'],
                    markersize=6,
                    linewidth=2.5
                )

        # 设置图表样式
        plt.title(f"{title} Training Progress", fontsize=20, pad=10)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel(title, fontsize=16)
        plt.xticks(x_ticks, fontsize=14)
        plt.yticks(fontsize=14)

        # 纵轴范围自动适应上升趋势（设置下限为最小值-0.05，但不低于0）
        ymin = max(min(all_values) - 0.05, 0)
        ymax = min(max(all_values) + 0.05, 1.05)
        plt.ylim((ymin, ymax))

        plt.grid(True, linestyle='--', alpha=0.3)

        # 图例放在右下角
        plt.legend(fontsize=14, loc='lower right', framealpha=0.95)

        # 保存图片
        filename = os.path.join(save_dir, f"{metric}_progress.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# 示例使用
if __name__ == "__main__":
    # 训练数据（11个点对应0-100epoch）
    training_data = {
        'train': {
            'acc': [0.6223, 0.8848, 0.8873, 0.8924, 0.8937, 0.8912, 0.8886, 0.9321, 0.9488, 0.9770, 0.9808],
            'precision': [0.5223, 0.8898, 0.8663, 0.8591, 0.8562, 0.8490, 0.8150, 0.9123, 0.8960, 0.9483, 0.9548],
            'recall': [0.5317, 0.7221, 0.7464, 0.7722, 0.7806, 0.7790, 0.8554, 0.8623, 0.9657, 0.9832, 0.9881],
            'f1': [0.5130, 0.7689, 0.7863, 0.8050, 0.8103, 0.8069, 0.8325, 0.8844, 0.9248, 0.9645, 0.9703],
            'auprc': [0.2188, 0.7633, 0.7828, 0.8073, 0.8194, 0.8328, 0.8113, 0.9126, 0.9237, 0.9764, 0.9690]
        },
        'val': {
            'acc': [0.8413, 0.8345, 0.9113, 0.9044, 0.9044, 0.9027, 0.9096, 0.9300, 0.9471, 0.9590, 0.9642],
            'precision': [0.4206, 0.7261, 0.8427, 0.8326, 0.8305, 0.8260, 0.8329, 0.8700, 0.8836, 0.9060, 0.9179],
            'recall': [0.5000, 0.8362, 0.8120, 0.7905, 0.7949, 0.7939, 0.8241, 0.8668, 0.9380, 0.9538, 0.9569],
            'f1': [0.4569, 0.7555, 0.8262, 0.8092, 0.8110, 0.8086, 0.8284, 0.8684, 0.9076, 0.9277, 0.9359],
            'auprc': [0.1233, 0.8009, 0.8207, 0.8116, 0.8104, 0.8171, 0.8435, 0.8708, 0.8664, 0.9035, 0.9312]
        }
    }

    plot_train_val_metrics(training_data, save_dir='training')
