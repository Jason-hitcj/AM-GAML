import matplotlib.pyplot as plt
import os

def plot_train_val_metrics(metrics_data, save_dir='training_metrics'):

    # 定义x轴（假设是0-100，每10epoch一个点）
    epochs = list(range(0, 101, 10))
    x_ticks = [0, 20, 40, 60, 80, 100]
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 定义绘图样式
    style = {
        'train': {'color': '#1f77b4', 'marker': 'o', 'label': 'Train'},
        'val': {'color': '#ff7f0e', 'marker': 's', 'label': 'Validation'}
    }
    
    # 定义要绘制的5个指标及其显示名称
    metrics = {
        'acc': {'title': 'Accuracy', 'ylim': (0, 1.05)},
        'precision': {'title': 'Precision', 'ylim': (0, 1.05)},
        'recall': {'title': 'Recall', 'ylim': (0, 1.05)},
        'f1': {'title': 'F1 Score', 'ylim': (0, 1.05)},
        'auprc': {'title': 'AUPRC', 'ylim': (0, 1.05)}
    }
    
    # 为每个指标创建图表
    for metric, config in metrics.items():
        plt.figure(figsize=(8, 5))
        
        # 绘制训练集和验证集曲线
        for phase in ['train', 'val']:
            if metric in metrics_data[phase]:
                plt.plot(
                    epochs,
                    metrics_data[phase][metric],
                    label=f"{style[phase]['label']} {config['title']}",
                    color=style[phase]['color'],
                    marker=style[phase]['marker'],
                    markersize=6,
                    linewidth=2
                )
        
        # 设置图表样式
        plt.title(f"{config['title']} Training Progress", fontsize=20, pad=10)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel(config['title'], fontsize=16)
        plt.xticks(x_ticks,fontsize=16)

        plt.yticks(fontsize=16) 
        plt.ylim(config['ylim'])
        plt.grid(True, alpha=0.3)
        
        # 添加图例
        plt.legend(fontsize=16, framealpha=0.9)
        
        # 保存图片
        filename = os.path.join(save_dir, f"{metric}_progress1.png")
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

# 示例使用
if __name__ == "__main__":
    # 示例数据（11个点对应0-100epoch）
    example_data = {
        'train': {
            'acc': [0.5749, 0.9389, 0.9456, 0.9448, 0.9498, 0.9498, 0.9523, 0.9531, 0.9515, 0.9548, 0.9598],
            'precision': [0.6405, 0.9374, 0.9443, 0.9427, 0.9483, 0.9482, 0.9506, 0.9512, 0.9496, 0.9534, 0.9586],
            'recall': [0.5176, 0.9389, 0.9455, 0.9464, 0.9501, 0.9503, 0.9531, 0.9545, 0.9526, 0.9552, 0.9601],
            'f1': [0.4091, 0.9381, 0.9448, 0.9442, 0.9491, 0.9491, 0.9517, 0.9526, 0.9509, 0.9542, 0.9593],
            'auprc': [0.5478, 0.9161, 0.9561, 0.9421, 0.9566, 0.9615, 0.9596, 0.9597, 0.9511, 0.9538, 0.9644]
        },
        'val': {
            'acc': [0.4784, 0.9176, 0.9333, 0.9333, 0.9333, 0.9333, 0.9333, 0.9333, 0.9255, 0.9216, 0.9176],
            'precision': [0.7263, 0.9165, 0.9325, 0.9325, 0.9325, 0.9325, 0.9325, 0.9325, 0.9237, 0.9201, 0.9152],
            'recall': [0.5414, 0.9155, 0.9315, 0.9315, 0.9315, 0.9315, 0.9315, 0.9315, 0.9246, 0.9201, 0.9177],
            'f1': [0.3880, 0.9160, 0.9320, 0.9320, 0.9320, 0.9320, 0.9320, 0.9320, 0.9241, 0.9201, 0.9163],
            'auprc': [0.8710, 0.9421, 0.9430, 0.9451, 0.9336, 0.9171, 0.8992, 0.8832, 0.8785, 0.8724, 0.8727]
        }
    }
    
    plot_train_val_metrics(
        metrics_data=example_data,
        save_dir='training_metrics'
    )