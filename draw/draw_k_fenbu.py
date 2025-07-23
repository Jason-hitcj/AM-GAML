import matplotlib.pyplot as plt
import numpy as np
import os

def plot_all_metrics_one_figure(k_values, metrics_data, save_path='img_knn/all_metrics_comparison.png'):
    """
    在一张图中绘制所有指标随K值的变化（定制颜色版）
    """
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 创建图形
    plt.figure(figsize=(10, 6))
    
    # 定制颜色和样式配置
    color_map = {
        'Accuracy': 'red',          # 红色
        'Precision': 'gold',   # 黄色
        'Recall': 'green',     # 绿色
        'F1 Score': 'blue',          # 蓝色
        'AUPRC': 'brown'       # 棕色
    }
    
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    markers = ['o', 's', 'D', '^', 'v']
    
    # 按指定顺序绘制指标（确保图例顺序一致）
    for i, metric_name in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUPRC']):
        plt.plot(
            k_values, 
            metrics_data[metric_name],
            label=metric_name,
            color=color_map[metric_name],
            linestyle=line_styles[i % len(line_styles)],
            marker=markers[i % len(markers)],
            markersize=8,
            linewidth=2
        )
    
    # 设置图表样式
    plt.title('Metrics Comparison Under Different K Values', fontsize=20, pad=12)
    plt.xlabel('K Value', fontsize=18)
    plt.ylabel('Metric Value', fontsize=18)
    plt.xticks(k_values,fontsize=16)

    
    # 调整y轴范围
    all_values = np.concatenate(list(metrics_data.values()))
    plt.ylim(
        max(0, np.min(all_values) - 0.05),
        min(1.02, np.max(all_values) + 0.05)
    )
    plt.yticks(fontsize=16)
    
    # 添加图例和网格
    plt.legend(fontsize=16, framealpha=0.9, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 保存输出
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"图表已保存到: {save_path}")
    plt.close()

# 示例使用
if __name__ == "__main__":
    # K值范围
    k_values = [3, 4, 5, 6, 7, 8, 9, 10]
    
    # 更新后的数据（与表格一致）
    metrics_data = {
        'Accuracy': [0.9559, 0.9609, 0.9604, 0.9593, 0.9632, 0.9674, 0.9666, 0.9643],
        'Precision': [0.906, 0.9215, 0.9216, 0.9187, 0.9183, 0.9276, 0.9241, 0.9196],
        'Recall': [0.9359, 0.9346, 0.9313, 0.9306, 0.9505, 0.9553, 0.9583, 0.9541],
        'F1 Score': [0.92, 0.9276, 0.9263, 0.9243, 0.9334, 0.9407, 0.9398, 0.9356],
        'AUPRC': [0.9394, 0.9373, 0.9628, 0.9640, 0.9301, 0.9577, 0.9210, 0.9539]  # 这部分数据保持不变，因为表格中没有
    }
    
    plot_all_metrics_one_figure(
        k_values=k_values,
        metrics_data=metrics_data,
        save_path='img_knn/all_metrics_comparison_new.png'
    )