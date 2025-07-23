import matplotlib.pyplot as plt
import os

def plot_and_save_metrics_separately(methods_metrics, save_dir='img'):
    """
    绘制每个指标的折线图，每个方法一条线，本文方法用红色突出显示，
    并单独绘制图例说明。
    """
    # 统一指标映射：原始key -> 目标标准显示名
    metric_name_map = {
        'acc': 'Accuracy',
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall',
        'f1': 'F1 Score',
        'f1 score': 'F1 Score',
        'auprc': 'AUPRC'
    }

    # 构造标准指标名列表
    standard_metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUPRC']

    # 获取 epoch 刻度
    epochs = [i * 10 for i in range(len(next(iter(methods_metrics.values()))['acc']))]
    x_ticks = [i for i in epochs if i % 20 == 0]

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 样式池
    base_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd',
                   '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    linestyles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1))]
    markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'H']

    # 获取方法列表
    method_names = list(methods_metrics.keys())

    # 本文方法为最后一个
    highlight_method = method_names[-1]

    # 遍历标准指标名
    for std_metric in standard_metric_names:
        plt.figure(figsize=(8, 5))

        for i, method in enumerate(method_names):
            # 找出该方法中匹配的指标key（大小写不一，可能为f1或F1 Score）
            metric_key = None
            for key in methods_metrics[method].keys():
                if metric_name_map.get(key.lower(), key) == std_metric:
                    metric_key = key
                    break
            if not metric_key:
                continue

            values = methods_metrics[method][metric_key]

            if method == highlight_method:
                plt.plot(
                    epochs, values, label=method,
                    color='red', linestyle='-',
                    marker='o', markersize=8,
                    linewidth=3, zorder=10
                )
            else:
                plt.plot(
                    epochs, values, label=method,
                    color=base_colors[i % len(base_colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markers[i % len(markers)],
                    markersize=7, linewidth=2
                )

        # 设置标题和坐标轴
        plt.title(f'{std_metric} Comparison', fontsize=20, pad=10)
        plt.xlabel('Epoch', fontsize=16)
        plt.ylabel(std_metric, fontsize=16)
        plt.xticks(x_ticks)
        plt.tick_params(axis='both', which='major', labelsize=12)

        # 设置 y 轴范围，保证最小值也能体现
        all_values = []
        for method in method_names:
            for key in methods_metrics[method]:
                if metric_name_map.get(key.lower(), key) == std_metric:
                    all_values.extend(methods_metrics[method][key])
        y_min = max(0.0-0.1, min(all_values) - 0.05)
        y_max = min(1.02, max(all_values) + 0.05)
        plt.ylim(y_min, y_max)

        # 保存图片（不加图例）
        filename = os.path.join(save_dir, f'{std_metric.replace(" ", "_")}_comparison.png')
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        print(f"已保存: {filename}")
        plt.close()

    # 单独绘制图例
    plt.figure(figsize=(10, 1.5))
    for i, method in enumerate(method_names):
        if method == highlight_method:
            plt.plot([], [], color='red', linestyle='-', marker='o',
                     label=method, linewidth=3, markersize=8)
        else:
            plt.plot([], [], color=base_colors[i % len(base_colors)],
                     linestyle=linestyles[i % len(linestyles)],
                     marker=markers[i % len(markers)],
                     label=method, linewidth=2, markersize=7)

    plt.axis('off')
    plt.legend(ncol=4, loc='center', fontsize=14, framealpha=0.9)
    plt.tight_layout()
    legend_path = os.path.join(save_dir, 'methods_legend.png')
    plt.savefig(legend_path, dpi=600, bbox_inches='tight')
    print(f"已保存图例说明: {legend_path}")
    plt.close()



# 示例使用（完全适配你的数据格式）
if __name__ == "__main__":
    # 你的实际数据格式示例（11个点对应0,10,20,...,100epoch）
    methods_metrics = {
        'LSTM': {
            'acc':       [0.8067, 0.8067, 0.8143, 0.8143, 0.8668, 0.8758, 0.8873, 0.8771, 0.8848, 0.8860, 0.8873],
            'precision': [0.4033, 0.4033, 0.9065, 0.9065, 0.8878, 0.8909, 0.9019, 0.9042, 0.8996, 0.8748, 0.8880],
            'recall':    [0.5000, 0.5000, 0.5199, 0.5199, 0.6682, 0.6939, 0.7237, 0.6922, 0.7171, 0.7355, 0.7313],
            'f1':        [0.4465, 0.4465, 0.4866, 0.4866, 0.7116, 0.7404, 0.7724, 0.7401, 0.7655, 0.7787, 0.7773],
            'auprc':     [0.1734, 0.3368, 0.4659, 0.5312, 0.7365, 0.7244, 0.7541, 0.7796, 0.7413, 0.7575, 0.7951]
        },
        'GraphSAGE': {
            'acc': [0.7875, 0.8246, 0.8886, 0.8899, 0.9091, 0.9245, 0.9488, 0.9488, 0.9501, 0.9449, 0.9501],
            'precision': [0.0000, 0.0000, 0.8289, 0.8312, 0.8056, 0.8000, 0.8212, 0.8170, 0.8224, 0.7975, 0.8182],
            'recall': [0.0000, 0.0000, 0.4599, 0.4672, 0.6350, 0.7591, 0.9051, 0.9124, 0.9124, 0.9197, 0.9197],
            'f1': [0.0000, 0.0000, 0.5915, 0.5981, 0.7102, 0.7790, 0.8611, 0.8621, 0.8651, 0.8542, 0.8660],
            'auprc': [0.1083, 0.7025, 0.7320, 0.7793, 0.7920, 0.8018, 0.8127, 0.8433, 0.8380, 0.8942, 0.8781],
        },
        'GAT': {
            'acc': [0.8246, 0.8246, 0.8246, 0.8886, 0.8886, 0.8873, 0.8860, 0.8860, 0.8873, 0.8873, 0.8912],
            'precision': [0.0000, 0.0000, 0.0000, 0.8049, 0.8049, 0.7952, 0.7857, 0.7857, 0.7634, 0.7579, 0.7653],
            'recall': [0.0000, 0.0000, 0.0000, 0.4818, 0.4818, 0.4818, 0.4818, 0.4818, 0.5182, 0.5255, 0.5474],
            'f1': [0.0000, 0.0000, 0.0000, 0.6027, 0.6027, 0.6000, 0.5973, 0.5973, 0.6174, 0.6207, 0.6383],
            'auprc': [0.1171, 0.2098, 0.6296, 0.6925, 0.6930, 0.6626, 0.6816, 0.6575, 0.7076, 0.6544, 0.6734],
        },
        'HGT': {
            'acc': [0.8246, 0.8835, 0.8860, 0.8912, 0.8848, 0.8937, 0.8924, 0.8976, 0.8912, 0.8937, 0.8963],
            'precision': [0.0000, 0.7300, 0.7449, 0.7653, 0.8310, 0.7596, 0.8442, 0.7615, 0.7063, 0.8000, 0.7692],
            'recall': [0.0000, 0.5328, 0.5328, 0.5474, 0.4307, 0.5766, 0.4745, 0.6058, 0.6496, 0.5255, 0.5839],
            'f1': [0.0000, 0.6160, 0.6213, 0.6383, 0.5673, 0.6556, 0.6075, 0.6748, 0.6768, 0.6344, 0.6639],
            'auprc': [0.1186, 0.6693, 0.6771, 0.6872, 0.6698, 0.7295, 0.7315, 0.7043, 0.7233, 0.7040, 0.7443],
        },
        'RGCN': {
            'acc': [0.8246, 0.8912, 0.8963, 0.9065, 0.9245, 0.9385, 0.9501, 0.9513, 0.9488, 0.9552, 0.9539],
            'precision': [0.0000, 0.8171, 0.7979, 0.7807, 0.7955, 0.7947, 0.8101, 0.8194, 0.8089, 0.8355, 0.8344],
            'recall': [0.0000, 0.4891, 0.5474, 0.6496, 0.7664, 0.8759, 0.9343, 0.9270, 0.9270, 0.9270, 0.9197],
            'f1': [0.0000, 0.6119, 0.6494, 0.7092, 0.7807, 0.8333, 0.8678, 0.8699, 0.8639, 0.8789, 0.8750],
            'auprc': [0.1324, 0.6885, 0.7666, 0.8274, 0.8446, 0.8645, 0.8943, 0.9176, 0.9309, 0.9393, 0.9451],
        },
        'MAGNN': {
            'acc': [0.8246, 0.8246, 0.8796, 0.8848, 0.8848, 0.8873, 0.8873, 0.8860, 0.8886, 0.8937, 0.8937],
            'precision': [0.0000, 0.0000, 0.8308, 0.7975, 0.7975, 0.8025, 0.8025, 0.7927, 0.7976, 0.8068, 0.8068],
            'recall': [0.0000, 0.0000, 0.3942, 0.4599, 0.4599, 0.4745, 0.4745, 0.4745, 0.4891, 0.5182, 0.5182],
            'f1': [0.0000, 0.0000, 0.5347, 0.5833, 0.5833, 0.5963, 0.5963, 0.5936, 0.6063, 0.6311, 0.6311],
            'auprc': [0.1411, 0.2617, 0.5991, 0.6512, 0.6641, 0.6735, 0.6743, 0.6799, 0.6805, 0.6819, 0.6853]
        },
        'AM-GAML': {
            'acc':       [0.6223, 0.8848, 0.8873, 0.8924, 0.8937, 0.8912, 0.8886, 0.9321, 0.9488, 0.9770, 0.9808],
            'precision': [0.5223, 0.8898, 0.8663, 0.8591, 0.8562, 0.8490, 0.8150, 0.9123, 0.8960, 0.9483, 0.9548],
            'recall':    [0.5317, 0.7221, 0.7464, 0.7722, 0.7806, 0.7790, 0.8554, 0.8623, 0.9657, 0.9832, 0.9881],
            'f1':        [0.5130, 0.7689, 0.7863, 0.8050, 0.8103, 0.8069, 0.8325, 0.8844, 0.9248, 0.9645, 0.9703],
            'auprc':     [0.2188, 0.7633, 0.7828, 0.8073, 0.8194, 0.8328, 0.8113, 0.9126, 0.9237, 0.9764, 0.9690]
        }
    }
    
    plot_and_save_metrics_separately(
        methods_metrics=methods_metrics,
        save_dir='img_all'
    )