import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
import os


# 设置SCI论文的绘图样式
def set_sci_style():
    """设置SCI论文的绘图样式"""
    plt.rcParams.update({
        # 'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 8,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 8,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (10, 6)
    })


def parse_log_file(log_path):
    """解析训练日志文件"""
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # 存储解析结果
    val_loss = []  # 验证集loss
    val_miou = []  # 验证集mIoU (每30个epoch)

    # 正则表达式模式
    val_miou_pattern = r'epoch (\d+), val, mean loss ([\d.]+), cmiou: ([\d.]+)'
    'epoch 5, val, mean loss 0.684, total time 2.91'
    val_loss_epochs = []
    test_loss_epochs = []
    val_miou_epochs = []
    test_miou_epochs = []
    v = 1
    for i, line in enumerate(lines):
        if v:
            val_miou_match = re.search(val_miou_pattern, line)
            if val_miou_match and 'class:' not in line:
                epoch = int(val_miou_match.group(1))
                loss = float(val_miou_match.group(2))
                miou = float(val_miou_match.group(3))
                val_loss.append((epoch, loss))
                val_loss_epochs.append(epoch)
                val_miou.append((epoch, miou))
                val_miou_epochs.append(epoch)
                v = 0
        else:
            test_miou_match = re.search(test_miou_pattern, line)
            if test_miou_match and 'class:' not in line:
                epoch = int(test_miou_match.group(1))
                loss = float(test_miou_match.group(2))
                miou = float(test_miou_match.group(3))

                v = 1

    return {
        'val_loss': val_loss,
        'val_miou': val_miou,
        'val_loss_epochs': val_loss_epochs,
        'val_miou_epochs': val_miou_epochs,
    }


def plot_convergence_curve(experiments, save_path=None):
    """绘制验证集loss收敛曲线"""
    set_sci_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for i, (exp_name, log_path) in enumerate(experiments.items()):
        data = parse_log_file(log_path)

        if not data['val_loss']:
            print(f"警告: {exp_name} 没有找到验证集loss数据")
            continue

        epochs = [x[0] for x in data['val_loss']]
        losses = [x[1] for x in data['val_loss']]
        # epochs = epochs[100::10]
        # losses = losses[100::10]

        ax.plot(epochs, losses, label=exp_name, color=colors[i],
                linewidth=2, marker='o', markersize=4)

    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=14, fontweight='bold')
    ax.set_title('Validation Loss Convergence Curve', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 设置科学计数法
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.show()


def plot_miou_comparison(experiments, save_path=None):
    """绘制验证集和测试集mIoU对比图"""
    set_sci_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for i, (exp_name, log_path) in enumerate(experiments.items()):
        data = parse_log_file(log_path)

        # 绘制验证集mIoU
        if data['val_miou']:
            val_epochs = [x[0] for x in data['val_miou']]
            val_mious = [x[1] for x in data['val_miou']]
            ax1.plot(val_epochs, val_mious, label=exp_name, color=colors[i],
                     linewidth=2, marker='s', markersize=5)

        # 绘制测试集mIoU
        if data['test_miou']:
            test_epochs = [x[0] for x in data['test_miou']]
            test_mious = [x[1] for x in data['test_miou']]
            ax2.plot(test_epochs, test_mious, label=exp_name, color=colors[i],
                     linewidth=2, marker='^', markersize=5)

    # 设置验证集子图
    ax1.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax1.set_ylabel('mIoU', fontsize=14, fontweight='bold')
    # ax1.set_title('Validation mIoU (Every 30 Epochs)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.8, 0.9)  # 根据实际数据调整

    # 设置测试集子图
    ax2.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax2.set_ylabel('mIoU', fontsize=14, fontweight='bold')
    # ax2.set_title('Test mIoU (Every 30 Epochs)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.8, 0.9)  # 根据实际数据调整

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.show()


def plot_combined_miou(experiments, save_path=None):
    set_sci_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, (exp_name, log_path) in enumerate(experiments.items()):
        data = parse_log_file(log_path)

        # # 绘制验证集mIoU
        # if data['val_miou']:
        #     val_epochs = [x[0] for x in data['val_miou']]
        #     val_mious = [x[1] for x in data['val_miou']]
        #     ax.plot(val_epochs, val_mious, label=f'{exp_name} (Val)',
        #             color=colors[i], linewidth=2, marker=markers[i % len(markers)],
        #             markersize=6, linestyle='-')

        # 绘制测试集mIoU
        if data['test_miou']:
            test_epochs = [x[0] for x in data['test_miou']]
            test_mious = [x[1] for x in data['test_miou']]
            ax.plot(test_epochs, test_mious, label=f'{exp_name} (Test)',
                    color=colors[i], linewidth=2, marker=markers[i % len(markers)],
                    markersize=6, linestyle='-')


    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('mIoU', fontsize=14)
    # ax.set_title('mIoU Comparison: Validation vs Test Set', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # # ----👇 新增：移除图例中的 marker（最小改动方案）👇----
    # legend = ax.legend()
    # for lh in legend.legendHandles:
    #     try:
    #         lh.set_marker(None)
    #     except Exception:
    #         pass
    # # ----👆 到这里结束 👆----
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.show()

def plot_combined_loss(experiments, save_path=None):

    set_sci_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']

    for i, (exp_name, log_path) in enumerate(experiments.items()):
        data = parse_log_file(log_path)

        # # 绘制验证集mIoU
        # if data['val_loss']:
        #     val_epochs = [x[0] for x in data['val_loss']]
        #     val_mious = [x[1] for x in data['val_loss']]
        #     ax.plot(val_epochs, val_mious, label=f'{exp_name} (Val)',
        #             color=colors[i], linewidth=2, marker=markers[i % len(markers)],
        #             markersize=6, linestyle='-')

        if data['test_loss']:
            test_epochs = [x[0] for x in data['test_loss']]
            test_mious = [x[1] for x in data['test_loss']]
            ax.plot(test_epochs, test_mious, label=f'{exp_name} (Test)',
                    color=colors[i], linewidth=2, marker=markers[i % len(markers)],
                    markersize=6, linestyle='-')


    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=14)
    # ax.set_title('mIoU Comparison: Validation vs Test Set', fontsize=16, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # ----👇 新增：移除图例中的 marker（最小改动方案）👇----
    # legend = ax.legend()
    # for lh in legend.legendHandles:
    #     try:
    #         lh.set_marker(None)
    #     except Exception:
    #         pass
    # # ----👆 到这里结束 👆----
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 定义实验配置
    path = '../save/mobile_sam_adapter/adapter_branch_AGFF_0.6866/train_log.txt'
    experiments = {
        "MobileSAM-Adapter": '../save/mobile_sam_adapter/adapter/train_log.txt',
        "DM-SAM-A": '../save/mobile_sam_adapter/adapter_branch_0.6781/train_log.txt',
        "DMA": '../save/mobile_sam_adapter/adapter_branch_AGFF_0.6866/train_log.txt',
    }

    # 检查文件是否存在
    valid_experiments = {}
    for exp_name, log_path in experiments.items():
        if os.path.exists(log_path):
            valid_experiments[exp_name] = log_path
        else:
            print(f"警告: 文件不存在 - {log_path}")

    if not valid_experiments:
        print("错误: 没有有效的实验文件")
    else:
        # 打印数据统计
        for exp_name, log_path in valid_experiments.items():
            data = parse_log_file(log_path)
            print(f"\n{exp_name} 数据统计:")
            print(f"验证集loss点数: {len(data['val_loss'])}")
            print(f"验证集mIoU点数: {len(data['val_miou'])}")
            print(f"测试集mIoU点数: {len(data['test_miou'])}")
        # 绘制验证集loss收敛曲线
        # plot_convergence_curve(valid_experiments, "validation_loss_convergence.png")
        #
        # # 绘制分开的mIoU对比图
        # plot_miou_comparison(valid_experiments, "miou_comparison.png")
        #
        # # 绘制合并的mIoU对比图
        plot_combined_miou(valid_experiments, "test_miou.png")

        plot_combined_loss(valid_experiments, "test_loss.png")


