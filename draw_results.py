import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from pathlib import Path

# --- 配置参数 ---
# CSV文件的路径，请根据你的实际位置修改
CSV_FILE = Path("runs/detect/fruit_detect_v1_embed/exp1/results.csv")
# 输出PNG文件的路径，会自动保存在CSV文件同目录下
OUTPUT_FILE = CSV_FILE.parent / "results.png"

# --- 读取和准备数据 ---
# 读取Ultralytics生成的CSV文件
df = pd.read_csv(CSV_FILE)

# 1. 将epoch列转换为整数索引
if 'epoch' in df.columns:
    df['epoch'] = df['epoch'].astype(int)
    x = df['epoch'].values
else:
    # 如果没有'epoch'列，则使用行号作为x轴
    x = range(len(df))

# 2. 清理列名：去除可能存在的首尾空格，并统一列名格式
df.columns = df.columns.str.strip()
cols_original = df.columns.tolist()

# 3. 定义需要绘制的指标及其在图表中的标题和坐标
# 参考Ultralytics官方绘图逻辑：https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/plotting.py
# 共9个子图，顺序为: 训练Loss (box, cls, dfl), 性能指标 (precision, recall, mAP50, mAP50-95), 验证Loss (box, cls, dfl)

metrics_to_plot = [
    # ---- 第一行: 训练损失 (1-3) ----
    {'col': 'train/box_loss', 'title': 'Train Box Loss', 'ylabel': 'Box Loss', 'loc': 1},
    {'col': 'train/cls_loss', 'title': 'Train Class Loss', 'ylabel': 'Class Loss', 'loc': 2},
    {'col': 'train/dfl_loss', 'title': 'Train DFL Loss', 'ylabel': 'DFL Loss', 'loc': 3},
    # ---- 第二行: 核心性能指标 (4-7) ----
    {'col': 'metrics/precision(B)', 'title': 'Precision', 'ylabel': 'Precision', 'loc': 4},
    {'col': 'metrics/recall(B)', 'title': 'Recall', 'ylabel': 'Recall', 'loc': 5},
    {'col': 'metrics/mAP50(B)', 'title': 'mAP@0.5', 'ylabel': 'mAP@0.5', 'loc': 6},
    {'col': 'metrics/mAP50-95(B)', 'title': 'mAP@0.5:0.95', 'ylabel': 'mAP@0.5:0.95', 'loc': 7},
    # ---- 第三行: 验证损失 (8-9) ----
    {'col': 'val/box_loss', 'title': 'Val Box Loss', 'ylabel': 'Box Loss', 'loc': 8},
    {'col': 'val/cls_loss', 'title': 'Val Class Loss', 'ylabel': 'Class Loss', 'loc': 9}
]

# --- 开始绘图 ---
# 初始化一个9子图的画布，官方分辨率通常为DPI=200
fig, axes = plt.subplots(3, 3, figsize=(15, 10), dpi=200)
axes = axes.flatten()  # 将3x3的axes扁平化为一维数组，方便按索引访问

# 遍历每个子图进行绘制
for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    col_name = metric['col']
    
    # 检查CSV中是否存在该指标列
    if col_name in df.columns:
        y = df[col_name].values
        ax.plot(x, y, marker='.', linestyle='-', color='b', linewidth=1.5, markersize=4)
        
        # 为验证损失或性能指标设置不同的y轴范围，使图表更美观
        if 'Loss' in metric['title']:
            # 损失曲线通常在0附近，可以设置一个合理的上限
            ax.set_ylim(bottom=0)
        else:
            # 性能指标 (Precision, Recall, mAP) 范围在0到1之间
            ax.set_ylim(0, 1)
        
        # 添加网格线，提升图表可读性
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        
    else:
        # 如果CSV中没有对应的列，在图表中央显示提示信息
        ax.text(0.5, 0.5, f'Column "{col_name}" not found', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_ylim(0, 1)

    # 设置子图的标题、x轴和y轴标签
    ax.set_title(metric['title'], fontsize=12)
    ax.set_ylabel(metric['ylabel'], fontsize=10)
    # 只在最下方的子图上显示x轴标签
    if i >= 6:
        ax.set_xlabel('Epoch', fontsize=10)

# 调整子图之间的间距，防止标签重叠
plt.tight_layout()

# --- 保存图片 ---
plt.savefig(OUTPUT_FILE, dpi=200, bbox_inches='tight')
print(f"成功生成 results.png 文件: {OUTPUT_FILE}")

# 如果你想在脚本运行时显示图表，可以取消下面一行的注释
# plt.show()