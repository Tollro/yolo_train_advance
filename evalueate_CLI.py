"""
YOLO11 模型性能对比工具 (通用版)
对比指标: Precision, Recall, mAP@0.5, FPS
支持任意数量 .pt 模型，基于同一验证集进行评估。
"""

import argparse
import time
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tabulate import tabulate
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO11 多模型性能对比工具")
    parser.add_argument('--models', nargs='+', required=True,
                        help='待对比的模型路径列表，如 model1.pt model2.pt ...')
    parser.add_argument('--data', type=str, required=True,
                        help='数据集配置文件 data.yaml 路径')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='推理/验证图像尺寸 (默认 640)')
    parser.add_argument('--device', type=str, default='0',
                        help='计算设备，如 0(单GPU) 或 cpu')
    parser.add_argument('--batch', type=int, default=16,
                        help='验证批次大小')
    parser.add_argument('--fps-iters', type=int, default=500,
                        help='FPS 测试迭代次数 (默认 500)')
    parser.add_argument('--half', action='store_true',
                        help='启用 FP16 半精度推理 (GPU建议启用)')
    parser.add_argument('--save-dir', type=str, default='./comparison_results',
                        help='结果保存目录')
    parser.add_argument('--plot', action='store_true', default=True,
                        help='生成对比图表')
    return parser.parse_args()


def validate_model(model_path, data_yaml, imgsz, device, batch):
    """验证单个模型，返回精确率、召回率、mAP@0.5"""
    print(f"\n{'='*50}")
    print(f"正在验证模型: {model_path}")
    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False
    )
    # 提取分类指标 (Detection 场景用 box)
    metrics = results.box
    precision = metrics.mp   # 平均精确率
    recall = metrics.mr      # 平均召回率
    map50 = metrics.map50    # mAP@0.5
    return {
        'model': Path(model_path).stem,
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'map50': round(float(map50), 4)
    }


def measure_fps(model_path, imgsz, device, half, iters=500):
    """测量模型推理 FPS (包含预处理与后处理，模拟真实推理)"""
    print(f"正在测量 FPS: {model_path} ...")
    model = YOLO(model_path)
    # 创建一个随机输入张量模拟图片
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    if half and device != 'cpu':
        model.model.half()
        dummy_input = dummy_input.half()

    # 预热 GPU
    for _ in range(100):
        _ = model.model(dummy_input)

    # 计时
    if device == 'cpu':
        # CPU 计时使用 time.perf_counter
        start = time.perf_counter()
        for _ in range(iters):
            _ = model.model(dummy_input)
        end = time.perf_counter()
        total_time = end - start
    else:
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            _ = model.model(dummy_input)
        torch.cuda.synchronize()
        end = time.perf_counter()
        total_time = end - start

    fps = iters / total_time
    return round(fps, 2)


def plot_comparison(metrics_list, save_dir):
    """绘制柱状图对比指标"""
    labels = [m['model'] for m in metrics_list]
    precision_vals = [m['precision'] for m in metrics_list]
    recall_vals = [m['recall'] for m in metrics_list]
    map50_vals = [m['map50'] for m in metrics_list]
    fps_vals = [m['fps'] for m in metrics_list]

    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    # Precision
    ax[0, 0].bar(x, precision_vals, width, color=colors[0])
    ax[0, 0].set_title('Precision')
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(labels, rotation=15)
    for i, v in enumerate(precision_vals):
        ax[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')

    # Recall
    ax[0, 1].bar(x, recall_vals, width, color=colors[1])
    ax[0, 1].set_title('Recall')
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(labels, rotation=15)
    for i, v in enumerate(recall_vals):
        ax[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center')

    # mAP@0.5
    ax[1, 0].bar(x, map50_vals, width, color=colors[2])
    ax[1, 0].set_title('mAP@0.5')
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels(labels, rotation=15)
    for i, v in enumerate(map50_vals):
        ax[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')

    # FPS
    ax[1, 1].bar(x, fps_vals, width, color=colors[3])
    ax[1, 1].set_title('FPS')
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(labels, rotation=15)
    for i, v in enumerate(fps_vals):
        ax[1, 1].text(i, v + 1, f'{v:.2f}', ha='center')

    plt.suptitle('YOLO11 Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    save_path = Path(save_dir) / 'comparison_plot.png'
    plt.savefig(save_path, dpi=200)
    print(f"对比图表已保存至 {save_path}")
    plt.close()


def main():
    args = parse_args()

    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 设备字符串处理
    device = args.device if args.device == 'cpu' else int(args.device)

    print(f"将对比 {len(args.models)} 个模型:")
    for i, m in enumerate(args.models, 1):
        print(f"  {i}. {m}")

    metrics_list = []
    # 1. 验证精度指标
    for model_path in args.models:
        metrics = validate_model(model_path, args.data, args.imgsz, device, args.batch)
        # 2. 测量 FPS (在 GPU/CPU 上单独进行)
        fps = measure_fps(model_path, args.imgsz, device, args.half, args.fps_iters)
        metrics['fps'] = fps
        metrics_list.append(metrics)

    # 输出表格
    print("\n" + "="*70)
    print("模型性能对比结果汇总")
    headers = ["Model", "Precision", "Recall", "mAP@0.5", "FPS"]
    table_data = [[m['model'], m['precision'], m['recall'], m['map50'], m['fps']] 
                  for m in metrics_list]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # 保存CSV
    import csv
    csv_path = save_dir / 'metrics.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({'Model': m['model'], 'Precision': m['precision'],
                             'Recall': m['recall'], 'mAP@0.5': m['map50'], 'FPS': m['fps']})
    print(f"指标表格已保存至 {csv_path}")

    # 绘图
    if args.plot:
        plot_comparison(metrics_list, save_dir)

    print("全部对比完成！")


if __name__ == '__main__':
    main()