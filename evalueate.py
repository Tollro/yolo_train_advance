"""
YOLO11 模型性能对比工具 (硬编码配置版 · 兼容自定义 head)
- 自动补充 nc_out, num_prototypes, embed_dim 等缺失属性
- 支持标准模型与自定义 yaml 模型混合对比
"""

import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO
import matplotlib.pyplot as plt
import csv
import warnings
warnings.filterwarnings("ignore")

# ================== 硬编码配置区 ==================
MODEL_PATHS = [
    "best.pt",
    "runs/detect/embed/exp1/weights/best.pt",
    "runs/detect/embed_DCNv2/exp1/weights/best.pt",
    "runs/detect/embed_proto-2/exp1/weights/best.pt"
]

MODEL_CONFIGS = [
    None,                           # 标准模型，直接用 pt 加载
    "my_yolo11_embed.yaml",
    "my_yolo11_embed_DCN.yaml",
    "my_yolo11_embed_proto.yaml"
]

DATA_YAML = "fruit_detect.yaml"
IMGSZ = 640
DEVICE = '0'
BATCH = 24
FPS_ITERS = 500
USE_HALF = True
SAVE_DIR = "./comparison_results"
# =================================================


def ensure_compatibility(model):
    """
    修复因修改 head 导致的旧模型缺失属性问题。
    若 Detect 层缺少 nc_out, num_prototypes, embed_dim 等属性，则自动补充为默认值。
    """
    try:
        head = model.model.model[-1]  # 最后一层（通常是 Detect）
        # 需要补充的属性和默认值
        defaults = {
            'nc_out': head.nc,         # 原版 nc_out = nc
            'num_prototypes': 1,       # 标准模型没有多原型，设为 1
            'embed_dim': 64,           # 嵌入维度默认 64
        }
        for attr, default_val in defaults.items():
            if not hasattr(head, attr):
                setattr(head, attr, default_val)
                print(f"  ℹ️ 已为模型添加兼容属性 {attr} = {default_val}")
    except Exception as e:
        print(f"  ⚠️ 兼容性处理失败: {e}")


def load_model_with_config(weights_path, config_path=None):
    """
    统一模型加载入口。
    - 若 config_path 有效：先用 yaml 构建自定义结构，再以非严格模式加载权重。
    - 否则直接使用 YOLO(weights_path) 加载完整 checkpoint。
    """
    if config_path and str(config_path).strip():
        print(f"  → 使用自定义配置构建模型: {config_path}")
        model = YOLO(config_path)

        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)
        if isinstance(ckpt, dict):
            if 'model' in ckpt:
                state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
            elif 'ema' in ckpt and hasattr(ckpt['ema'], 'state_dict'):
                state_dict = ckpt['ema'].state_dict()
            else:
                state_dict = ckpt
        elif hasattr(ckpt, 'state_dict'):
            state_dict = ckpt.state_dict()
        else:
            raise TypeError("无法从 checkpoint 提取 state_dict，请检查权重格式。")

        missing_keys, unexpected_keys = model.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"  ⚠️ 缺失的键（新增层，将随机初始化）: {missing_keys}")
        if unexpected_keys:
            print(f"  ⚠️ 多余的键（旧权重中存在但新模型不需要）: {unexpected_keys}")

        ensure_compatibility(model)
        return model
    else:
        print(f"  → 直接加载完整 checkpoint")
        model = YOLO(weights_path)
        ensure_compatibility(model)
        return model


def get_model_label(model_path, config_path=None):
    """返回模型标签，用于图表/CSV区分模型。"""
    if config_path and str(config_path).strip():
        return Path(config_path).stem
    path = Path(model_path)
    if path.stem.lower() == 'best' and path.parent.name:
        return path.parent.name
    return path.stem


def validate_model(model_path, config_path, data_yaml, imgsz, device, batch):
    print(f"\n{'='*50}")
    print(f"正在验证模型: {model_path}")
    model = load_model_with_config(model_path, config_path)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=batch,
        device=device,
        verbose=False
    )
    metrics = results.box
    return {
        'model': get_model_label(model_path, config_path),
        'precision': round(float(metrics.mp), 4),
        'recall': round(float(metrics.mr), 4),
        'map50': round(float(metrics.map50), 4)
    }


def measure_fps(model_path, config_path, imgsz, device, half, iters=500):
    """测量模型推理 FPS，若 FP16 失败则自动回退到 FP32"""
    print(f"正在测量 FPS: {model_path} ...")
    model = load_model_with_config(model_path, config_path)
    try:
        model.fuse()
    except Exception:
        pass
    model.model.eval()

    # Normalize device to torch.device
    if device == 'cpu':
        dev = torch.device('cpu')
    else:
        # device may be an int index or string
        try:
            dev_idx = int(device)
            dev = torch.device(f'cuda:{dev_idx}')
        except Exception:
            dev = torch.device('cuda')

    # Move model to target device first
    try:
        model.model.to(dev)
    except Exception:
        pass

    use_half = bool(half) and dev.type == 'cuda'

    # Set dtype and move model weights accordingly
    dtype = torch.half if use_half else torch.float
    try:
        model.model.to(dtype=dtype)
    except Exception:
        # Some modules may not support direct dtype conversion; try half()/float()
        try:
            if use_half:
                model.model.half()
            else:
                model.model.float()
        except Exception:
            pass

    # Prepare dummy input on same device and dtype
    dummy_input = torch.randn(1, 3, imgsz, imgsz, device=dev, dtype=dtype)

    # Test forward and gracefully fallback from FP16 to FP32 if needed
    try:
        _ = model.model(dummy_input)
    except RuntimeError as e:
        if use_half:
            print(f"  ⚠️ FP16 推理失败，回退到 FP32: {e}")
            use_half = False
            dtype = torch.float
            try:
                model.model.to(dtype=dtype)
            except Exception:
                try:
                    model.model.float()
                except Exception:
                    pass
            dummy_input = torch.randn(1, 3, imgsz, imgsz, device=dev, dtype=dtype)
            # retry
            _ = model.model(dummy_input)
        else:
            raise

    # Warmup
    for _ in range(100):
        _ = model.model(dummy_input)

    # Timed runs
    if dev.type == 'cpu':
        start = time.perf_counter()
        for _ in range(iters):
            _ = model.model(dummy_input)
        end = time.perf_counter()
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
    labels = [m['model'] for m in metrics_list]
    precision = [m['precision'] for m in metrics_list]
    recall = [m['recall'] for m in metrics_list]
    map50 = [m['map50'] for m in metrics_list]
    fps = [m['fps'] for m in metrics_list]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    ax[0, 0].bar(x, precision, width, color=colors[0])
    ax[0, 0].set_title('Precision')
    ax[0, 0].set_xticks(x)
    ax[0, 0].set_xticklabels(labels, rotation=15)
    for i, v in enumerate(precision):
        ax[0, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')

    ax[0, 1].bar(x, recall, width, color=colors[1])
    ax[0, 1].set_title('Recall')
    ax[0, 1].set_xticks(x)
    ax[0, 1].set_xticklabels(labels, rotation=15)
    for i, v in enumerate(recall):
        ax[0, 1].text(i, v + 0.01, f'{v:.4f}', ha='center')

    ax[1, 0].bar(x, map50, width, color=colors[2])
    ax[1, 0].set_title('mAP@0.5')
    ax[1, 0].set_xticks(x)
    ax[1, 0].set_xticklabels(labels, rotation=15)
    for i, v in enumerate(map50):
        ax[1, 0].text(i, v + 0.01, f'{v:.4f}', ha='center')

    ax[1, 1].bar(x, fps, width, color=colors[3])
    ax[1, 1].set_title('FPS')
    ax[1, 1].set_xticks(x)
    ax[1, 1].set_xticklabels(labels, rotation=15)
    for i, v in enumerate(fps):
        ax[1, 1].text(i, v + 1, f'{v:.2f}', ha='center')

    plt.suptitle('YOLO11 Model Performance Comparison', fontsize=16)
    plt.tight_layout()
    save_path = Path(save_dir) / 'comparison_plot.png'
    plt.savefig(save_path, dpi=200)
    print(f"对比图表已保存至 {save_path}")
    plt.close()


def main():
    assert len(MODEL_PATHS) == len(MODEL_CONFIGS), "MODEL_PATHS 与 MODEL_CONFIGS 长度必须相等！"

    device = DEVICE if DEVICE == 'cpu' else int(DEVICE)
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"共对比 {len(MODEL_PATHS)} 个模型:")
    for i, (m, c) in enumerate(zip(MODEL_PATHS, MODEL_CONFIGS), 1):
        cfg_info = f" (配置: {c})" if c else ""
        print(f"  {i}. {m}{cfg_info}")

    metrics_list = []
    for model_path, config_path in zip(MODEL_PATHS, MODEL_CONFIGS):
        metrics = validate_model(model_path, config_path, DATA_YAML, IMGSZ, device, BATCH)
        fps = measure_fps(model_path, config_path, IMGSZ, device, USE_HALF, FPS_ITERS)
        metrics['fps'] = fps
        metrics_list.append(metrics)

    csv_path = save_dir / 'metrics.csv'
    headers = ["Model", "Precision", "Recall", "mAP@0.5", "FPS"]
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for m in metrics_list:
            writer.writerow({'Model': m['model'], 'Precision': m['precision'],
                             'Recall': m['recall'], 'mAP@0.5': m['map50'], 'FPS': m['fps']})
    print(f"指标 CSV 已保存至 {csv_path}")

    plot_comparison(metrics_list, save_dir)
    print("\n全部对比完成！")


if __name__ == '__main__':
    main()