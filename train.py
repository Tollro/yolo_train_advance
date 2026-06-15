from ultralytics import YOLO
import torch

if __name__ == '__main__':
    
    # 1. 加载你的自定义模型结构（.yaml 文件）
    model = YOLO("my_yolo11.yaml")  # 你的 yaml 文件，包含 Detect 的 embed_dim 参数

    # # 2. 加载官方预训练权重，允许不匹配的层跳过
    # model.load("yolo11n.pt", strict=False)

    # 2. 加载官方预训练权重到 CPU
    ckpt = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)

    # 3. 提取模型权重（Ultralytics checkpoint 的 "model" 键保存了状态字典）
    ckpt_model = ckpt["model"]                                   # 可能是整个模型对象
    if hasattr(ckpt_model, "state_dict"):
        ckpt_state_dict = ckpt_model.state_dict()                # 提取 state_dict
    else:
        ckpt_state_dict = ckpt_model                             # 如果已经是 dict

    # 4. 加载到你的模型，允许形状不匹配（新增的 cv4 层会随机初始化）
    missing, unexpected = model.model.load_state_dict(ckpt_state_dict, strict=False)

    # 5. 打印信息（可选）
    print(f"缺失的键（新层，随机初始化）: {missing}")
    print(f"多余的键（官方有但你没用到的）: {unexpected}")

    # Train the model
    results = model.train(
        data="fruit_detect.yaml",
        epochs=170,          # 总训练轮次
        # patience=80,         # 早停，连续50轮无提升则停止
        imgsz=640,           # 输入图像尺寸
        batch=24,            # 手动设置批次大小
        device=0,            # GPU设备号
        workers=4,           # 数据加载进程数
        project='fruit_detect_v1_embed',
        name='exp1',
        # 数据增强参数 (示例：略微调整HSV增强)
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        # mosaic=1.0,        # 默认为1.0，可注释掉
        # close_mosaic=10,   # 最后10轮关闭马赛克增强，防止波动'
    )
