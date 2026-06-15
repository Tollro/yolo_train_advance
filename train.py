from ultralytics import YOLO

if __name__ == '__main__':
    
    # 1. 加载你的自定义模型结构（.yaml 文件）
    model = YOLO("my_yolo11.yaml")  # 你的 yaml 文件，包含 Detect 的 embed_dim 参数

    # 2. 加载官方预训练权重，允许不匹配的层跳过
    model.load("yolo11n.pt", strict=False)

    pretrained_params = 0
    new_params = 0
    for name, p in model.model.named_parameters():
        if p.requires_grad:
            if hasattr(p, '_loaded_from_ckpt'):
                pretrained_params += p.numel()
            else:
                new_params += p.numel()
    print(f"预训练参数: {pretrained_params}, 新增参数: {new_params}")

    # Train the model
    results = model.train(
        data="fruit_detect.yaml",
        epochs=170,          # 总训练轮次
        # patience=80,         # 早停，连续50轮无提升则停止
        imgsz=640,           # 输入图像尺寸
        batch=24,            # 手动设置批次大小
        device=0,            # GPU设备号
        workers=4,           # 数据加载进程数
        project='fruit_detect_myloss',
        name='exp1',
        # 数据增强参数 (示例：略微调整HSV增强)
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
        # mosaic=1.0,        # 默认为1.0，可注释掉
        # close_mosaic=10,   # 最后10轮关闭马赛克增强，防止波动'
    )
