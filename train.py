from ultralytics import YOLO
import torch

if __name__ == '__main__':
    
    # 1. 加载你的自定义模型结构（.yaml 文件）
    model = YOLO("my_yolo11_embed_proto.yaml")  # 你的 yaml 文件，包含 Detect 的 embed_dim 参数

    # # 2. 加载官方预训练权重，允许不匹配的层跳过
    # model.load("runs/detect/embed_proto/exp1-2/weights/last.pt")

    # 2. 加载预训练权重到 CPU
    ckpt = torch.load("yolo11n.pt", map_location="cpu", weights_only=False)
    # ckpt = torch.load("runs/detect/embed_proto/exp1/weights/last.pt", map_location="cpu", weights_only=False)  

    # 3. 提取模型权重（Ultralytics checkpoint 的 "model" 键保存了状态字典）
    ckpt_model = ckpt["model"]                                   # 可能是整个模型对象
    if hasattr(ckpt_model, "state_dict"):
        ckpt_state_dict = ckpt_model.state_dict()                # 提取 state_dict
    else:
        ckpt_state_dict = ckpt_model                             # 如果已经是 dict

    # ----------------- [新增：动态过滤形状不匹配的层] -----------------
    model_state_dict = model.model.state_dict()
    filtered_state_dict = {}
    
    for k, v in ckpt_state_dict.items():
        if k in model_state_dict:
            # 只有当名字相同且形状完全一致时，才保留该权重
            if v.shape == model_state_dict[k].shape:
                filtered_state_dict[k] = v
            else:
                print(f"⚠️ 形状不匹配，已跳过加载层: {k} | 权重形状: {list(v.shape)} -> 模型形状: {list(model_state_dict[k].shape)}")
        else:
            filtered_state_dict[k] = v
    # ---------------------------------------------------------------

    # 4. 加载到你的模型，允许形状不匹配（新增的 cv4 层会随机初始化）
    missing, unexpected = model.model.load_state_dict(filtered_state_dict, strict=False)

    # 5. 打印信息（可选）
    print(f"缺失的键（新层，随机初始化）: {missing}")
    print(f"多余的键（官方有但你没用到的）: {unexpected}")

    # results = model.train(
    #     data="fruit_detect.yaml",
    #     epochs=170,        # 总训练轮次
    #     imgsz=640,         # 输入图像尺寸
    #     resume=True,
    #     batch=24,        # 你原来 batch=24
    #     device=0,            # GPU设备号
    #     workers=4,       # 你原来 workers=4
        
    #     # 数据增强参数 (示例：略微调整HSV增强)
    #     hsv_h=0.02,
    #     hsv_s=0.6,
    #     hsv_v=0.4,
    #     fliplr=0.5,
    # )                 # resume=True 会恢复 epoch、优化器等
    # Train the model
    results = model.train(
        data="fruit_detect.yaml",
        epochs=170,          # 总训练轮次
        patience=30,         # 早停，连续30轮无提升则停止
        imgsz=640,           # 输入图像尺寸
        batch=24,            # 手动设置批次大小
        device=0,            # GPU设备号
        workers=4,           # 数据加载进程数
        project='embed_proto-2',
        name='exp1',
        # 数据增强参数 (示例：略微调整HSV增强)
        hsv_h=0.02,
        hsv_s=0.6,
        hsv_v=0.4,
        fliplr=0.5,
    )
