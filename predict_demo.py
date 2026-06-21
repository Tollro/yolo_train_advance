import cv2
import time
import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 初始化模型并加载你训练出来的自定义权重文件
    # 确保 my_yolo11.yaml 路径正确
    model = YOLO("my_yolo11.yaml")  
    model.load("runs/detect/embed_proto/exp1-2/weights/last.pt")  

    # 2. 你的 9 个水果类别名称（严格按照 0~8 的顺序）
    class_names = ["Apple", "Watermelon", "Orange", "Banana", "Strawberry", "Kiwifruit", "Pineapple", "Durian", "Pitaya"]

    # 3. 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 错误：无法打开摄像头，请检查设备连接或权限！")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1240)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0
    print("▶️ 实时检测已启动！按键盘上的 'Q' 键可以安全退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 错误：无法接收相机画面，正在退出...")
            break

        # 4. 模型推理 (stream=True 开启流式推理，内存占用极低)
        results = model.predict(source=frame, imgsz=640, conf=0.4, iou=0.45, stream=True, verbose=False)

        # 5. 解析预测结果
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # ✨【修改点 1】直接获取真实的 9 分类 ID (0~8)
                real_cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)

                # 越界安全保护
                if real_cls_id >= len(class_names):
                    continue

                # ✨【修改点 2】直接使用真实英文名，不再拼接 _P0 或 _P1
                label = f"{class_names[real_cls_id]} {conf:.2f}"
                
                # 绘制绿色的物体框
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                
                # 绘制标签背景板
                cv2.putText(frame, label, (xyxy[0], xyxy[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 7. 实时计算并绘制 FPS 帧率
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        # 在图像左上角打印红色的 FPS
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 8. 窗口展示画面
        cv2.imshow("YOLO11 Multi-Prototype Real-Time Detection", frame)

        # 9. 退出监听：按 'q' 键或者 'Q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 彻底释放摄像头并销毁窗口
    cap.release()
    cv2.destroyAllWindows()
    print("✅ 正常退出，测试结束。")