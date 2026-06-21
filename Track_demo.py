import cv2
import time
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 初始化模型并加载权重
    model = YOLO("my_yolo11.yaml")  
    model.load("runs/detect/embed_proto/exp1-2/weights/last.pt")  

    # 2. 9 个水果类别名称
    class_names = ["Apple", "Watermelon", "Orange", "Banana", "Strawberry", "Kiwifruit", "Pineapple", "Durian", "Pitaya"]

    # 3. 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ 错误：无法打开摄像头！")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1240)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0
    print("▶️ 实时追踪已启动！按 'Q' 键安全退出...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 4. 模型追踪 (改用 .track 代替 .predict)
        # persist=True：告诉模型这是连续视频流，需要记住上一帧的 ID
        # tracker="bytetrack.yaml"：使用 ByteTrack 算法，比默认的 BoT-SORT 更轻量且适合实时抓拍
        results = model.track(source=frame, imgsz=640, conf=0.3, iou=0.45, 
                              persist=True, tracker="bytetrack.yaml", stream=True, verbose=False)

        # 5. 解析追踪结果
        for result in results:
            boxes = result.boxes
            
            # 安全检查：只有当画面中有物体，且成功分配了追踪 ID 时才处理
            if boxes.id is not None:
                # 批量提取数据并转为常规 Python 列表
                track_ids = boxes.id.int().cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.float().cpu().tolist()
                xyxys = boxes.xyxy.int().cpu().tolist()

                for track_id, cls_id, conf, xyxy in zip(track_ids, cls_ids, confs, xyxys):
                    # 越界保护
                    if cls_id >= len(class_names):
                        continue

                    # 解析边界框坐标
                    x1, y1, x2, y2 = xyxy

                    # ✨【核心计算】计算水果的中心点坐标
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # [此处可以插入你的接口逻辑]
                    # 比如：将 (class_names[cls_id], track_id, cx, cy) 通过串口或 TCP 发送给机械臂
                    # print(f"输出接口 -> ID:{track_id} | 类别:{class_names[cls_id]} | 中心坐标:({cx}, {cy})")

                    # 6. 动态绘制可视化界面
                    # 标签格式：ID:1 Apple (0.85)
                    label = f"ID:{track_id} {class_names[cls_id]} {conf:.2f}"
                    
                    # 画外围的绿色边界框
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # 画顶部的标签背景板
                    cv2.putText(frame, label, (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # ✨ 画中心点及其坐标
                    # 画一个醒目的红色实心圆点代表中心
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                    # 在中心点旁边标注坐标文本
                    cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # 7. 计算并显示 FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        # 8. 展示画面
        cv2.imshow("YOLO11 Fruit Tracking & Center Coordinates", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ 正常退出，测试结束。")