import cv2
import os
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# --- Cấu hình --- #
VIDEO_PATH = "video1.mp4"
MODEL_PATH = "yolov8n.pt"  # Đường dẫn đến mô hình YOLOv8
TF_MODEL_PATH = "traffic_light_model.h5"  # Đường dẫn đến mô hình TensorFlow

# Khởi tạo YOLOv8 và mô hình TensorFlow
yolo_model = YOLO(MODEL_PATH)
tf_model = tf.keras.models.load_model(TF_MODEL_PATH)

# Đọc video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Không thể mở video")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Sử dụng YOLOv8 để phát hiện các đèn giao thông
    results = yolo_model(frame)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Chỉ xử lý các đối tượng là đèn giao thông (CLASS_ID_TRAFFIC_LIGHT = 9)
            if cls_id != 9 or conf < 0.3:  # Thay đổi ngưỡng độ tin cậy nếu cần
                continue

            # Lấy tọa độ của bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]  # Cắt vùng chứa đèn giao thông

            # Tiền xử lý để dự đoán bằng TensorFlow
            cropped_img = cv2.resize(crop, (150, 150))  # Điều chỉnh kích thước cho phù hợp
            cropped_img = np.expand_dims(cropped_img, axis=0)  # Thêm batch dimension
            cropped_img = cropped_img / 255.0  # Chuẩn hóa ảnh

            # Dự đoán màu của đèn giao thông
            predictions = tf_model.predict(cropped_img)
            predicted_class = np.argmax(predictions, axis=1)

            # Gán tên màu của đèn giao thông
            class_names = ['green', 'red', 'yellow']
            predicted_color = class_names[predicted_class[0]]

            # Vẽ bounding box và màu đèn giao thông lên video
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, predicted_color, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Hiển thị video với các dự đoán
    cv2.imshow("Traffic Light Detection", frame)

    # Nếu nhấn phím 'q', thoát khỏi vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
