import cv2
import os
from ultralytics import YOLO
import time
import numpy as np

# --- Cấu hình ---
VIDEO_PATH = "video1.mp4"
MODEL_PATH = "yolov8n.pt"
BASE_DIR = "dataset"
CLASS_ID_TRAFFIC_LIGHT = 9
CONFIDENCE_THRESHOLD = 0.3
SAVE_INTERVAL = 0.5  # giây

INIT_SAMPLE_COUNT = 10
TOLERANCE = 7  # sai số pixel

# Tạo thư mục lưu
for color in ["green", "red", "yellow"]:
    os.makedirs(os.path.join(BASE_DIR, color), exist_ok=True)

# Khởi tạo model và video
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Không thể mở video")
    exit()

counter = {"red": 0, "yellow": 0, "green": 0}
last_save_time = time.time()

# Dùng để lấy mẫu ban đầu
init_sizes = []
avg_width = None
avg_height = None

def detect_traffic_light_color(crop):
    """Phát hiện màu chủ đạo (red/yellow/green) dựa trên contour lớn nhất."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    masks = {
        "red": cv2.inRange(hsv, (0,70,50), (10,255,255)) +
               cv2.inRange(hsv, (160,70,50), (180,255,255)),
        "yellow": cv2.inRange(hsv, (20,100,100), (30,255,255)),
        # Mở rộng ngưỡng xanh để bắt cả bóng sáng/tối
        "green": cv2.inRange(hsv, (35,40,40), (90,255,255))
    }

    best_color = None
    max_area = 0
    for c, mask in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        # Lấy diện tích contour lớn nhất
        area = max(cv2.contourArea(cnt) for cnt in contours)
        if area > max_area:
            max_area = area
            best_color = c

    # Chỉ chấp nhận nếu diện tích đủ lớn
    return best_color if max_area > 200 else None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    now = time.time()
    results = model(frame)

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id != CLASS_ID_TRAFFIC_LIGHT or conf < CONFIDENCE_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            crop = frame[y1:y2, x1:x2]
            h, w = crop.shape[:2]

            # Bỏ qua nếu chiều dài không lớn hơn 2 lần chiều rộng
            if h <= 2.5 * w:
                continue

            color = detect_traffic_light_color(crop)
            if color is None:
                continue

            # 1) Thu thập mẫu kích thước
            if len(init_sizes) < INIT_SAMPLE_COUNT:
                init_sizes.append((w, h))
                print(f"[Mẫu] {len(init_sizes)}/{INIT_SAMPLE_COUNT}: size=({w},{h})")
                if len(init_sizes) == INIT_SAMPLE_COUNT:
                    avg_width = int(np.mean([s[0] for s in init_sizes]))
                    avg_height = int(np.mean([s[1] for s in init_sizes]))
                    print(f"[INFO] Kích thước trung bình: {avg_width}x{avg_height}")
                continue  # tiếp tục thu thập, chưa lưu

            # 2) So sánh và lưu
            if now - last_save_time >= SAVE_INTERVAL:
                dw = abs(w - avg_width)
                dh = abs(h - avg_height)
                ok = (dw <= TOLERANCE and dh <= TOLERANCE)
                print(f"[Kiểm tra] size=({w},{h}), Δw={dw}, Δh={dh}, OK={ok}")
                if ok:
                    path = os.path.join(BASE_DIR, color, f"{color}_{counter[color]}.jpg")
                    cv2.imwrite(path, crop)
                    counter[color] += 1
                    last_save_time = now

            # Vẽ bounding box để hiển thị
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, color, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Kết quả lưu ảnh:")
for c, cnt in counter.items():
    print(f" - {c}: {cnt}")
