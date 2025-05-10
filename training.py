import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import os

# --- Cấu hình --- #
BASE_DIR = "dataset"  # Đường dẫn đến thư mục dataset đã phân loại
IMG_SIZE = (150, 150)  # Kích thước ảnh đầu vào
BATCH_SIZE = 32
EPOCHS = 10

# --- Tiền xử lý dữ liệu --- #
# Tạo ImageDataGenerator cho huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1./255,  # chuẩn hóa giá trị pixel về phạm vi [0, 1]
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Tạo generator cho tập huấn luyện
train_generator = train_datagen.flow_from_directory(
    BASE_DIR,
    target_size=IMG_SIZE,  # Thay đổi kích thước ảnh
    batch_size=BATCH_SIZE,
    class_mode='categorical')  # Vì là phân loại nhiều lớp (red, yellow, green)

# --- Xây dựng mô hình CNN --- #
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Giảm thiểu overfitting
    Dense(3, activation='softmax')  # 3 lớp cho 3 loại đèn giao thông (green, red, yellow)
])

# Tóm tắt mô hình
model.summary()

# --- Biên dịch mô hình --- #
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- Huấn luyện mô hình --- #
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=EPOCHS)  # Số epoch có thể điều chỉnh tùy theo yêu cầu

# --- Lưu mô hình --- #
model.save('traffic_light_model.h5')
