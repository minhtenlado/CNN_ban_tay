import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np

DATA_DIR = r'D:\tri_tue_nhan_tao\thay_Lanh\nhan_dang_ban_tay\data2'
MODEL_PATH = 'finger_count_model.h5'

# Kích thước đưa vào mạng
IMG_HEIGHT = 128
IMG_WIDTH = 128
NUM_CLASSES = 6  # Bao gồm các lớp 0, 1, 2, 3, 4, 5
BATCH_SIZE = 60
EPOCHS = 1000

def load_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    for label in range(NUM_CLASSES):  # Từ 0 đến 5
        label_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(label_dir):
            print(f"[WARN] Thư mục không tồn tại: {label_dir}")
            continue
        for fn in os.listdir(label_dir):
            full_path = os.path.join(label_dir, fn)
            if not os.path.isfile(full_path):
                continue
            if not fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            image_paths.append(full_path)
            labels.append(label)
    return image_paths, labels

print("1) Load danh sách ảnh và nhãn...")
paths, labs = load_image_paths_and_labels(DATA_DIR)
print(f"   → Đã tìm thấy {len(paths)} ảnh hợp lệ.")

if len(paths) == 0:
    print("[LỖI] Không tìm thấy ảnh hợp lệ nào trong thư mục.")
    exit()

BATCH_SIZE = min(BATCH_SIZE, len(paths))
if BATCH_SIZE == 0:
    print("[LỖI] Batch size không thể là 0.")
    exit()
print(f"   → Cập nhật Batch Size: {BATCH_SIZE}")

def preprocess_image(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = img / 255.0
    label_onehot = tf.one_hot(label, depth=NUM_CLASSES)
    return img, label_onehot

augmentation_layers = tf.keras.Sequential([
    layers.RandomRotation(factor=0.15),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    layers.RandomZoom(height_factor=0.1, width_factor=0.1),
    layers.RandomFlip("horizontal"),
])

def apply_augmentation(image, label):
    augmented_image = augmentation_layers(image, training=True)
    return augmented_image, label

print("2) Tạo tf.data.Dataset...")
dataset = tf.data.Dataset.from_tensor_slices((paths, labs))
dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.AUTOTUNE)

print(f"   → Dataset sẵn sàng: batch size = {BATCH_SIZE}")

print("3) Xây dựng mô hình CNN...")
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.BatchNormalization(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(2),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

print("4) Compile và bắt đầu huấn luyện...")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='loss', patience=50, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=20, min_lr=1e-5)

print(f"   → Huấn luyện trên {len(paths)} ảnh với batch size {BATCH_SIZE}...")
history = model.fit(
    dataset,
    epochs=EPOCHS,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

print(f"5) Lưu mô hình xuống: {MODEL_PATH}")
model.save(MODEL_PATH)
print("Hoàn tất.")