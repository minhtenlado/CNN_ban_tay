import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

MODEL_PATH = 'finger_count_model.h5'
IMG_SIZE = (128, 128)

# Load mô hìnhimport cv2
import mediapipe as mp

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Khởi tạo camera
cap = cv2.VideoCapture(0)

def count_fingers(hand_landmarks):
    # Danh sách các điểm mốc cho các đầu ngón tay
    finger_tips = [8, 12, 16, 20]  # Điểm mốc cho các ngón trỏ, giữa, áp út, út
    thumb_tip = 4  # Điểm mốc cho ngón cái
    
    count = 0
    
    # Kiểm tra ngón cái
    if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
        count += 1
    
    # Kiểm tra các ngón khác
    for tip in finger_tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            count += 1
            
    return count

while True:
    success, image = cap.read()
    if not success:
        break
        
    # Chuyển đổi màu ảnh từ BGR sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Xử lý ảnh để phát hiện bàn tay
    results = hands.process(image_rgb)
    
    # Khởi tạo finger_count với giá trị mặc định
    finger_count = 0
    
    # Vẽ các điểm mốc trên bàn tay và đếm số ngón tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Vẽ các điểm mốc
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Đếm số ngón tay
            finger_count = count_fingers(hand_landmarks)
            
            # Hiển thị số ngón tay
            cv2.putText(image, str(finger_count), (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

            print(finger_count)
    
    # Hiển thị hình ảnh
    cv2.imshow('Hand Tracking', image)
    
    # Kiểm tra phím nhấn
    key = cv2.waitKey(1) & 0xFF
    
    # Dừng nếu nhấn 'q' hoặc phát hiện 5 ngón tay
    if key == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
print(f"1) Loading model từ {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("   → Model loaded.")

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Hàm tiền xử lý vùng tay
def preprocess_hand_roi(frame, bbox):
    x_min, y_min, x_max, y_max = bbox
    roi = frame[y_min:y_max, x_min:x_max]
    roi = cv2.resize(roi, IMG_SIZE)
    roi = roi.astype(np.float32) / 255.0
    roi = np.expand_dims(roi, axis=0)
    return roi

# Bắt đầu webcam
print("2) Bắt đầu webcam. Nhấn ESC để thoát.")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Không thể mở camera!")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x_min = int(min([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_min = int(min([landmark.y for landmark in hand_landmarks.landmark]) * h)
            x_max = int(max([landmark.x for landmark in hand_landmarks.landmark]) * w)
            y_max = int(max([landmark.y for landmark in hand_landmarks.landmark]) * h)
            
            margin = 20
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(w, x_max + margin)
            y_max = min(h, y_max + margin)
            
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            
            bbox = (x_min, y_min, x_max, y_max)
            roi_input = preprocess_hand_roi(frame, bbox)
            
            preds = model.predict(roi_input)
            class_id = np.argmax(preds[0])
            prob = preds[0][class_id]
            finger_count = class_id
            
            text = f"Ngon: {finger_count} ({prob*100:.1f}%)"
            cv2.putText(frame, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "Khong phat hien ban tay", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow('Finger Count', frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
hands.close()
print("Đã đóng chương trình.")