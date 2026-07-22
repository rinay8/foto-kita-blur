import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

MODEL_PATH = "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Mengunduh model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)

def is_peace_sign(hand_landmarks):
    # Index tip=8, pip=6 ; Middle tip=12, pip=10
    # Ring tip=16, pip=14 ; Pinky tip=20, pip=18
    index_up = hand_landmarks[8].y < hand_landmarks[6].y
    middle_up = hand_landmarks[12].y < hand_landmarks[10].y
    ring_down = hand_landmarks[16].y > hand_landmarks[14].y
    pinky_down = hand_landmarks[20].y > hand_landmarks[18].y
    return index_up and middle_up and ring_down and pinky_down

cap = cv2.VideoCapture(0)
timestamp = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect_for_video(mp_image, timestamp)
    timestamp += 1

    peace_detected = False
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            if is_peace_sign(hand_landmarks):
                peace_detected = True
                break

    if peace_detected:
        frame = cv2.GaussianBlur(frame, (99, 99), 0)
        cv2.putText(frame, "Blur!", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (203, 192, 255), 2)

    cv2.imshow("Hand Blur", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()