import cv2
import numpy as np
import os
import pyttsx3
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from datetime import datetime
import time
import threading
import queue
import json

voice_queue = queue.Queue()

def voice_worker():
    engine = pyttsx3.init()
    engine.setProperty('rate', 155)
    while True:
        text = voice_queue.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()
        voice_queue.task_done()

threading.Thread(target=voice_worker, daemon=True).start()

def speak(text):
    voice_queue.put(text)

EMBEDDINGS_DIR, MODEL_PATH = "embeddings", "face_landmarker.task"
THRESHOLD = 1.15
WINDOW_NAME = "3D IDENTITY SCANNER"

profiles = {}
face_db = {}

if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)

for f in os.listdir(EMBEDDINGS_DIR):
    if f.endswith(".json"):
        with open(os.path.join(EMBEDDINGS_DIR, f)) as p:
            data = json.load(p)
            profiles[data['reg']] = data
    if f.endswith(".npy"):
        face_db[f[:-4]] = np.load(os.path.join(EMBEDDINGS_DIR, f))

def get_normalized_embedding(face_landmarks):
    coords = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks])
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    max_dist = np.max(np.linalg.norm(centered, axis=1))
    return (centered / max_dist).flatten() if max_dist != 0 else np.zeros(coords.flatten().shape)

landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO))

cap = cv2.VideoCapture(0)
last_ts = -1

print("🚀 3D BIOMETRIC SCANNER ACTIVE...")
print("💡 AUTO-STOP ENABLED UPON RECOGNITION.")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        ts = int(time.time() * 1000)
        if ts <= last_ts: ts = last_ts + 1
        last_ts = ts
        h, w, _ = frame.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, ts)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]

            for lm in landmarks:
                z_depth = int(abs(lm.z) * 5000)
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (255 - min(z_depth, 255), 255, min(z_depth, 255)), -1)

            current_emb = get_normalized_embedding(landmarks)
            min_dist, user_id = 2.0, None

            for rid, saved in face_db.items():
                dist = np.linalg.norm(current_emb - saved)
                if dist < min_dist:
                    min_dist, user_id = dist, rid

            if user_id and min_dist < THRESHOLD:
                student = profiles.get(user_id)
                if student:
                    now = datetime.now()

                    print("\n" + "═" * 55)
                    print("        🔍 BIOMETRIC IDENTITY CONFIRMED")
                    print("═" * 55)
                    print(f"  👤 NAME      :  {student['name']}")
                    print(f"  🆔 REG NO    :  {student['reg']}")
                    print(f"  🎓 COURSE    :  {student['course']} ({student['year']})")
                    print(f"  🩸 BLOOD GRP :  {student['blood']}")
                    print(f"  ⏰ TIMESTAMP :  {now.strftime('%I:%M:%S %p')}")
                    print("═" * 55 + "\n")

                    speak(f"Access granted for {student['name']}.")
                    time.sleep(0.5)
                    break

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("👋 Scanner Terminated Successfully.")