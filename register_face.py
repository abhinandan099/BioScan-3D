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
    engine.setProperty('rate', 150)
    while True:
        text = voice_queue.get()
        if text is None: break
        engine.say(text)
        engine.runAndWait()
        voice_queue.task_done()

threading.Thread(target=voice_worker, daemon=True).start()

def speak(text): voice_queue.put(text)

EMBEDDINGS_DIR = "embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
MODEL_PATH = "face_landmarker.task"

print("╔═════════════════════════════════════════════╗")
print("║     ADVANCED 3D STUDENT ENROLLMENT          ║")
print("╚═════════════════════════════════════════════╝")
reg_no = input("🔹 Reg No: ").strip()
name = input("🔹 Name: ").upper().strip()
course = input("🔹 Course: ").upper().strip()
year = input("🔹 Batch/Year: ").strip()
blood = input("🔹 Blood Group: ").upper().strip()

landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO))

cap = cv2.VideoCapture(0)
all_embs, samples = [], 0
speak(f"Initializing 3D scan for {name}. Please look at the camera.")

try:
    while samples < 100:
        ret, frame = cap.read()
        ts = int(time.time() * 1000)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_image, ts)

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            samples += 1

            coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
            centered = coords - np.mean(coords, axis=0)
            all_embs.append((centered / np.max(np.linalg.norm(centered, axis=1))).flatten())

            cv2.rectangle(frame, (50, h - 40), (w - 50, h - 20), (40, 40, 40), -1)
            cv2.rectangle(frame, (50, h - 40), (50 + int((samples / 100) * (w - 100)), h - 20), (0, 255, 100), -1)
            cv2.putText(frame, f"3D MAPPING: {samples}%", (50, h - 50), 1, 1, (0, 255, 100), 1)

        cv2.imshow("ENROLLMENT", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
finally:
    cap.release()
    cv2.destroyAllWindows()

profile = {"reg": reg_no, "name": name, "course": course, "year": year, "blood": blood}
np.save(os.path.join(EMBEDDINGS_DIR, f"{reg_no}.npy"), np.mean(all_embs, axis=0))
with open(os.path.join(EMBEDDINGS_DIR, f"{reg_no}.json"), 'w') as f:
    json.dump(profile, f)

now = datetime.now()
print("\n" + "═" * 50)
print("        🧬 3D BIOMETRIC ENROLL SUCCESS 🧬")
print("═" * 50)
print(f"  👤 NAME      :  {name}\n  🆔 REG NO    :  {reg_no}\n  🎓 COURSE    :  {course}\n  🩸 BLOOD     :  {blood}")
print("═" * 50)
speak("Profile created successfully.")