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
import csv

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

EMBEDDINGS_DIR, MODEL_PATH = "embeddings", "face_landmarker.task"
LATE_TIME = "09:00:00"

profiles = {}
face_db = {}
for f in os.listdir(EMBEDDINGS_DIR):
    if f.endswith(".json"):
        with open(os.path.join(EMBEDDINGS_DIR, f)) as p:
            d = json.load(p)
            profiles[d['reg']] = d
    if f.endswith(".npy"):
        face_db[f[:-4]] = np.load(os.path.join(EMBEDDINGS_DIR, f))

landmarker = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO))

cap, shutdown_timer = cv2.VideoCapture(0), None

while cap.isOpened():
    ret, frame = cap.read()
    ts = int(time.time() * 1000)
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    result = landmarker.detect_for_video(mp_image, ts)

    if result.face_landmarks and shutdown_timer is None:
        landmarks = result.face_landmarks[0]
        coords = np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
        curr_emb = ((coords - np.mean(coords, axis=0)) / np.max(
            np.linalg.norm(coords - np.mean(coords, axis=0), axis=1))).flatten()

        min_dist, user_id = 1.15, None
        for rid, saved in face_db.items():
            dist = np.linalg.norm(curr_emb - saved)
            if dist < min_dist: min_dist, user_id = dist, rid

        if user_id and user_id in profiles:
            student = profiles[user_id]
            now = datetime.now()
            status = "PRESENT" if now.strftime("%H:%M:%S") < LATE_TIME else "LATE"

            print("\n" + "═" * 55)
            print(f"        ✅ ATTENDANCE LOGGED: {status}")
            print("═" * 55)
            print(f"  👤 NAME: {student['name']}")
            print(f"  🆔 REG : {student['reg']}")
            print(f"  🎓 CRS : {student['course']}")
            print(f"  ⏰ TIME: {now.strftime('%I:%M:%S %p')}")
            print("═" * 55)

            csv_file = f"attendance_{now.strftime('%Y-%m-%d')}.csv"
            file_exists = os.path.isfile(csv_file)

            with open(csv_file, mode='a', newline='') as f:
                writer = csv.writer(f, delimiter=',')

                if not file_exists:
                    writer.writerow(
                        ["Reg_No", "Name", "Course", "Batch", "Blood_Group", "Date", "Day", "Timestamp", "Status"])

                writer.writerow([
                    student['reg'],
                    student['name'],
                    student['course'],
                    student['year'],
                    student['blood'],
                    now.strftime('%d-%m-%Y'),
                    now.strftime('%A'),
                    now.strftime('%I:%M:%S %p'),
                    status
                ])

            speak(f"Welcome {student['name']}. Your attendance is marked.")
            shutdown_timer = time.time()

    cv2.imshow("3D ATTENDANCE", frame)
    if shutdown_timer and (time.time() - shutdown_timer > 3): break
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()