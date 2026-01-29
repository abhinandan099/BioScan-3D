# 🧬 BioScan-3D
### Voice-Guided 3D Biometric Face Recognition & Attendance System

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/) 
[![MediaPipe](https://img.shields.io/badge/Biometrics-MediaPipe-green.svg)](https://google.github.io/mediapipe/) 
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)](#)
[![Status](https://img.shields.io/badge/Status-Working-success.svg)](#)

> 📌 Tip: Add a GIF or screenshot here showing the 3D facial mesh detection.

---

## 🔍 Project Overview
**BioScan-3D** is a **3D face-based biometric system** for secure student recognition and automated attendance.  
It uses **MediaPipe’s 478-point 3D facial mesh** to extract depth-aware embeddings (x, y, z), providing **more robust identification than traditional 2D systems**.

The system is **real-time, voice-guided, and stores biometric data locally** for privacy.

---

## 🚀 Key Features
- 🧬 **3D Biometric Mapping** – Uses 478 facial landmarks  
- 🗣️ **Voice-Guided AI** – Real-time TTS via `pyttsx3`  
- 🧠 **High-Precision Recognition** – Euclidean distance matching  
- 📝 **Automated Attendance Logging** – CSV reports generated daily  
- ⚡ **Real-Time OpenCV UI** – Optimized with auto-stop  
- 🔐 **Privacy-Focused** – No cloud upload, local storage only  

---

## 🛠️ Tech Stack
Python 3.9+, OpenCV, MediaPipe Face Landmarker, pyttsx3, NumPy, JSON, CSV  

---

## 📂 Project Structure
```plaintext
BioScan-3D/
├── register_face.py        # Student enrollment module
├── recognize_face.py       # Real-time face recognition
├── attendance.py           # Attendance logger
├── face_landmarker.task    # MediaPipe 3D Face model
├── requirements.txt        # Python dependencies
├── models/                 # Supporting models
├── embeddings/             # Local biometric data (ignored by git)
└── README.md               # Project documentation

⚙️ Installation & Setup
# Clone repository
git clone https://github.com/abhinandan099/BioScan-3D.git
cd BioScan-3D

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

▶️ Usage
Register a Student
python register_face.py

Recognize a Face
python recognize_face.py

Mark Attendance
python attendance.py

📊 Sample Attendance CSV
| Reg_No | Name       | Course    | Batch | Blood_Group | Date       | Day    | Time     | Status  |
| ------ | ---------- | --------- | ----- | ----------- | ---------- | ------ | -------- | ------- |
| 101    | ABHINANDAN | B-TECH CS | 2026  | B+          | 29-01-2026 | Monday | 09:02 AM | PRESENT |

Attendance files saved as attendance_YYYY-MM-DD.csv.

👨‍💻 Author
Abhinandan Trivedi
GitHub: https://github.com/abhinandan099

🚧 Future Enhancements
GUI interface (Tkinter / PyQt)
Multi-face recognition
Database integration (MySQL / MongoDB)
Cloud backup & liveness detection

