# 🛑 Driver Drowsiness & Fatigue Detection using YOLOv8 + Streamlit

A real-time system that detects **drowsiness**, **yawning**, and **abnormal head movement** using a webcam and a custom-trained YOLOv8 model. The system triggers alerts, plays alarms, and sends **SMS notifications via Twilio** to family members when a dangerous state is detected.

---

## 🚀 Key Features

- 👁️ Eye Closure & Drowsiness Detection  
- 🥱 Yawning Alert System  
- ↕️ Head Movement Monitoring (Down / Up)  
- 🔔 Alarm Sounds via Pygame  
- 📩 SMS Alert Integration with Twilio  
- 📷 Real-Time Detection using Webcam  
- 🧠 YOLOv8 Custom Model  
- 📦 Deployable with **Streamlit** on Hugging Face or Railway

---

## 🧠 Model Labels Used

| Label ID | Description       |
|----------|-------------------|
| 0        | Eye Closed (Left) |
| 1        | Eye Closed (Right)|
| 2        | Focused           |
| 4        | Head Down         |
| 5        | Head Up           |
| 8        | Yawning           |

---

## 🖥️ Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/YourUsername/drowsiness-detection
cd drowsiness-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run final.py
🌐 Deployment Options
✅ Hugging Face Spaces
Create a new Space (Streamlit)

Upload the following:

final.py

requirements.txt

Procfile

best.pt

Alarm MP3 files
📲 Twilio SMS Setup (Optional)
To enable family alert notifications:

1.Create a Twilio account

2.Get your:

ACCOUNT SID

AUTH TOKEN

3.Verified phone numbers

4.Enter them in the Streamlit sidebar during app execution.
🛠 Libraries Used
opencv-python

ultralytics (YOLOv8)

pygame

pyttsx3

streamlit

twilio

numpy


👨‍💻 Contributors
Joel Giftson
Seenivasaperumal
