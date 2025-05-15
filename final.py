import streamlit as st
import cv2
import pygame
import tempfile
import time
from ultralytics import YOLO
from collections import deque
import threading
from twilio.rest import Client  # Import Twilio API

# Streamlit setup
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🛑 Real-Time Driver Drowsiness Detection")

# Sidebar for uploading alarm sounds and Twilio credentials
st.sidebar.subheader("🔊 Alarm Settings")
drowsy_file = st.sidebar.file_uploader("Drowsiness Alarm (.mp3)", type="mp3")
yawn_file = st.sidebar.file_uploader("Yawn Alarm (.mp3)", type="mp3")
head_file = st.sidebar.file_uploader("Head Movement Alarm (.mp3)", type="mp3")

# Save files temporarily to play with pygame
temp_files = {}
pygame.mixer.init()

def save_and_store_temp(file, key):
    if file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            tmp.write(file.read())
            temp_files[key] = tmp.name

def play_alarm_for_duration(key, duration=4):
    """Plays alarm for a specified duration in a separate thread."""
    def _play():
        if key in temp_files:
            pygame.mixer.music.load(temp_files[key])
            pygame.mixer.music.play()
            time.sleep(duration)
            pygame.mixer.music.stop()
    threading.Thread(target=_play).start()

# Twilio credentials input
with st.sidebar.expander("📲 Twilio Credentials"):
    account_sid = st.text_input("Account SID", type="password")
    auth_token = st.text_input("Auth Token", type="password")
    from_number = st.text_input("From Number")
    to_number = st.text_input("To Number")

# Twilio Client setup
def send_sms(message):
    if account_sid and auth_token and from_number and to_number:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=message,
            from_=from_number,
            to=to_number
        )
        return message.sid
    return None

# Load model
model = YOLO("best.pt")  # Replace with your model path

# Display containers
frame_display = st.empty()
status_display = st.empty()

# Detection function
def run_detection():
    save_and_store_temp(drowsy_file, "drowsy")
    save_and_store_temp(yawn_file, "yawn")
    save_and_store_temp(head_file, "head")

    cap = cv2.VideoCapture(0)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    eye_queue = deque(maxlen=int(fps * 2))
    yawn_queue = deque(maxlen=int(fps * 2))
    head_queue = deque(maxlen=int(fps * 2))

    alarm_triggered = {"drowsy": 0, "yawn": 0, "head": 0}

    try:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            results = model.predict(frame, verbose=False)[0]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            current_eye = current_yawn = current_head = False
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                label = int(cls)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if label in [0, 1, 2]: current_eye = True
                if label == 8: current_yawn = True
                if label in [4, 5]: current_head = True

            eye_queue.append(current_eye)
            yawn_queue.append(current_yawn)
            head_queue.append(current_head)

            status = "✅ Awake"
            now = time.time()

            if sum(eye_queue) > int(fps * 0.8):
                status = "😴 Drowsy"
                if now - alarm_triggered["drowsy"] > 4:
                    play_alarm_for_duration("drowsy", 4)
                    send_sms("Alert: Driver is drowsy!")
                    alarm_triggered["drowsy"] = now
            elif sum(yawn_queue) > int(fps * 0.8):
                status = "🥱 Yawning"
                if now - alarm_triggered["yawn"] > 4:
                    play_alarm_for_duration("yawn", 4)
                    send_sms("Alert: Driver is yawning!")
                    alarm_triggered["yawn"] = now
            elif sum(head_queue) > int(fps * 0.8):
                status = "⚠️ Head Movement"
                if now - alarm_triggered["head"] > 4:
                    play_alarm_for_duration("head", 4)
                    send_sms("Alert: Driver's head movement detected!")
                    alarm_triggered["head"] = now

            frame_display.image(frame, channels="RGB")
            status_display.markdown(f"### Detection Status: {status}")

            # Ensure processing time doesn't block next frame
            elapsed = time.time() - start_time
            time_to_wait = max(1.0 / fps - elapsed, 0)
            time.sleep(time_to_wait)

    except Exception as e:
        with open("log.txt", "a") as f:
            f.write(f"[ERROR] {e}\n")
    finally:
        cap.release()
        pygame.mixer.music.stop()

# Start detection button
if st.button("🚀 Start Detection"):
    run_detection()
