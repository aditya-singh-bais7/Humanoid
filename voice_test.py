import cv2
import face_recognition
import numpy as np
import os
import time
import psutil
import threading
import queue
import speech_recognition as sr

# --- GPU Monitoring (NVIDIA Only) ---
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_ENABLED = True
except (ImportError, Exception):
    GPU_MONITORING_ENABLED = False

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.35
DETECTION_MODEL = "hog"
UPSAMPLE_TIMES = 1
NUM_JITTERS = 1
PROCESS_EVERY_N_FRAMES = 5
REGISTRATION_TIME_SECONDS = 15

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
known_encodings = []
known_names = []
print("[INFO] Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    if filename.endswith(".npy"):
        try:
            encoding = np.load(os.path.join(KNOWN_FACES_DIR, filename))
            name = os.path.splitext(filename)[0]
            known_encodings.append(encoding)
            known_names.append(name)
            print(f"- Found {name}")
        except Exception as e:
            print(f"[WARNING] Could not load encoding for {filename}: {e}")

print("[INFO] Initializing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

STATE_RECOGNIZING = "recognizing"
STATE_GETTING_NAME = "getting_name"
STATE_REGISTERING = "registering"
current_state = STATE_RECOGNIZING
registration_name_buffer = ""
registration_encodings = []
registration_start_time = 0
frame_count = 0
start_time = time.time()
last_stats_update_time = 0
cpu_usage, mem_usage, gpu_usage = 0, 0, "N/A"
fps = 0
last_known_locations = []
last_known_names = []

# --- Speech Recognition Thread Setup ---
def listen_for_command(cmd_queue):
    r = sr.Recognizer()
    mic = sr.Microphone()
    while True:
        with mic as source:
            print("[VOICE] Listening for 'register face'...")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source, phrase_time_limit=3)
        try:
            phrase = r.recognize_google(audio).lower()
            print(f"[VOICE] Heard: {phrase}")
            if "register face" in phrase:
                cmd_queue.put("register")
        except sr.UnknownValueError:
            pass  # ignore and re-listen

def get_name_from_voice():
    r = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        print("[VOICE] Please state your name:")
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source, phrase_time_limit=3)
    try:
        name = r.recognize_google(audio)
        print(f"[VOICE] Name recognized as: {name}")
        return name.strip().replace(" ", "_")  # To avoid file issues
    except sr.UnknownValueError:
        print("[VOICE] Could not understand name input.")
        return ""

cmd_queue = queue.Queue()
voice_thread = threading.Thread(target=listen_for_command, args=(cmd_queue,), daemon=True)
voice_thread.start()

print("\n[INFO] Camera started. Say 'register face' to register a new face. Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break
    current_time = time.time()

    # Poll for registration command from voice
    if current_state == STATE_RECOGNIZING and not cmd_queue.empty():
        voice_cmd = cmd_queue.get()
        if voice_cmd == "register" and "Unknown" in last_known_names:
            current_state = STATE_GETTING_NAME
            registration_name_buffer = get_name_from_voice()
            if registration_name_buffer != "":
                current_state = STATE_REGISTERING
                registration_start_time = time.time()
                registration_encodings.clear()
            else:
                # If name wasn't captured go back to recognizing
                current_state = STATE_RECOGNIZING

    # --- State Machine Logic ---
    if current_state == STATE_RECOGNIZING:
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_small_frame, number_of_times_to_upsample=UPSAMPLE_TIMES, model=DETECTION_MODEL)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations, num_jitters=NUM_JITTERS)
            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                if known_encodings:
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < TOLERANCE:
                        name = known_names[best_match_index]
                face_names.append(name)
            last_known_locations = [(top * 4, right * 4, bottom * 4, left * 4)
                                    for (top, right, bottom, left) in face_locations]
            last_known_names = face_names

        if "Unknown" in last_known_names:
            cv2.putText(frame, "Say 'register face' to register", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    elif current_state == STATE_REGISTERING:
        elapsed_time = current_time - registration_start_time
        remaining_time = max(0, REGISTRATION_TIME_SECONDS - elapsed_time)
        cv2.putText(frame, f"Registering '{registration_name_buffer}'...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        cv2.putText(frame, f"Time left: {remaining_time:.1f}s. Look around.", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        if frame_count % (PROCESS_EVERY_N_FRAMES // 2 + 1) == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(
                rgb_small_frame, number_of_times_to_upsample=UPSAMPLE_TIMES, model=DETECTION_MODEL)
            if face_locations:
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations, num_jitters=NUM_JITTERS)
                registration_encodings.append(face_encodings[0])
        if remaining_time <= 0:
            if len(registration_encodings) > 5:
                avg_encoding = np.mean(registration_encodings, axis=0)
                np.save(os.path.join(KNOWN_FACES_DIR, f"{registration_name_buffer}.npy"), avg_encoding)
                known_encodings.append(avg_encoding)
                known_names.append(registration_name_buffer)
                print(f"[INFO] Successfully registered {registration_name_buffer}")
            else:
                print("[WARNING] Registration failed: Not enough face samples collected.")
            current_state = STATE_RECOGNIZING
            registration_name_buffer = ""
            registration_encodings.clear()

    # --- Draw boxes and names on EVERY frame ---
    for (top, right, bottom, left), name in zip(last_known_locations, last_known_names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # --- Update and Display Performance Metrics ---
    frame_count += 1
    if current_time - last_stats_update_time > 1.0:
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        if GPU_MONITORING_ENABLED:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = f"{gpu_info.gpu}%"
        elapsed_time = current_time - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        start_time, frame_count = current_time, 0
        last_stats_update_time = current_time
    stats_text = f"CPU: {cpu_usage:.1f}% | RAM: {mem_usage:.1f}% | GPU: {gpu_usage} | FPS: {fps:.1f}"
    cv2.putText(frame, stats_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imshow("Face Recognition", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

# --- Cleanup ---
if GPU_MONITORING_ENABLED:
    pynvml.nvmlShutdown()
cap.release()
cv2.destroyAllWindows()
