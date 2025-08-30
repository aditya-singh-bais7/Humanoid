import cv2
import face_recognition
import numpy as np
import os
import time
import psutil
import json
from datetime import datetime # ADDED: For timestamp generation

# --- GPU Monitoring (NVIDIA Only) ---
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_MONITORING_ENABLED = True
except (ImportError, Exception):
    GPU_MONITORING_ENABLED = False

# --- CONFIGURATION ---
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.35
MODEL = "hog"
PROCESS_EVERY_N_FRAMES = 5
REGISTRATION_TIME_SECONDS = 15

# --- Load Known Faces ---
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
known_encodings = []
known_names = []
known_descriptions = []
print("[INFO] Loading known faces...")
for filename in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, filename)
    name = os.path.splitext(filename)[0]
    try:
        if filename.endswith(".json"):
            with open(filepath, 'r') as f:
                data = json.load(f)
                known_encodings.append(np.array(data["encoding"]))
                known_names.append(data["name"])
                known_descriptions.append(data["description"])
                print(f"- Found {data['name']} (Description: {data['description']})")
        elif filename.endswith(".npy"):
            encoding = np.load(filepath)
            known_encodings.append(encoding)
            known_names.append(name)
            known_descriptions.append("No description")
            print(f"- Found {name} (Legacy format)")
    except Exception as e:
        print(f"[WARNING] Could not load data for {filename}: {e}")

# --- Initialize Camera ---
print("[INFO] Initializing camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("[ERROR] Cannot open camera.")
    exit()

# --- State & Performance Variables ---
STATE_RECOGNIZING = "recognizing"
STATE_GETTING_NAME = "getting_name"
# REMOVED: STATE_GETTING_DESCRIPTION
STATE_REGISTERING = "registering"
current_state = STATE_RECOGNIZING

registration_name_buffer = ""
# REMOVED: registration_description_buffer
current_registration_description = "" # ADDED: To hold the generated description
registration_encodings = []
registration_start_time = 0

frame_count = 0
start_time = time.time()
last_stats_update_time = 0
cpu_usage, mem_usage, gpu_usage = 0, 0, "N/A"
fps = 0

last_known_locations = []
last_known_names = []

print("\n[INFO] Camera started. Press 'ESC' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame from camera.")
        break

    current_time = time.time()
    
    # --- State Machine Logic ---
    if current_state == STATE_RECOGNIZING:
        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                name = "Unknown"
                if known_encodings:
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if face_distances[best_match_index] < TOLERANCE:
                        name = known_names[best_match_index]
                face_names.append(name)
            
            last_known_locations = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in face_locations]
            last_known_names = face_names
        
        if "Unknown" in last_known_names:
            cv2.putText(frame, "Press 'r' to register new face", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

    elif current_state == STATE_GETTING_NAME:
        cv2.putText(frame, "Enter Name:", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, registration_name_buffer, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Press ENTER to confirm", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # REMOVED: The entire state block for getting a description manually
    
    elif current_state == STATE_REGISTERING:
        elapsed_time = current_time - registration_start_time
        remaining_time = max(0, REGISTRATION_TIME_SECONDS - elapsed_time)
        
        cv2.putText(frame, f"Registering '{registration_name_buffer}'...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)
        cv2.putText(frame, f"Time left: {remaining_time:.1f}s. Look around.", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 2)

        if frame_count % (PROCESS_EVERY_N_FRAMES // 2 + 1) == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                registration_encodings.append(face_encodings[0])

        if remaining_time <= 0:
            if len(registration_encodings) > 5:
                avg_encoding = np.mean(registration_encodings, axis=0)
                
                # MODIFIED: Use the dynamically generated description
                face_data = {
                    "name": registration_name_buffer,
                    "description": current_registration_description,
                    "encoding": avg_encoding.tolist()
                }
                json_path = os.path.join(KNOWN_FACES_DIR, f"{registration_name_buffer}.json")
                with open(json_path, 'w') as f:
                    json.dump(face_data, f)

                known_encodings.append(avg_encoding)
                known_names.append(registration_name_buffer)
                known_descriptions.append(current_registration_description)
                print(f"[INFO] Successfully registered {registration_name_buffer}")
            else:
                print("[WARNING] Registration failed: Not enough face samples collected.")
            
            current_state = STATE_RECOGNIZING
            registration_name_buffer = ""
            current_registration_description = "" # Reset the temp description
            registration_encodings.clear()

    # --- Draw boxes and names on EVERY frame ---
    for (top, right, bottom, left), name in zip(last_known_locations, last_known_names):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        if name != "Unknown":
            try:
                idx = known_names.index(name)
                description = known_descriptions[idx]
                cv2.putText(frame, description, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            except ValueError:
                pass

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

    # --- Display the final frame ---
    cv2.imshow("Face Recognition", frame)

    # --- Key Handling ---
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    
    if current_state == STATE_RECOGNIZING and key == ord('r'):
        if "Unknown" in last_known_names:
            current_state = STATE_GETTING_NAME
    elif current_state == STATE_GETTING_NAME:
        if key == 13: # ENTER
            if registration_name_buffer:
                # MODIFIED: Generate description and go directly to registering
                now = datetime.now()
                # Format: "Registered on Aug 26, 2025 at 11:40 AM"
                date_str = now.strftime("%b %d, %Y at %I:%M %p") 
                current_registration_description = f"Registered on {date_str}"

                current_state = STATE_REGISTERING
                registration_start_time = time.time()
                registration_encodings.clear()
        elif key == 8: # BACKSPACE
            registration_name_buffer = registration_name_buffer[:-1]
        elif 32 <= key <= 126:
            registration_name_buffer += chr(key)

# --- Cleanup ---
if GPU_MONITORING_ENABLED:
    pynvml.nvmlShutdown()
cap.release()
cv2.destroyAllWindows()