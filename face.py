import cv2
import face_recognition
import numpy as np
import os
import time
import psutil  # For CPU and Memory usage

# --- GPU Monitoring (NVIDIA Only) ---
try:
    import pynvml

    pynvml.nvmlInit()
    GPU_MONITORING_ENABLED = True
except (ImportError, Exception):
    GPU_MONITORING_ENABLED = False

# --- CONFIGURATION ---
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.55
MODEL = "hog"
PROCESS_EVERY_N_FRAMES = 4  # Reduce CPU load by processing every 4th frame

# --- State Management ---
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

# --- Initialize Cameras ---
print("[INFO] Initializing cameras...")

cap0 = cv2.VideoCapture(0)  # External cam 1
cap3 = cv2.VideoCapture(3)  # External cam 2

if not cap0.isOpened():
    print("[ERROR] Cannot open Camera 1.")
if not cap3.isOpened():
    print("[ERROR] Cannot open Camera 3.")
if not cap0.isOpened() or not cap3.isOpened():
    print("[ERROR] Exiting due to camera error.")
    exit()

# --- Performance Metrics Variables ---
fps_start_time_1 = time.time()
fps_frame_count_1 = 0
fps_1 = 0

fps_start_time_3 = time.time()
fps_frame_count_3 = 0
fps_3 = 0

last_stats_update_time = 0
cpu_usage = 0
mem_usage = 0
gpu_usage = "N/A"

print("\n[INFO] Webcams started. Press 'ESC' to quit.")


def process_frame(frame, known_encodings, known_names, tolerance, model):
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_small_frame, model=model)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        name = "Unknown"
        if known_encodings:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if face_distances[best_match_index] < tolerance:
                name = known_names[best_match_index]
        face_names.append(name)

    # Scale face location coords to full frame size
    scaled_locations = [(top * 4, right * 4, bottom * 4, left * 4) for (top, right, bottom, left) in face_locations]
    return scaled_locations, face_names


while True:
    ret0, frame0 = cap0.read()
    ret3, frame3 = cap3.read()
    if not ret0 or not ret3:
        print("[ERROR] Failed to grab frame from one or more cameras.")
        break

    fps_frame_count_1 += 1
    fps_frame_count_3 += 1
    current_time = time.time()

    # Calculate FPS for cam1
    elapsed_1 = current_time - fps_start_time_1
    if elapsed_1 >= 1.0:
        fps_1 = fps_frame_count_1 / elapsed_1
        fps_start_time_1 = current_time
        fps_frame_count_1 = 0

    # Calculate FPS for cam3
    elapsed_3 = current_time - fps_start_time_3
    if elapsed_3 >= 1.0:
        fps_3 = fps_frame_count_3 / elapsed_3
        fps_start_time_3 = current_time
        fps_frame_count_3 = 0

    # Update system stats once per second
    if current_time - last_stats_update_time > 1.0:
        cpu_usage = psutil.cpu_percent()
        mem_usage = psutil.virtual_memory().percent
        if GPU_MONITORING_ENABLED:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = f"{gpu_info.gpu}%"
        last_stats_update_time = current_time

    # Process frames only every N frames for efficiency
    if (fps_frame_count_1 % PROCESS_EVERY_N_FRAMES == 0):
        locations1, names1 = process_frame(frame0, known_encodings, known_names, TOLERANCE, MODEL)
    else:
        locations1, names1 = [], []

    if (fps_frame_count_3 % PROCESS_EVERY_N_FRAMES == 0):
        locations3, names3 = process_frame(frame3, known_encodings, known_names, TOLERANCE, MODEL)
    else:
        locations3, names3 = [], []

    # Draw boxes and labels on camera 1
    for (top, right, bottom, left), name in zip(locations1, names1):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame0, (left, top), (right, bottom), color, 2)
        cv2.putText(frame0, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Draw boxes and labels on camera 3
    for (top, right, bottom, left), name in zip(locations3, names3):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame3, (left, top), (right, bottom), color, 2)
        cv2.putText(frame3, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Overlay system stats and FPS on each frame
    stats_text_cam1 = f"FPS: {fps_1:.2f} | CPU: {cpu_usage:.1f}% | RAM: {mem_usage:.1f}% | GPU: {gpu_usage}"
    stats_text_cam3 = f"FPS: {fps_3:.2f} | CPU: {cpu_usage:.1f}% | RAM: {mem_usage:.1f}% | GPU: {gpu_usage}"
    cv2.putText(frame0, stats_text_cam1, (10, frame0.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame3, stats_text_cam3, (10, frame3.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Show frames
    cv2.imshow("Camera 1 (External)", frame0)
    cv2.imshow("Camera 3 (External)", frame3)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key to exit
        break
    # Add registration or other key handling below if needed

# Cleanup
if GPU_MONITORING_ENABLED:
    pynvml.nvmlShutdown()

cap0.release()
cap3.release()
cv2.destroyAllWindows()
