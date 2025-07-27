import cv2
import face_recognition
import os
import numpy as np
import time
from datetime import datetime
import csv

# === Configuration ===
KNOWN_FACES_DIR = r'C:\Users\itzch\OneDrive\Desktop\Projects\face_recognition\Face-Recognition\known_faces'
UNKNOWN_FACES_DIR = r'C:\Users\itzch\OneDrive\Desktop\Projects\face_recognition\Face-Recognition\unknown_faces'
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # 'hog' or 'cnn'
SAVE_UNKNOWN = True
LOG_FILE = "face_log.csv"
# === Initialization ===
print("Loading known faces...")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, name)
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_faces.append(encodings[0])
        known_names.append(os.path.splitext(name)[0])

# Create unknown_faces folder if it doesn't exist
os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)

# Start webcam
video = cv2.VideoCapture(0)
prev_frame_time = time.time()

print("Starting camera. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize and convert
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_count = len(face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERANCE)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        name = "Unknown"
        accuracy = 0.0
        box_color = (0, 0, 255)  # Red for unknown

        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]
            accuracy = (1 - face_distances[best_match_index]) * 100
            box_color = (0, 255, 0)  # Green for known

        # Scale back up face location
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, FRAME_THICKNESS)

        label = f'{name} ({accuracy:.2f}%)' if name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, FONT_THICKNESS)

        # Save unknown face
        if name == "Unknown" and SAVE_UNKNOWN:
            face_image = frame[top:bottom, left:right]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unknown_path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{timestamp}.jpg")
            cv2.imwrite(unknown_path, face_image)

    # === FPS Calculation ===
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time + 1e-5)
    prev_frame_time = new_frame_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

    # Face count
    cv2.putText(frame, f"Faces: {face_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show frame
    cv2.imshow("Live Face Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


def log_face(name, accuracy, image_path=""):
    with open(LOG_FILE, "a", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), name, f"{accuracy:.2f}", image_path])
# Cleanup
video.release()
cv2.destroyAllWindows()
