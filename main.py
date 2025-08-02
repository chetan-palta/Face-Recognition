import cv2
import face_recognition
import os
import numpy as np
import time
from datetime import datetime
import csv
import hashlib

# === Config ===
KNOWN_FACES_DIR = r'known_faces'
UNKNOWN_FACES_DIR = r'unknown_faces'
BATCH_IMAGES_DIR = r'batch_images'
LOG_FILE = 'results_log.csv'

TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'cnn'  # 'hog' or 'cnn'
SAVE_UNKNOWN = True

# === Load Known Faces ===
print("Loading known faces...")
known_faces = []
known_names = []

os.makedirs(UNKNOWN_FACES_DIR, exist_ok=True)
os.makedirs(BATCH_IMAGES_DIR, exist_ok=True)

# Updated loading for multiple images per person
for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        path = os.path.join(person_dir, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        for enc in encodings:
            known_faces.append(enc)
            known_names.append(person_name)

# === CSV Log Setup ===
with open(LOG_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Timestamp', 'Image', 'Name', 'Accuracy'])

# === Mode Selection ===
mode = input("Choose mode - [live / batch / test]: ").strip().lower()


# === Batch Mode ===
if mode == 'batch':
    for image_file in os.listdir(BATCH_IMAGES_DIR):
        image_path = os.path.join(BATCH_IMAGES_DIR, image_file)
        input_image = face_recognition.load_image_file(image_path)
        rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model=MODEL)
        encodings = face_recognition.face_encodings(rgb, locations)

        for face_encoding, face_location in zip(encodings, locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERANCE)
            distances = face_recognition.face_distance(known_faces, face_encoding)

            name = "Unknown"
            accuracy = 0.0

            if True in matches:
                best_match_index = np.argmin(distances)
                name = known_names[best_match_index]
                accuracy = (1 - distances[best_match_index]) * 100

            # Save log
            with open(LOG_FILE, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now(), image_file, name, f"{accuracy:.2f}"])

            # Draw
            top, right, bottom, left = face_location
            cv2.rectangle(input_image, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name} ({accuracy:.2f}%)"
            cv2.putText(input_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Side-by-side compare
            if name != "Unknown":
                # Find first image of this person to compare
                person_dir = os.path.join(KNOWN_FACES_DIR, name)
                if os.path.isdir(person_dir):
                    first_image = next((f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', ',.png'))), None)
                    if first_image:
                        known_image_path = os.path.join(person_dir, first_image)
                        known_img = cv2.imread(known_image_path)
                        known_img = cv2.resize(known_img, (input_image.shape[1], input_image.shape[0]))
                        comparison = np.hstack((known_img, input_image))
                        cv2.imshow(f"{name} comparison", comparison)
            else:
                cv2.imshow("Unknown", input_image)

            print("Press any key to continue, 'q' to quit.")
            key = cv2.waitKey(0)
            if key == 27:  # ESC key to skip early
                continue
            elif key == ord('q'):
                cv2.destroyAllWindows()
                exit()

elif mode == 'test':
    TEST_IMAGES_DIR = 'test_images'
    for image_file in os.listdir(TEST_IMAGES_DIR):
        image_path = os.path.join(TEST_IMAGES_DIR, image_file)
        input_image = face_recognition.load_image_file(image_path)
        rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model=MODEL)
        encodings = face_recognition.face_encodings(rgb, locations)

        for face_encoding, face_location in zip(encodings, locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERANCE)
            distances = face_recognition.face_distance(known_faces, face_encoding)

            name = "Unknown"
            accuracy = 0.0

            if True in matches:
                best_match_index = np.argmin(distances)
                name = known_names[best_match_index]
                accuracy = (1 - distances[best_match_index]) * 100

            # Draw
            top, right, bottom, left = face_location
            cv2.rectangle(input_image, (left, top), (right, bottom), (0, 255, 0), 2)
            label = f"{name} ({accuracy:.2f}%)"
            cv2.putText(input_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(image_file, input_image)
        print("Press any key to continue, 'q' to quit.")
        key = cv2.waitKey(0)
        if key == ord('q'):
            cv2.destroyAllWindows()
            exit()

# === Live Mode ===
else:
    recent_unknown_encodings = []
    UNKNOWN_SAVE_THRESHOLD = 0.6  # Avoid saving same unknown multiple times
    video = cv2.VideoCapture(0)
    prev_time = time.time()

    video.set(cv2.CAP_PROP_FPS, 30)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Starting webcam. Press 'q' to exit.")
    while True:
        ret, frame = video.read()
        if not ret:
            break

        small = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        locations = face_recognition.face_locations(rgb, model=MODEL)
        encodings = face_recognition.face_encodings(rgb, locations)

        face_count = len(locations)

        for face_encoding, face_location in zip(encodings, locations):
            matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERANCE)
            distances = face_recognition.face_distance(known_faces, face_encoding)

            name = "Unknown"
            accuracy = 0.0
            box_color = (0, 0, 255)

            if True in matches:
                best_match_index = np.argmin(distances)
                name = known_names[best_match_index]
                accuracy = (1 - distances[best_match_index]) * 100
                box_color = (0, 255, 0)

            top, right, bottom, left = [v * 4 for v in face_location]
            cv2.rectangle(frame, (left, top), (right, bottom), box_color, FRAME_THICKNESS)
            label = f"{name} ({accuracy:.2f}%)" if name != "Unknown" else "Unknown"
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, FONT_THICKNESS)

            # Save unknown
            if name == "Unknown" and SAVE_UNKNOWN:
                # Avoid saving duplicates
                should_save = True
                for recent in recent_unknown_encodings:
                    distance = np.linalg.norm(recent - face_encoding)
                    if distance < UNKNOWN_SAVE_THRESHOLD:
                        should_save = False
                        break

                if should_save:
                    face_img = frame[top:bottom, left:right]
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(UNKNOWN_FACES_DIR, f"unknown_{timestamp}.jpg")
                    cv2.imwrite(path, face_img)
                    recent_unknown_encodings.append(face_encoding)

                    # Log to CSV
                    with open(LOG_FILE, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([datetime.now(), "webcam_frame", name, f"{accuracy:.2f}"])

        # FPS counter
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-5)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 255, 50), 2)

        # Face count
        cv2.putText(frame, f"Faces: {face_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()