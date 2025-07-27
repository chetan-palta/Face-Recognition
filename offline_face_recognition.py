import face_recognition
import cv2
import os
import numpy as np
from datetime import datetime

# Paths
KNOWN_FACES_DIR = "known_faces"
TEST_IMAGES_DIR = "test_images"
RESULTS_DIR = "results"

TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"  # or 'cnn' if you have GPU

# Create results folder if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load known faces
print("[INFO] Loading known faces...")
known_faces = []
known_names = []

for filename in os.listdir(KNOWN_FACES_DIR):
    image_path = os.path.join(KNOWN_FACES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_faces.append(encodings[0])
        known_names.append(os.path.splitext(filename)[0])

print(f"[INFO] Found {len(known_faces)} known face(s).")

# Process test images
print(f"[INFO] Processing images in '{TEST_IMAGES_DIR}'...")
for filename in os.listdir(TEST_IMAGES_DIR):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    print(f"\n➡ Processing: {filename}")
    image_path = os.path.join(TEST_IMAGES_DIR, filename)
    image = face_recognition.load_image_file(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    face_locations = face_recognition.face_locations(image, model=MODEL)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        name = "Unknown"
        accuracy = 0.0
        color = (0, 0, 255)

        if True in matches:
            best_index = np.argmin(face_distances)
            name = known_names[best_index]
            accuracy = (1 - face_distances[best_index]) * 100
            color = (0, 255, 0)

        # Draw box
        cv2.rectangle(rgb_image, (left, top), (right, bottom), color, FRAME_THICKNESS)
        label = f"{name} ({accuracy:.2f}%)" if name != "Unknown" else "Unknown"
        cv2.putText(rgb_image, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, FONT_THICKNESS)

    # Save result
    save_path = os.path.join(RESULTS_DIR, f"result_{filename}")
    cv2.imwrite(save_path, rgb_image)
    print(f"[✔] Saved result to: {save_path}")

print("\n🎉 All images processed successfully.")
