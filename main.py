import cv2
import face_recognition
import os
import numpy as np

# Paths
KNOWN_FACES_DIR = 'known_faces'
TOLERANCE = 0.6  # lower means more strict
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'  # or 'cnn' if you have GPU

# Load known faces
print("Loading known faces...")
known_faces = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    filepath = os.path.join(KNOWN_FACES_DIR, name)
    image = face_recognition.load_image_file(filepath)
    encodings = face_recognition.face_encodings(image)

    if encodings:
        known_faces.append(encodings[0])
        known_names.append(os.path.splitext(name)[0])  # Remove .jpeg

# Start webcam
video = cv2.VideoCapture(0)

print("Starting camera. Press 'q' to quit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Resize frame to 1/4 for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame, model=MODEL)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding, tolerance=TOLERANCE)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)

        name = "Unknown"
        accuracy = 0.0

        if True in matches:
            best_match_index = np.argmin(face_distances)
            name = known_names[best_match_index]
            accuracy = (1 - face_distances[best_match_index]) * 100

        # Scale back up face locations to original size
        top, right, bottom, left = [v * 4 for v in face_location]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), FRAME_THICKNESS)

        label = f'{name} ({accuracy:.2f}%)' if name != "Unknown" else "Unknown"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), FONT_THICKNESS)

    cv2.imshow("Live Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
