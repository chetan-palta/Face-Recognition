
# 🎯 Face Recognition Project

A real-time face recognition system using Python, OpenCV, and the `face_recognition` library.

This project loads a known face (like `chetan.jpeg`) and matches it in real-time from your webcam feed. If a match is found, it displays thegit add README.md name with the accuracy percentage.

---

## 📁 Project Structure

```

FACE\_RECOGNITION/
│
├── Face-Recognition/
│   ├── known\_faces/          # Store known people (e.g., chetan.jpeg)
│   ├── unknown\_faces/        # (Optional) Place test images here
│   └── main.py               # Main script to run webcam-based recognition

````

---

## ⚙️ Requirements

- Python 3.8+
- OpenCV (`cv2`)
- face_recognition
- numpy

Install dependencies:
```bash
pip install opencv-python face_recognition numpy
````

---

## 🚀 How to Run

1. Place known face images in the `known_faces/` folder (e.g., `chetan.jpeg`).
2. Run the script:

   ```bash
   python main.py
   ```
3. A webcam window will open.
4. If your face matches a known image, your name and match accuracy will be shown.
5. Press `q` to quit.

---

## 📌 Features Implemented

* ✅ Real-time webcam face detection
* ✅ Face matching with accuracy percentage
* ✅ Matches against all known faces in the folder
* ✅ Clean exit using `q`
* ✅ Simple folder-based organization

---

## 🔧 Upcoming Improvements

* Save logs of matched faces with timestamps
* Capture & save unknown faces automatically
* Detect multiple faces in a frame
* Train from subfolders (for multiple people)
* GUI version (Tkinter or PyQT)

---

## 📸 Sample Output

![Preview](./sample_output.jpg) <!-- Optional screenshot -->

---

## 🤝 Credits

* [face\_recognition](https://github.com/ageitgey/face_recognition)
* [OpenCV](https://opencv.org/)
* Author: Only Chetan

---

## 📄 License

MIT License. Use freely with credit.

```

---
