
```markdown
# 🎯 Face Recognition Project

A real-time face recognition system using Python, OpenCV, and the `face_recognition` library.

This project supports three modes:  
- **Live:** Real-time webcam recognition  
- **Batch:** Test a folder of images  
- **Test:** Run sample test images one by one  
It logs results, shows accuracy, detects multiple faces, and avoids saving duplicate unknowns.

---

## 📁 Project Structure

```

Face-Recognition/
├── known\_faces/          # Store known people's subfolders (e.g., known\_faces/chetan/\*.jpg)
├── unknown\_faces/        # Automatically saved unknown faces
├── batch\_images/         # Batch testing input images
├── test\_images/          # Sample test images (for test mode)
├── results\_log.csv       # CSV log of all recognition results
├── main.py               # Main script for all modes
└── .gitignore            # Excludes image data from Git history

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

1. Place face images in `known_faces/<person_name>/` (e.g., `known_faces/chetan/img1.jpg`)

2. Run the script:

   ```bash
   python main.py
   ```

3. Choose one of the modes: `live`, `batch`, or `test`

4. Results will appear in the terminal and optionally in image windows

5. Logs are saved in `results_log.csv`

---

## 📌 Features Implemented

* ✅ **Live recognition** with webcam
* ✅ **Batch mode** for processing multiple images
* ✅ **Test mode** for checking single sample images
* ✅ **Accuracy %**, **FPS**, and **face count** display
* ✅ **Side-by-side comparison** for known matches
* ✅ **CSV logging** of all recognized faces
* ✅ **Duplicate prevention** when saving unknowns
* ✅ Organized folder structure for easy usage

---

## 🔧 Upcoming Improvements

* GUI-based interface (Tkinter or PyQT)
* Face training optimization
* Export logs in Excel/PDF format
* Face recognition tuning for low-light environments

---

## 🤝 Credits

* [face\_recognition](https://github.com/ageitgey/face_recognition)
* [OpenCV](https://opencv.org/)
* Author: **Chetan Palta**

---

## 📄 License

Licensed under the [MIT License](./LICENSE)
