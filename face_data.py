import cv2
import numpy as np
import os
#Abhimanyu
# ===== CONFIG =====
DATASET_PATH = r"D:\CODE\final year project\Real-time-Face-Recognition-Project\face_dataset"
HAAR_PATH = r"D:\CODE\final year project\Real-time-Face-Recognition-Project\haarcascade_frontalface_alt.xml"
IMG_SIZE = 100          # Size to which each face is resized
SKIP_FRAMES = 5         # Capture every 5th frame to avoid duplicates

# ===== Setup =====
os.makedirs(DATASET_PATH, exist_ok=True)
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty():
    raise IOError("❌ Haar Cascade not loaded. Check path: " + HAAR_PATH)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("⚠️ Cannot access webcam.")

# ===== Input user name =====
file_name = input("Enter the name of the person: ").strip()
if not file_name:
    raise ValueError("❌ Name cannot be empty.")

face_data = []
skip = 0
print("\n🎥 Starting face capture. Press 'q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # Sort faces by area (largest first)
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    skip += 1

    for (x, y, w, h) in faces[:1]:
        offset = 10
        face_section = gray[y-offset:y+h+offset, x-offset:x+w+offset]

        if face_section.size == 0:
            continue

        # Resize, normalize, and equalize histogram
        face_section = cv2.resize(face_section, (IMG_SIZE, IMG_SIZE))
        face_section = cv2.equalizeHist(face_section)
        face_section = face_section.astype(np.float32) / 255.0  # Normalize 0-1

        if skip % SKIP_FRAMES == 0:
            face_data.append(face_section)
            print(f"Captured sample #{len(face_data)}")

        # Display rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("Captured Face", face_section)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== Save dataset =====
if len(face_data) == 0:
    print("❌ No face data captured. Exiting.")
else:
    face_data = np.array(face_data)
    face_data = face_data.reshape((face_data.shape[0], -1))  # Flatten each face to 1D
    save_path = os.path.join(DATASET_PATH, f"{file_name}.npy")
    np.save(save_path, face_data)
    print(f"\n✅ Dataset saved at: {save_path}")
    print(f"📦 Total samples collected: {face_data.shape[0]}")

cap.release()
cv2.destroyAllWindows()
