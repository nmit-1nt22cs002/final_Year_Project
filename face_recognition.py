import cv2
import numpy as np
import os

# ===== Helper Functions =====
def distance(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())

def knn(train, test, k=5):
    dist = []
    for i in range(train.shape[0]):
        ix = train[i, :-1]
        iy = train[i, -1]
        d = distance(test, ix)
        dist.append([d, iy])
    dist = sorted(dist, key=lambda x: x[0])[:k]
    labels = np.array(dist)[:, -1]
    output = np.unique(labels, return_counts=True)
    index = np.argmax(output[1])
    return output[0][index], dist[0][0]

# ===== Load Dataset =====
dataset_path = r"D:\CODE\final year project\Real-time-Face-Recognition-Project\face_dataset"
face_data = []
labels = []
class_id = 0
names = {}

print("Loading dataset...")

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        data_item = np.load(os.path.join(dataset_path, fx))
        face_data.append(data_item)
        target = class_id * np.ones((data_item.shape[0],))
        labels.append(target)
        names[class_id] = fx[:-4]
        class_id += 1
        print(f"Loaded {fx} with shape {data_item.shape}")

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
trainset = np.concatenate((face_dataset, face_labels), axis=1)

print("✅ Dataset Loaded")
print("Trainset shape:", trainset.shape)
print("Labels mapping:", names)

# ===== Load Haar Cascade =====
HAAR_PATH = r"D:\CODE\final year project\Real-time-Face-Recognition-Project\haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(HAAR_PATH)
if face_cascade.empty():
    raise IOError("❌ Could not load Haar Cascade. Check path: " + HAAR_PATH)

# ===== Start Video Capture =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ Could not open camera.")
    exit()

print("\n🎥 Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        offset = 10
        face_section = gray[y-offset:y+h+offset, x-offset:x+w+offset]
        face_section = cv2.resize(face_section, (100, 100))

        # Normalize brightness but not scale
        face_section = cv2.equalizeHist(face_section)
        face_section = face_section.flatten()

        out_label, min_dist = knn(trainset, face_section)
        print("Min distance:", round(min_dist, 2))

        # Adaptive threshold
        if min_dist < 3000:
            pred_name = names[int(out_label)]
        else:
            pred_name = "Unknown"

        cv2.putText(frame, pred_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
