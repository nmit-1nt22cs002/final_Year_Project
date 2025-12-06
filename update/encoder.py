# encoder.py - Save this file in the same main folder as the others.

from imutils import paths
import face_recognition
import pickle
import cv2
import os

# Your dataset folder
DATASET_DIR = "dataset"
# The output file where encodings will be stored
ENCODINGS_FILE = "encodings.pickle"

# grab the paths to the input images in the dataset folder
print("[INFO] quantifying faces...")
imagePaths = list(paths.list_images(DATASET_DIR))

# initialize the list of known encodings and known names
knownEncodings = []
knownNames = []

# loop over the image paths
for (i, imagePath) in enumerate(imagePaths):
    # extract the person name from the image path
    print(f"[INFO] processing image {i + 1}/{len(imagePaths)}")
    # The name is derived from the immediate parent folder name
    name = imagePath.split(os.path.sep)[-2]

    # --- FIX 1: SKIP the PREVIEW folder if it was mistakenly created ---
    if name.upper() == "PREVIEW":
        print("[WARNING] Skipping files in 'PREVIEW' folder, as they are not for encoding.")
        continue
    # -----------------------------------------------------------------

    # load the input image
    image = cv2.imread(imagePath)

    # --- FIX 2: Check if the image was successfully loaded ---
    # The previous error occurred here because image was None
    if image is None:
        print(f"[ERROR] Failed to read image: {imagePath}. Skipping file.")
        continue
    # ------------------------------------------------------
    
    # convert it from BGR (OpenCV) to RGB (face_recognition)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image
    boxes = face_recognition.face_locations(rgb, model='hog')

    # compute the facial embedding for the face
    encodings = face_recognition.face_encodings(rgb, boxes)

    # loop over the encodings (usually one face per image)
    for encoding in encodings:
        # add each encoding + name to our set of known names and encodings
        knownEncodings.append(encoding)
        knownNames.append(name)

# dump the facial encodings + names to disk
print(f"[INFO] serializing {len(knownEncodings)} encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
with open(ENCODINGS_FILE, "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Encoding complete. 'encodings.pickle' created.")