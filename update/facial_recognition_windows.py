import face_recognition
import cv2
import numpy as np
import time
import pickle

# --- Configuration ---
# ⚠️ ACTION REQUIRED: You might need to adjust this value!
# 0.6 is the default. Try lowering it (e.g., 0.5) for stricter matching.
# Try raising it (e.g., 0.65 or 0.7) for looser matching.
TOLERANCE = 0.42 
cv_scaler = 4 # for performance/accuracy trade-off (1 = full resolution, slower)
# --- End Configuration ---

# Load pre-trained face encodings
print("[INFO] loading encodings...")
try:
    with open("encodings.pickle", "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]
except FileNotFoundError:
    print("[ERROR] 'encodings.pickle' not found. Run the encoding script first.")
    exit()

# Initialize the webcam 
cap = cv2.VideoCapture(0)

# Set the resolution (if previously requested)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 

print(f"[INFO] Camera initialized with resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")


# Initialize our variables
face_locations = []
face_encodings = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

def process_frame(frame):
    """
    Processes the frame to find faces and compare them to known encodings,
    using the TOLERANCE to filter unknowns.
    """
    global face_locations, face_encodings, face_names
    
    # Resize the frame using cv_scaler to increase performance
    resized_frame = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler), interpolation=cv2.INTER_AREA)
    rgb_resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_resized_frame, model='hog') 
    face_encodings = face_recognition.face_encodings(rgb_resized_frame, face_locations)
    
    face_names = []
    
    for face_encoding in face_encodings:
        name = "Unknown"
        
        # Check if there are known faces to compare against
        if known_face_encodings:
            # Calculate the distance to every known face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            
            # Find the index of the face with the minimum distance (best match)
            best_match_index = np.argmin(face_distances)
            
            # Get the minimum distance
            min_distance = face_distances[best_match_index]

            # --- THE CRITICAL LOGIC ---
            # If the best match distance is LESS than the tolerance, it is a known person.
            if min_distance < TOLERANCE:
                name = known_face_names[best_match_index]
            
            # 💡 Helpful Debugging Tip: Print the distance to see what value is being calculated.
            print(f"Closest match distance: {min_distance:.3f} | Result: {name}")

        face_names.append(name)
    
    return frame

# ... (draw_results and calculate_fps functions remain the same) ...

def draw_results(frame):
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled
        top *= cv_scaler
        right *= cv_scaler
        bottom *= cv_scaler
        left *= cv_scaler
        
        # Determine color based on whether the person is known
        color = (0, 0, 255) if name == "Unknown" else (244, 42, 3) # Red for Unknown, Orange-Red for Known
        
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left -3, top - 35), (right+3, top), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
    
    return frame

def calculate_fps():
    global frame_count, start_time, fps
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()
    return fps

## Main Video Loop

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("[ERROR] Failed to capture frame from camera. Exiting...")
        break
        
    processed_frame = process_frame(frame)
    display_frame = draw_results(processed_frame)
    
    current_fps = calculate_fps()
    cv2.putText(display_frame, f"FPS: {current_fps:.1f}", (display_frame.shape[1] - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Video', display_frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

# Shutting down
print("[INFO] Shutting down...")
cv2.destroyAllWindows()
cap.release()