import face_recognition
import cv2
import numpy as np
import time
import pickle
import pandas as pd
from datetime import datetime
import os
import sys

# --- CONFIGURATION (Must match Dashboard) ---
TOLERANCE = 0.40  
CV_SCALER = 4     
ATTENDANCE_COOLDOWN = 60 # Seconds
LOG_FILE_PATH = "attendance_log_temp.csv" # Shared log file path

# Global state for logging
attendance_log = []
last_logged_time = {}

# ... (load_encodings and mark_attendance functions remain the same) ...

def load_encodings():
    print("[INFO] loading encodings...")
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.loads(f.read())
        return data["encodings"], data["names"]
    except FileNotFoundError:
        print("[ERROR] 'encodings.pickle' not found. Run the encoder script first.")
        sys.exit()

def mark_attendance(name):
    global attendance_log, last_logged_time
    now = datetime.now()
    
    if name in last_logged_time:
        time_diff = (now - last_logged_time[name]).total_seconds()
        if time_diff < ATTENDANCE_COOLDOWN:
            return 
    
    last_logged_time[name] = now
    time_str = now.strftime("%H:%M:%S")
    date_str = now.strftime("%Y-%m-%d")
    
    attendance_log.append({"Name": name, "Date": date_str, "Time": time_str})
    print(f"[ATTENDANCE] Logged: {name} at {time_str}")

def save_log_to_temp_file():
    global attendance_log
    if attendance_log:
        df = pd.DataFrame(attendance_log)
        try:
            # Save the log to a temporary CSV file which the GUI will read later
            df.to_csv(LOG_FILE_PATH, index=False)
            print(f"[LOG] Attendance data temporarily saved to {LOG_FILE_PATH}")
        except Exception as e:
            print(f"[LOG ERROR] Could not save temporary log file: {e}")

def run_recognition():
    known_face_encodings, known_face_names = load_encodings()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Camera initialized. Running Recognition loop...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to capture frame.")
            break

        # Processing 
        small_frame = cv2.resize(frame, (0, 0), fx=1/CV_SCALER, fy=1/CV_SCALER, interpolation=cv2.INTER_AREA)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            if known_face_encodings:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                min_distance = face_distances[best_match_index]

                if min_distance < TOLERANCE:
                    name = known_face_names[best_match_index]
                    mark_attendance(name) 

            # Drawing Results (Scaled back up)
            top *= CV_SCALER
            right *= CV_SCALER
            bottom *= CV_SCALER
            left *= CV_SCALER

            color = (0, 0, 255) if name == "Unknown" else (0, 255, 0)
            
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Display the result
        cv2.imshow('Facial Recognition System', frame)
        
    # Break loop on 'q' or 'Q' or if the OpenCV window is closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    # Cleanup
    print("[INFO] Shutting down camera and saving temporary log...")
    cap.release()
    cv2.destroyAllWindows()
    save_log_to_temp_file()

if __name__ == "__main__":
    run_recognition()