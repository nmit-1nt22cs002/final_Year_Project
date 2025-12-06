import cv2
import os
from datetime import datetime
import sys 
import time

# --- Configuration ---
DATASET_DIR = "dataset" # Folder where all images will be saved
AUTO_CAPTURE_TARGET = 150 # Target number of images for auto mode
AUTO_CAPTURE_DELAY = 0.2 # Delay in seconds between auto captures (5 frames per second)

def create_folder(name):
    # If the name is the special preview placeholder, do not create a folder.
    if name == "PREVIEW":
        return None
        
    # Ensure the main dataset folder exists
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    # Create the person's specific folder
    person_folder = os.path.join(DATASET_DIR, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
        print(f"[INFO] Created new folder: {person_folder}")
    else:
        print(f"[INFO] Using existing folder: {person_folder}")
    return person_folder

def capture_photos(name, mode):
    folder = create_folder(name)
    
    # Determine the operating mode
    is_preview_mode = (mode == "PREVIEW")
    is_auto_mode = (mode == "AUTO")
    is_capture_mode = not is_preview_mode
    
    # Initialize the webcam 
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check if a webcam is connected and available.")
        return

    # Set a standard resolution for capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    photo_count = 0
    
    if is_auto_mode:
        print(f"Starting AUTO capture for {name}. Target: {AUTO_CAPTURE_TARGET} images.")
    elif is_capture_mode:
        # Changed instruction text to reflect the fix
        print(f"Starting MANUAL capture for {name}. Press SPACE to capture, 'Q' or 'q' to quit.")
    else:
        # Changed instruction text to reflect the fix
        print(f"Running camera in PREVIEW mode. Press 'Q' or 'q' to quit.")

    last_capture_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("[ERROR] Cannot read frame from camera. Exiting...")
            break
        
        # --- UI DISPLAY ---
        display_name = 'Preview' if is_preview_mode else name
        cv2.putText(frame, f"Person: {display_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        if is_auto_mode:
            status_text = f"AUTO MODE ({photo_count}/{AUTO_CAPTURE_TARGET})"
            instruction_text = "Move your head slowly! Press 'q' or 'Q' to stop early."
            text_color = (0, 255, 255) # Yellow
            
            # --- AUTO CAPTURE LOGIC ---
            current_time = time.time()
            if photo_count < AUTO_CAPTURE_TARGET and (current_time - last_capture_time) >= AUTO_CAPTURE_DELAY:
                
                # Check that a folder exists (it should, as is_preview_mode is false)
                if folder:
                    photo_count += 1
                    # Added microseconds for unique filename
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f") 
                    filename = f"{name}_{timestamp}.jpg"
                    filepath = os.path.join(folder, filename)
                    cv2.imwrite(filepath, frame)
                    print(f"Captured {photo_count}/{AUTO_CAPTURE_TARGET}")
                    last_capture_time = current_time
            elif photo_count >= AUTO_CAPTURE_TARGET:
                # Target reached
                break
                
        elif is_capture_mode:
            status_text = f"MANUAL MODE (Captures: {photo_count})"
            instruction_text = "Press SPACE to capture, 'Q' or 'q' to quit."
            text_color = (0, 255, 0) # Green
        else: # Preview mode
            status_text = "PREVIEW MODE - SAVING DISABLED"
            instruction_text = "Press 'Q' or 'q' to quit."
            text_color = (0, 0, 255) # Red

        # Update HUD
        cv2.putText(frame, status_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.putText(frame, instruction_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow('Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' '):  # Space key
            if not is_capture_mode:
                # Ignore space key press in auto/preview mode
                continue

            # MANUAL CAPTURE LOGIC
            photo_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = os.path.join(folder, filename)
            cv2.imwrite(filepath, frame)
            print(f"Captured {filename}")
        
        # FIX: Check for both lowercase 'q' (ASCII 113) and uppercase 'Q' (ASCII 81)
        elif key == ord('q') or key == ord('Q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if is_capture_mode:
        print(f"[INFO] Photo session for {name} finished. Captured {photo_count} photos.")

if __name__ == "__main__":
    person_name = ""
    capture_mode = "MANUAL" 

    # Check for arguments: Expect 2 arguments (name and mode)
    if len(sys.argv) == 3:
        person_name = sys.argv[1].strip()
        capture_mode = sys.argv[2].strip().upper()
    elif len(sys.argv) == 2:
        # If only one argument, assume it's for console fallback testing
        person_name = sys.argv[1].strip()
        if person_name == "PREVIEW":
              capture_mode = "PREVIEW"
    else:
        # Fallback if run directly from console without arguments
        person_name = input("Enter person's name for capture (or type PREVIEW): ").strip()
        if person_name and person_name != "PREVIEW":
            mode_input = input("Enter mode (AUTO/MANUAL): ").strip().upper()
            if mode_input in ["AUTO", "MANUAL"]:
                capture_mode = mode_input
        elif person_name == "PREVIEW":
            capture_mode = "PREVIEW"

    if person_name:
        capture_photos(person_name, capture_mode)
    else:
        print("[ERROR] No name provided. Exiting capture script.")