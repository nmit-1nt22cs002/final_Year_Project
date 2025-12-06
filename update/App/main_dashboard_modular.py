import customtkinter as ctk
from PIL import Image
import os
import subprocess
import tkinter.messagebox as messagebox
from datetime import datetime
import sys 
import pandas as pd
import tkinter as tk # Required for the full log pop-up window

# --- CONFIGURATION ---
THEME_COLOR = "dark-blue"
LOG_FILE_PATH = "attendance_log_temp.csv" # Shared log file for manual export
YELLOWISH_WHITE_COLOR = "#FFFFE0" # Subtle Light Yellow for the custom "white" background
DEFAULT_DARK_BG = "#242424" # Standard CTk dark theme background color

# --- UI SETUP ---
ctk.set_appearance_mode("Dark") # Start in Dark mode
ctk.set_default_color_theme(THEME_COLOR)

class FaceRecognitionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Smart Face Recognition & Modular System")
        self.geometry("1200x720")
        
        # State variables
        self.video_running = False
        self.recognition_process = None
        self.full_log_text = "" # Stores the complete, plain text log

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Create a black image placeholder
        self.black_img = Image.new('RGB', (800, 450), 'black')
        self.black_ctk_img = ctk.CTkImage(light_image=self.black_img, dark_image=self.black_img, size=(800, 450))

        # --- LEFT SIDEBAR (Column 0) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure((0, 10), weight=1) 

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="FACEX Modular", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        # Action Buttons (Calling External Scripts/EXEs)
        self._create_sidebar_button("Capture New Faces", 1, self.prompt_for_enrollment_details, icon="camera") 
        self._create_sidebar_button("Run Encoder (Update)", 2, self.run_encoder, icon="reload")
        self._create_sidebar_button("Run Facial Recognition", 3, self.run_recognition_script, fg_color="#2CC985", icon="scan")
        
        # Camera Control & Export
        self._create_sidebar_button("Start Camera (Preview)", 5, self.start_camera_preview, icon="play")
        self._create_sidebar_button("Stop All Processes", 6, self.stop_all_processes, fg_color="#C92C2C", icon="stop")
        self._create_sidebar_button("Export Data", 7, self.export_to_excel, icon="export") 

        # --- THEME SWITCHER ---
        self.theme_label = ctk.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.theme_label.grid(row=8, column=0, padx=20, pady=(10, 0), sticky="s")
        self.theme_optionmenu = ctk.CTkOptionMenu(self.sidebar_frame, 
                                                 values=["Dark", "Light"],
                                                 command=self.change_appearance_mode_event)
        self.theme_optionmenu.grid(row=9, column=0, padx=20, pady=(0, 10), sticky="s")
        self.theme_optionmenu.set(ctk.get_appearance_mode())

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Status: Ready", text_color="gray")
        self.status_label.grid(row=10, column=0, padx=20, pady=20, sticky="s")

        # --- MAIN VIDEO AREA (Column 1) ---
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        
        self.video_label = ctk.CTkLabel(self.video_frame, text="Camera Feed Off", 
                                        image=self.black_ctk_img, compound="center")
        self.video_label.pack(expand=True, fill="both", padx=10, pady=10)

        # --- EVENT LOG (Column 2) ---
        # Log frame initialization (default vertical scroll is fine, horizontal is disabled)
        self.log_frame = ctk.CTkScrollableFrame(self, width=250) 
        self.log_frame.grid(row=0, column=2, padx=(0, 20), pady=20, sticky="nsew")
        
        # Create a separate title label and bind the click event
        self.log_title_label = ctk.CTkLabel(self.log_frame, text="Event Log (Click to Enlarge)", 
                                            font=ctk.CTkFont(weight="bold"))
        self.log_title_label.pack(pady=(10, 5))
        self.log_title_label.bind("<Button-1>", self.open_full_log_window)

        # Add a container frame inside the scrollable frame
        self.log_container = ctk.CTkFrame(self.log_frame, fg_color="transparent")
        self.log_container.pack(fill="x", anchor="nw") 

        self.log_message("System Initialized.")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _create_sidebar_button(self, text, row, command, fg_color=None, icon=None):
        btn = ctk.CTkButton(self.sidebar_frame, text=text, command=command, fg_color=fg_color)
        btn.grid(row=row, column=0, padx=20, pady=10)
        return btn

    def log_message(self, message, color="white"):
        now = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{now}] {message}"
        
        # 1. Update the internal log storage
        self.full_log_text += full_message + "\n"

        # 2. Create and pack the visible log entry
        # FIX: Added wraplength to force text to wrap within the available space (250px)
        log_entry = ctk.CTkLabel(self.log_container, text=full_message, font=ctk.CTkFont(size=12), 
                                 text_color=color, anchor="w", justify="left", wraplength=250)
        
        # fill="x" ensures the log entry spans the container width (for consistent background)
        log_entry.pack(anchor="w", padx=5, pady=2, fill="x") 
        
        # Scroll to the bottom to see the newest message
        self.log_frame._parent_canvas.yview_moveto(1.0)
        
        self.status_label.configure(text=f"Status: {message.split('.')[0]}") 
        
    def open_full_log_window(self, event):
        """Opens a new window showing the full text log."""
        log_window = ctk.CTkToplevel(self)
        log_window.title("Full System Event Log")
        log_window.geometry("600x400")
        log_window.grab_set() 
        log_window.transient(self) # Keep it on top of the main window

        # Use standard tkinter Text widget 
        log_text_widget = tk.Text(log_window, wrap="word", padx=10, pady=10,
                                  bg=self._apply_appearance_mode(ctk.ThemeManager.theme["CTkFrame"]["fg_color"]),
                                  fg=self._apply_appearance_mode(ctk.ThemeManager.theme["CTkLabel"]["text_color"]),
                                  font=("CTkDefaultFont", 12))
        
        log_text_widget.insert(tk.END, self.full_log_text)
        log_text_widget.configure(state="disabled") # Make it read-only
        log_text_widget.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Scroll to bottom immediately
        log_text_widget.see(tk.END)


    # --- Appearance Mode Handler ---
    def change_appearance_mode_event(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)
        
        if new_appearance_mode == "Light":
            self.configure(fg_color=YELLOWISH_WHITE_COLOR) 
        else:
            self.configure(fg_color=DEFAULT_DARK_BG)

    # --- External Script Execution Helper (MODIFIED FOR EXE) ---
    def run_external_script(self, script_name, args=None, is_camera_script=False):
        """Runs an external script using its new EXE name."""
        self.stop_all_processes() 
        
        # 1. Map the original script name to the new EXE file name
        executable_name = ""
        if script_name == "encoder.py":
            executable_name = "encoder_process.exe"
        elif script_name == "image_capture_windows.py":
            executable_name = "capture_process.exe"
        elif script_name == "run_recognition.py":
            executable_name = "recognition_process.exe"
        else:
            self.log_message(f"Error: Unknown script name {script_name}", color="red")
            return

        # 2. Build the command list, starting with the EXE name
        command = [executable_name]
        if args:
            command.extend(args)
            
        try:
            if is_camera_script:
                # Launch the standalone EXE for camera/recognition in a new console
                process = subprocess.Popen(command, creationflags=subprocess.CREATE_NEW_CONSOLE)
                self.recognition_process = process 
                self.log_message(f"Started {executable_name} with args {args if args else ''} in new window.")
                self.video_running = True
                self.video_label.configure(image=None, text="Running in External Window...")
                return process
            else:
                # Run the standalone EXE for encoder (synchronous)
                self.log_message(f"Running {executable_name}...")
                subprocess.run(command, check=True)
                self.log_message(f"{executable_name} completed successfully.", color="green")
        except subprocess.CalledProcessError as e:
            self.log_message(f"Error running {executable_name}: {e}", color="red")
        except FileNotFoundError:
            # Crucial message when the helper EXE is missing
            self.log_message(f"Error: Executable {executable_name} not found. Ensure it is in the same folder as the dashboard EXE.", color="red")
        
        return None

    # --- Enrollment and Capture Logic ---
    def prompt_for_enrollment_details(self):
        self.stop_all_processes()
        
        # 1. Prompt for Name
        name_dialog = ctk.CTkInputDialog(text="Enter the name of the person to capture photos for:", title="New Person Capture")
        person_name = name_dialog.get_input()

        if not person_name or not person_name.strip():
            self.log_message("Capture cancelled: No name provided.")
            return

        person_name = person_name.strip()
        
        # 2. Prompt for Capture Mode
        mode_window = ctk.CTkToplevel(self)
        mode_window.title("Capture Mode")
        mode_window.geometry("300x150")
        mode_window.resizable(False, False)
        mode_window.grab_set() 

        mode_label = ctk.CTkLabel(mode_window, text=f"Select capture mode for {person_name}:")
        mode_label.pack(pady=10, padx=20)
        
        def start_capture_with_mode(mode):
            mode_window.destroy()
            # Calls the capture_process.exe
            self.run_external_script("image_capture_windows.py", args=[person_name, mode], is_camera_script=True)
            
        btn_manual = ctk.CTkButton(mode_window, text="MANUAL (Press SPACE)", command=lambda: start_capture_with_mode("MANUAL"))
        btn_manual.pack(pady=5, padx=20)
        
        btn_auto = ctk.CTkButton(mode_window, text="AUTO (150-200 Images)", command=lambda: start_capture_with_mode("AUTO"), fg_color="#2CC985")
        btn_auto.pack(pady=5, padx=20)
        
        mode_window.wait_window() 
    
    # --- Other External Function Mappings ---

    def run_encoder(self):
        # Calls the encoder_process.exe
        self.run_external_script("encoder.py")
    
    def run_recognition_script(self):
        # Calls the recognition_process.exe
        self.run_external_script("run_recognition.py", is_camera_script=True)

    def start_camera_preview(self):
        # Calls the capture_process.exe in preview mode
        self.run_external_script("image_capture_windows.py", args=["PREVIEW", "PREVIEW"], is_camera_script=True)


    def stop_all_processes(self):
        if self.recognition_process:
            self.recognition_process.terminate()
            self.recognition_process = None
            self.video_running = False
            self.video_label.configure(image=self.black_ctk_img, text="Camera Feed Off")
            self.log_message("All camera processes stopped.", color="red")
        else:
            self.log_message("No camera process was running.")


    # --- Export to Excel Function ---
    def export_to_excel(self):
        if not os.path.exists(LOG_FILE_PATH):
            messagebox.showinfo("Export Error", "No attendance log file found. Run Facial Recognition first.")
            self.log_message("Export failed: No log file found.", color="orange")
            return
        
        try:
            # Read the CSV file saved by run_recognition.py
            df = pd.read_csv(LOG_FILE_PATH)
            
            if df.empty:
                messagebox.showinfo("Export Complete", "Log file was empty. No data to save.")
                os.remove(LOG_FILE_PATH) # Clean up empty file
                return
            
            # Generate unique Excel filename
            filename = f"Attendance_Export_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.xlsx"
            
            # Save to Excel
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Attendance Log')
            
            # Clean up the temporary CSV log file after successful export
            os.remove(LOG_FILE_PATH) 

            messagebox.showinfo("Export Complete", f"Attendance data successfully saved to {filename}")
            self.log_message(f"Exported data to {filename}. Log cleared.", color="green")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred during export: {e}")
            self.log_message(f"Export failed: {e}", color="red")


    def on_closing(self):
        self.stop_all_processes()
        self.destroy()

if __name__ == "__main__":
    app = FaceRecognitionApp()
    app.mainloop()