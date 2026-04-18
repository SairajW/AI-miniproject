import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import threading
import time
import os
from datetime import datetime

from src.video_handler import VideoHandler
from src.face_detector import FaceDetector
from src.preprocess import adjust_image, draw_faces
from src.logger import DetectionLogger

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Detection System")
        self.root.geometry("1100x700")
        self.root.configure(bg="#0d1117")
        
        # Core components
        self.video_handler = VideoHandler(camera_index=0)
        self.detector = FaceDetector()
        self.logger = DetectionLogger()
        self.last_log_time = time.time()
        
        self.is_paused = False
        self.last_frame = None
        
        # Recording state
        self.is_recording = False
        self.video_writer = None
        self.video_writer_filename = ""
        
        # Threading states for inference
        self.current_faces = []
        self.inference_running = False
        self.inference_thread = None
        self.frame_to_infer = None
        self.inference_lock = threading.Lock()
        
        self._setup_gui()
        self._update_feed()

    def _setup_gui(self):
        # Main layout structure: Side by side
        main_frame = tk.Frame(self.root, bg="#0d1117")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ------------------ LEFT PANEL (FEED) ------------------
        left_frame = tk.Frame(main_frame, bg="#0d1117")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        tk.Label(left_frame, text="Live Feed", font=("Helvetica", 14, "bold"), fg="white", bg="#0d1117").pack(pady=(0, 10))
        
        self.canvas_width = 720
        self.canvas_height = 540
        self.video_frame = tk.Frame(left_frame, width=self.canvas_width, height=self.canvas_height, bg="black", highlightbackground="#00ffff", highlightthickness=2)
        self.video_frame.pack(expand=True)
        # Ensure frame doesn't shrink
        self.video_frame.pack_propagate(False)
        
        self.video_label = tk.Label(self.video_frame, bg="black")
        self.video_label.place(relheight=1, relwidth=1)

        # ------------------ RIGHT PANEL (SIDEBAR) ------------------
        right_frame = tk.Frame(main_frame, bg="#0d1117", width=300)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 20), pady=20)
        right_frame.pack_propagate(False)

        # Section 1: Controls
        tk.Label(right_frame, text="Controls", font=("Helvetica", 12, "bold"), fg="white", bg="#0d1117").pack(pady=(0, 10))
        
        controls_panel = tk.Frame(right_frame, bg="#1a1e2e", padx=10, pady=10)
        controls_panel.pack(fill=tk.X, pady=(0, 20))
        
        btn_font = ("Helvetica", 10, "bold")
        self.btn_start = tk.Button(controls_panel, text="▶ START", bg="#00fb8a", fg="black", font=btn_font, bd=0, height=2, command=self.start_video)
        self.btn_start.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_pause = tk.Button(controls_panel, text="⏸ PAUSE", bg="#ffab00", fg="black", font=btn_font, bd=0, height=2, command=self.pause_video)
        self.btn_pause.pack(fill=tk.X, pady=(0, 10))
        
        self.btn_stop = tk.Button(controls_panel, text="⏹ STOP", bg="#fc4a4a", fg="black", font=btn_font, bd=0, height=2, command=self.stop_video)
        self.btn_stop.pack(fill=tk.X)

        # Section 2: Status
        tk.Label(right_frame, text="Status", font=("Helvetica", 12, "bold"), fg="white", bg="#0d1117").pack(pady=(0, 10))
        
        status_panel = tk.Frame(right_frame, bg="#1a1e2e", padx=15, pady=15)
        status_panel.pack(fill=tk.X, pady=(0, 20))
        
        status_stack = tk.Frame(status_panel, bg="#1a1e2e")
        status_stack.pack(anchor=tk.W)
        tk.Label(status_stack, text="State:", font=("Helvetica", 10), fg="#8b949e", bg="#1a1e2e").pack(anchor=tk.W)
        self.lbl_state = tk.Label(status_stack, text="● STOPPED", font=("Helvetica", 11, "bold"), fg="#fc4a4a", bg="#1a1e2e")
        self.lbl_state.pack(anchor=tk.W, pady=(5, 0))
        
        self.lbl_face_count = tk.Label(status_stack, text="Faces Detected: 0", font=("Helvetica", 10), fg="#8b949e", bg="#1a1e2e")
        self.lbl_face_count.pack(anchor=tk.W, pady=(5, 0))
        
        # Section 3: Image Enhancement
        tk.Label(right_frame, text="Image Enhancement", font=("Helvetica", 12, "bold"), fg="white", bg="#0d1117").pack(pady=(0, 10))
        enhance_panel = tk.Frame(right_frame, bg="#1a1e2e", padx=15, pady=15)
        enhance_panel.pack(fill=tk.X, pady=(0, 20))
        
        self.brightness_var = tk.DoubleVar(value=1.0)
        tk.Label(enhance_panel, text="Brightness:", font=("Helvetica", 9), fg="#8b949e", bg="#1a1e2e").pack(anchor=tk.W)
        self.slider_bright = ttk.Scale(enhance_panel, from_=0.1, to=2.5, orient=tk.HORIZONTAL, variable=self.brightness_var)
        self.slider_bright.pack(fill=tk.X, pady=(5, 15))
        
        self.contrast_var = tk.DoubleVar(value=1.0)
        tk.Label(enhance_panel, text="Contrast:", font=("Helvetica", 9), fg="#8b949e", bg="#1a1e2e").pack(anchor=tk.W)
        self.slider_contrast = ttk.Scale(enhance_panel, from_=0.1, to=2.5, orient=tk.HORIZONTAL, variable=self.contrast_var)
        self.slider_contrast.pack(fill=tk.X, pady=(5, 0))

        # Section 4: Extra Modules
        tk.Label(right_frame, text="Extra Modules", font=("Helvetica", 12, "bold"), fg="white", bg="#0d1117").pack(pady=(0, 10))
        extras_panel = tk.Frame(right_frame, bg="#1a1e2e", padx=10, pady=10)
        extras_panel.pack(fill=tk.X)
        
        extra_btn_font = ("Helvetica", 9, "bold")
        self.btn_snap = tk.Button(extras_panel, text="Snapshot", bg="#34495e", fg="white", bd=0, height=1, font=extra_btn_font, command=self.take_snapshot)
        self.btn_snap.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 5))
        
        self.btn_record = tk.Button(extras_panel, text="Record", bg="#d35400", fg="white", bd=0, height=1, font=extra_btn_font, command=self.toggle_record)
        self.btn_record.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(5, 0))
        
        self.privacy_var = tk.BooleanVar(value=False)
        self.chk_privacy = tk.Checkbutton(extras_panel, text="Privacy Mode", variable=self.privacy_var, bg="#1a1e2e", fg="white", selectcolor="#2c3e50", font=("Helvetica", 9), activebackground="#1a1e2e", activeforeground="white")
        self.chk_privacy.pack(side=tk.BOTTOM, pady=(15, 0))

    def update_state_label(self, state):
        if state == "RUNNING":
            self.lbl_state.config(text="● RUNNING", fg="#00fb8a")
        elif state == "PAUSED":
            self.lbl_state.config(text="● PAUSED", fg="#ffab00")
        elif state == "STOPPED":
            self.lbl_state.config(text="● STOPPED", fg="#fc4a4a")

    def start_video(self):
        if not self.video_handler.is_active():
            self.video_handler.start()
            self.start_inference_thread()
        self.is_paused = False
        self.update_state_label("RUNNING")

    def pause_video(self):
        self.is_paused = not self.is_paused
        self.update_state_label("PAUSED" if self.is_paused else "RUNNING")
        
    def stop_video(self):
        self.video_handler.stop()
        self.stop_inference_thread()
        if self.is_recording:
            self.toggle_record()
        self.is_paused = False
        self.update_state_label("STOPPED")
        self.video_label.configure(image='')
        self.last_frame = None
        self.lbl_face_count.config(text="Faces Detected: 0", fg="#8b949e")
        
    def start_inference_thread(self):
        if not self.inference_running:
            self.inference_running = True
            self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self.inference_thread.start()

    def stop_inference_thread(self):
        self.inference_running = False
        if self.inference_thread is not None:
            self.inference_thread.join(timeout=1.0)

    def _inference_loop(self):
        while self.inference_running:
            frame_copy = None
            with self.inference_lock:
                if self.frame_to_infer is not None:
                    frame_copy = self.frame_to_infer.copy()
            
            if frame_copy is not None:
                faces = self.detector.detect(frame_copy, resize_factor=0.5)
                self.current_faces = faces
                
                # By clearing the frame_to_infer, we wait for a fresh frame
                with self.inference_lock:
                    self.frame_to_infer = None
            else:
                time.sleep(0.01)

    def take_snapshot(self):
        if self.last_frame is not None:
            if not os.path.exists("snapshots"):
                os.makedirs("snapshots")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/face_snap_{timestamp}.jpg"
            cv2.imwrite(filename, self.last_frame)
            print(f"Snapshot saved: {filename}")

    def toggle_record(self):
        if not self.is_recording:
            if not os.path.exists("recordings"):
                os.makedirs("recordings")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.video_writer_filename = f"recordings/video_{timestamp}.avi"
            self.is_recording = True
            self.btn_record.config(text="Stop Rec", bg="#e74c3c")
            print(f"Started recording to {self.video_writer_filename}")
        else:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.btn_record.config(text="Record", bg="#d35400")
            print("Stopped recording.")

    def _update_feed(self):
        # Only process if not paused and camera is active
        if self.video_handler.is_active() and not self.is_paused:
            frame = self.video_handler.get_frame()
            if frame is not None:
                # Update the frame for the inference thread
                with self.inference_lock:
                    self.frame_to_infer = frame

                # 1. Adjust Brightness and Contrast
                brightness_factor = self.brightness_var.get()
                contrast_factor = self.contrast_var.get()
                frame = adjust_image(frame, brightness_factor, contrast_factor)
                
                # 2. Get the latest asynchronously calculated bounding boxes
                faces = self.current_faces
                
                # Update Face Count Label
                num_faces = len(faces)
                color = "#00fb8a" if num_faces > 0 else "#8b949e"
                self.lbl_face_count.config(text=f"Faces Detected: {num_faces}", fg=color)
                
                # 3. Draw Bounding Boxes with Privacy Option
                frame = draw_faces(frame, faces, apply_blur=self.privacy_var.get())
                
                # Store the most recently drawn frame in BGR for snapshots
                self.last_frame = frame
                
                # 4. Handle Video Recording
                if self.is_recording:
                    if self.video_writer is None:
                        h, w = frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        # Using 20.0 fps as a baseline, but actual framerate might vary
                        self.video_writer = cv2.VideoWriter(self.video_writer_filename, fourcc, 20.0, (w, h))
                    self.video_writer.write(frame)
                    
                # 5. Handle Logging
                if time.time() - self.last_log_time > 2.0:
                    self.logger.log_detection(num_faces)
                    self.last_log_time = time.time()
                
                # Convert frame for Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                
                # Resize if needed to fit layout but preserve aspect ratio
                img.thumbnail((self.canvas_width, self.canvas_height), Image.Resampling.LANCZOS)
                
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.configure(image=imgtk)
                
        # Call this function again after ~30ms (approx 30fps update rate)
        self.root.after(30, self._update_feed)
        
    def on_closing(self):
        self.stop_video()
        self.root.destroy()
