import cv2
import threading

class VideoHandler:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.current_frame = None
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        """Starts the webcam capture stream."""
        if not self.is_running:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW) # Fast init on Windows
            if not self.cap.isOpened():
                print("Error: Could not open webcam.")
                return False
                
            self.is_running = True
            
            # Start a background thread to read frames continuously
            self.thread = threading.Thread(target=self._update, daemon=True)
            self.thread.start()
            return True
        return True

    def _update(self):
        """Continuous frame reading loop."""
        while self.is_running:
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.current_frame = frame

    def get_frame(self):
        """Returns the most recent frame."""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def stop(self):
        """Stops the capture stream."""
        self.is_running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
    def is_active(self):
        return self.is_running
