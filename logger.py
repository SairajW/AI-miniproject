import csv
import os
from datetime import datetime
import threading

class DetectionLogger:
    def __init__(self, filename="detections_log.csv"):
        self.filename = filename
        self.lock = threading.Lock()
        self._initialize_csv()

    def _initialize_csv(self):
        """Creates the CSV file with headers if it doesn't exist."""
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Timestamp", "Faces_Detected"])

    def log_detection(self, num_faces):
        """Logs the detection event asynchronously."""
        # Only log if there are faces to avoid spamming the log with 0s
        # or log everything if desired. Here we log when num_faces > 0.
        if num_faces > 0:
            thread = threading.Thread(target=self._write_log, args=(num_faces,), daemon=True)
            thread.start()

    def _write_log(self, num_faces):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            with open(self.filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([timestamp, num_faces])
