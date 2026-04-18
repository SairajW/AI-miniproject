import cv2
import os

# Suppress noisy TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from mtcnn import MTCNN

class FaceDetector:
    def __init__(self):
        """
        Initializes the MTCNN (Multi-task Cascaded Convolutional Networks)
        model from TensorFlow/Keras backend.
        """
        self.detector = MTCNN()

    def detect(self, frame, resize_factor=0.5):
        """
        Detects faces in a given BGR frame.
        We can optionally resize the frame for faster inference, then map boxes back.
        Returns a list of dicts: {'box': [x,y,w,h], 'confidence': float, ...}
        """
        if frame is None or frame.size == 0:
            return []

        try:
            # Resize for faster performance in real time
            small_frame = cv2.resize(frame, (0, 0), fx=resize_factor, fy=resize_factor)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            results = self.detector.detect_faces(rgb_frame)
            
            # Map bounding boxes back to the original frame size
            for res in results:
                box = res['box']
                res['box'] = [
                    int(box[0] / resize_factor),
                    int(box[1] / resize_factor),
                    int(box[2] / resize_factor),
                    int(box[3] / resize_factor)
                ]
                
            return results
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
