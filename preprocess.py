import cv2
import numpy as np

def adjust_image(frame, brightness_factor=1.0, contrast_factor=1.0):
    """
    Adjusts the brightness and contrast of an image.
    Brightness factor: 1.0 is unchanged, <1 is darker, >1 is brighter.
    Contrast factor: 1.0 is unchanged, <1 is lower contrast, >1 is higher contrast.
    """
    if brightness_factor == 1.0 and contrast_factor == 1.0:
        return frame
    
    # 1. Adjust Brightness using HSV Value channel
    if brightness_factor != 1.0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(cv2.multiply(np.float32(v), brightness_factor), 0, 255).astype(np.uint8)
        hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(hsv, cv2.HSV2BGR)
        
    # 2. Adjust Contrast using convertScaleAbs
    if contrast_factor != 1.0:
        # alpha is contrast control, beta is brightness offset (0 here)
        frame = cv2.convertScaleAbs(frame, alpha=contrast_factor, beta=0)
        
    return frame

def draw_faces(frame, faces, apply_blur=False):
    """
    Draws bounding boxes and labels for a list of detected faces.
    `faces` is a list of dictionaries from the mtcnn detector.
    If `apply_blur` is True, applies privacy blur to the detected faces.
    """
    # Create a copy so we don't modify the original frame immediately
    annotated_frame = frame.copy()
    
    for idx, face in enumerate(faces):
        x, y, width, height = face['box']
        confidence = face['confidence'] * 100
        
        # Boundary checks to prevent slicing errors
        h_frame, w_frame = annotated_frame.shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_frame, x + width), min(h_frame, y + height)
        
        if apply_blur:
            face_roi = annotated_frame[y1:y2, x1:x2]
            if face_roi.size > 0:
                # Apply heavy Gaussian blur
                blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
                annotated_frame[y1:y2, x1:x2] = blurred_face
        
        # Color and formatting
        color = (50, 255, 50) # Green for bounding box
        thickness = 2
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x, y), (x + width, y + height), color, thickness)
        
        # Text settings
        label = f"Face {idx + 1}"
        text = f"{label} ({confidence:.1f}%)"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        
        # Draw background rectangle for text
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, 1)
        cv2.rectangle(annotated_frame, (x, y - text_height - baseline - 5), (x + text_width, y), color, -1)
        
        # Draw text
        cv2.putText(annotated_frame, text, (x, y - 5), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        
    return annotated_frame
