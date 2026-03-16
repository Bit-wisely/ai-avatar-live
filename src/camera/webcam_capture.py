import cv2
import logging

class WebcamCapture:
    def __init__(self, device_id=0, width=640, height=480):
        self.cap = cv2.VideoCapture(device_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        if not self.cap.isOpened():
            raise Exception("Could not open webcam.")
            
        logging.info(f"Webcam initialized: {width}x{height}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Flip frame for mirror effect
        return cv2.flip(frame, 1)

    def release(self):
        self.cap.release()
