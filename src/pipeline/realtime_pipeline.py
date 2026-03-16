import threading
import queue
import time
import cv2
from src.camera.webcam_capture import WebcamCapture
from src.tracking.face_landmarks import FaceTracker
from src.animation.avatar_renderer import AvatarRenderer
from src.audio.mic_input import MicInput
from src.audio.voice_conversion import VoiceConverter

class RealTimePipeline:
    def __init__(self, config):
        self.config = config
        self.running = False
        
        # Queues for inter-thread communication
        self.frame_queue = queue.Queue(maxsize=config['pipeline']['max_queue_size'])
        self.landmark_queue = queue.Queue(maxsize=config['pipeline']['max_queue_size'])
        self.audio_queue = queue.Queue(maxsize=10)
        self.render_queue = queue.Queue(maxsize=config['pipeline']['max_queue_size'])

        # Initialize modules
        self.camera = WebcamCapture(
            config['camera']['device_id'],
            config['camera']['width'],
            config['camera']['height']
        )
        self.tracker = FaceTracker()
        self.renderer = AvatarRenderer(
            config['animation']['avatar_path'],
            config['animation']['model_path']
        )
        self.mic = MicInput(config['audio']['sample_rate'], config['audio']['chunk_size'])
        self.converter = VoiceConverter(
            config['audio']['rvc_model'],
            config['audio']['index_path']
        )

    def _capture_thread(self):
        while self.running:
            frame = self.camera.get_frame()
            if frame is not None:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
            time.sleep(1/self.config['camera']['fps'])

    def _tracking_thread(self):
        while self.running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                landmarks = self.tracker.process(frame)
                if not self.landmark_queue.full():
                    self.landmark_queue.put(landmarks)

    def _rendering_thread(self):
        while self.running:
            if not self.landmark_queue.empty():
                landmarks = self.landmark_queue.get()
                rendered_frame = self.renderer.render(landmarks)
                if not self.render_queue.full():
                    self.render_queue.put(rendered_frame)

    def _audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        converted_audio = self.converter.convert(indata.copy())
        # In a real app, this would be piped to a virtual audio cable or speaker
        self.audio_queue.put(converted_audio)

    def run(self):
        self.running = True
        
        # Start processing threads
        threads = [
            threading.Thread(target=self._capture_thread, daemon=True),
            threading.Thread(target=self._tracking_thread, daemon=True),
            threading.Thread(target=self._rendering_thread, daemon=True)
        ]
        
        for t in threads:
            t.start()
            
        self.mic.start_stream(self._audio_callback)

        print("Pipeline running. Press 'q' to quit.")
        
        try:
            while self.running:
                if not self.render_queue.empty():
                    frame = self.render_queue.get()
                    cv2.imshow("AI Avatar Live", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    time.sleep(0.01)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.camera.release()
        self.mic.stop_stream()
        cv2.destroyAllWindows()
