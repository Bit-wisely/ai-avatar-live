import sounddevice as sd
import numpy as np

class MicInput:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.stream = None

    def start_stream(self, callback):
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=callback,
            blocksize=self.chunk_size
        )
        self.stream.start()

    def stop_stream(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
