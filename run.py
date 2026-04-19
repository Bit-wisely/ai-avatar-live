import sys
import yaml
import uvicorn
import threading
from src.streaming.video_stream import app, state
from src.pipeline.realtime_pipeline import RealTimePipeline

def load_config():
    try:
        with open("config/settings.yaml", "r") as f:
            return yaml.safe_load(f)
    except Exception:
        return {"camera": {"device_id": 0, "width": 640, "height": 480, "fps": 30}}

def main():
    print("--- 3D Portrait Mimic System ---")
    print("Serving at http://localhost:8000")
    
    config = load_config()
    pipeline = RealTimePipeline(config)
    
    # Start the processing pipeline in a background thread
    threading.Thread(target=pipeline.run, daemon=True).start()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

if __name__ == "__main__":
    main()
