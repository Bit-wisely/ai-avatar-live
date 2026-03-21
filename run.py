import yaml
import logging
import threading
import uvicorn
import asyncio
import time
from src.pipeline.realtime_pipeline import RealTimePipeline
from src.streaming.video_stream import app, state

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

def load_config(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def capture_loop(loop: asyncio.AbstractEventLoop):
    """Share the uvicorn event loop with the rest of the app."""
    state.loop = loop


async def _wait_for_loop():
    """Called inside uvicorn's event loop to report it back to state."""
    state.loop = asyncio.get_running_loop()


@app.on_event("startup")
async def on_startup():
    state.loop = asyncio.get_running_loop()
    print(f"[Server] Event loop captured: {state.loop}")


def run_pipeline(config):
    # Wait a moment for uvicorn to set state.loop
    for _ in range(30):
        if state.loop is not None:
            break
        time.sleep(0.2)

    pipeline = RealTimePipeline(config)
    pipeline.run()


def main():
    print("=" * 55)
    print("  3D Portrait Mimic System")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 55)

    config = load_config()

    # Start capture pipeline in a background thread
    pipeline_thread = threading.Thread(
        target=run_pipeline,
        args=(config,),
        daemon=True
    )
    pipeline_thread.start()

    # Run FastAPI / uvicorn in the main thread
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
