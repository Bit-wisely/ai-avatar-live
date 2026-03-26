import sys
import io
import yaml
import logging
import threading
import uvicorn
import asyncio
import time
from contextlib import asynccontextmanager

# Force UTF-8 output so print() never crashes on Windows CP1252
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from src.streaming.video_stream import app, state
from src.pipeline.realtime_pipeline import RealTimePipeline

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)


def load_config(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(config):
    """Wait for the uvicorn event loop then start the capture loop."""
    for _ in range(40):
        if state.loop is not None:
            break
        time.sleep(0.25)

    if state.loop is None:
        print("[run_pipeline] ERROR: event loop never became available.")
        return

    pipeline = RealTimePipeline(config)
    pipeline.run()


@asynccontextmanager
async def lifespan(fastapi_app):
    # Startup — capture the running event loop
    state.loop = asyncio.get_running_loop()
    print(f"[Server] Event loop captured: {state.loop}")
    yield
    # Shutdown (nothing to clean up at app level)


# Attach lifespan to the FastAPI app
app.router.lifespan_context = lifespan


def main():
    print("=" * 55)
    print("  3D Portrait Mimic System")
    print("  Open http://localhost:8000 in your browser")
    print("=" * 55)

    config = load_config()

    pipeline_thread = threading.Thread(
        target=run_pipeline,
        args=(config,),
        daemon=True,
        name="PipelineThread"
    )
    pipeline_thread.start()

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
