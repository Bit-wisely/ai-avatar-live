import yaml
import logging
from src.pipeline.realtime_pipeline import RealTimePipeline

def load_config(path="config/settings.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("--- Starting AI Avatar Live System ---")
    
    try:
        config = load_config()
        pipeline = RealTimePipeline(config)
        pipeline.run()
    except Exception as e:
        logging.error(f"Failed to start system: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
