import argparse
import asyncio
import logging
import sys
from pathlib import Path

from src.config import Settings
from src.repository import ImageRepository
from src.services import PredictionService, TrainingService
from src.storage import ModelStorage
from utils.logger import setup_logger

logger = setup_logger(name="aspicv", log_level=logging.INFO, log_file="logs/training.log")


async def main(use_cache=True):
    repo = ImageRepository(Settings.DATASET_PATH)
    storage = ModelStorage(Settings.STORAGE_DIR)
    training_service = TrainingService(repo, storage)
    prediction_service = PredictionService(repo)
    
    try:
        classifier, feature_extractor = await training_service.train(
            Settings.TRAIN_CSV,
            use_cache=use_cache
        )
        
        predictions, test_ids = await prediction_service.predict(
            classifier,
            feature_extractor,
            Settings.TEST_START_ID,
            Settings.TEST_END_ID
        )
        
        await prediction_service.save_submission(predictions, test_ids, Settings.OUTPUT_FILE)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Facial Expression Recognition")
    parser.add_argument("--no-cache", action="store_true", help="Disable model caching")
    args = parser.parse_args()
    asyncio.run(main(use_cache=not args.no_cache))

