import asyncio
import logging
from typing import Tuple

import numpy as np
import pandas as pd

from src.features import FeatureExtractor
from src.models import EmotionClassifier
from src.repository import ImageRepository

logger = logging.getLogger(__name__)


class PredictionService:
    def __init__(self, repo: ImageRepository):
        self.repo = repo

    async def predict(
        self,
        classifier: EmotionClassifier,
        feature_extractor: FeatureExtractor,
        start_id: int = 2701,
        end_id: int = 3000,
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info("[4/5] Predicting test images...")
        test_images, test_ids = await self.repo.load_test_data(start_id, end_id)
        logger.info(f"Loaded {len(test_images)} test images")

        test_features = await feature_extractor.transform(
            test_images, context="test images"
        )
        # Run prediction in thread pool (CPU-bound)
        predictions = await asyncio.to_thread(classifier.predict, test_features)

        return predictions, test_ids

    async def save_submission(
        self,
        predictions: np.ndarray,
        test_ids: np.ndarray,
        output_file: str = "submission.csv",
    ) -> None:
        # Simple DataFrame operations - no need for thread pool
        submission = pd.DataFrame({"id": test_ids, "label": predictions})
        submission = submission.sort_values("id")
        # File I/O - run in thread pool
        await asyncio.to_thread(submission.to_csv, output_file, index=False)

        logger.info(f"Submission file created: {output_file}")
        logger.info(f"First 10 predictions:\n{submission.head(10).to_string()}")

        unique, counts = np.unique(predictions, return_counts=True)
        logger.info("Prediction distribution:")
        for label, count in zip(unique, counts):
            logger.info(f"  {label}: {count}")

        logger.info(f"Finished! Submission saved as {output_file}")
        logger.info("=" * 60)
