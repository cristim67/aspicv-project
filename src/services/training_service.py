import asyncio
import logging

import numpy as np

from src.features import FeatureExtractor
from src.models import EmotionClassifier
from src.repository import ImageRepository
from src.storage import ModelStorage

logger = logging.getLogger(__name__)


class TrainingService:
    def __init__(self, repo: ImageRepository, storage: ModelStorage):
        self.repo = repo
        self.storage = storage

    async def train(
        self, train_csv: str, use_cache: bool = True
    ) -> tuple[EmotionClassifier, FeatureExtractor]:
        logger.info("=" * 60)
        logger.info("Facial Expression Recognition System")
        logger.info(f"Cache: {'ON' if use_cache else 'OFF'}")
        logger.info("=" * 60)

        feature_extractor = (
            await self.storage.load_feature_extractor() if use_cache else None
        )

        logger.info("[1/5] Loading training data...")
        images, labels, train_ids = await self.repo.load_training_data(train_csv)
        logger.info(f"Loaded {len(images)} images")

        unique, counts = np.unique(labels, return_counts=True)
        logger.info("Label distribution:")
        for label, count in zip(unique, counts):
            logger.info(f"  {label}: {count}")

        logger.info("[2/5] Extracting features...")
        if feature_extractor is None:
            feature_extractor = FeatureExtractor(feature_types=["hog", "lbp", "raw"])
            features = await feature_extractor.fit_transform(images)
            if use_cache:
                await self.storage.save_feature_extractor(feature_extractor)
        else:
            features = await feature_extractor.transform(
                images, context="training images"
            )

        logger.info(f"Feature vector size: {features.shape[1]}")

        logger.info("[3/5] Training classifier...")
        classifier = await self.storage.load_classifier() if use_cache else None

        if classifier is None:
            classifier = EmotionClassifier(use_ensemble=True)
            # Run training in thread pool (CPU-bound)
            await asyncio.to_thread(classifier.fit, features, labels)
            if use_cache:
                await self.storage.save_classifier(classifier)
        else:
            logger.info("Using cached classifier")

        return classifier, feature_extractor
