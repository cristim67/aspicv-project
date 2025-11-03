import asyncio
import logging
from pathlib import Path
from typing import cast

import joblib

from src.features import FeatureExtractor
from src.models import EmotionClassifier

logger = logging.getLogger(__name__)


class ModelStorage:
    def __init__(self, storage_dir: str = "models") -> None:
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def save_feature_extractor(
        self, extractor: FeatureExtractor, name: str = "feature_extractor"
    ) -> bool:
        return await self._save(extractor, name)

    async def load_feature_extractor(
        self, name: str = "feature_extractor"
    ) -> FeatureExtractor | None:
        result = await self._load(name)
        return result if isinstance(result, FeatureExtractor) else None

    async def save_classifier(
        self, classifier: EmotionClassifier, name: str = "classifier"
    ) -> bool:
        return await self._save(classifier, name)

    async def load_classifier(
        self, name: str = "classifier"
    ) -> EmotionClassifier | None:
        result = await self._load(name)
        return result if isinstance(result, EmotionClassifier) else None

    def exists(self, name: str = "classifier") -> bool:
        return (self.storage_dir / f"{name}.joblib").exists()

    async def _save(self, obj: FeatureExtractor | EmotionClassifier, name: str) -> bool:
        filepath = self.storage_dir / f"{name}.joblib"
        try:
            await asyncio.to_thread(joblib.dump, obj, filepath)
            logger.info(f"Saved {name} to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {name}: {e}", exc_info=True)
            return False

    async def _load(self, name: str) -> FeatureExtractor | EmotionClassifier | None:
        filepath = self.storage_dir / f"{name}.joblib"
        absolute_path = filepath.resolve()

        if not filepath.exists():
            logger.warning(f"{name} not found: {absolute_path}")
            return None

        try:
            obj = await asyncio.to_thread(joblib.load, filepath)
            logger.info(f"Loaded {name} from {absolute_path}")
            return cast(FeatureExtractor | EmotionClassifier, obj)
        except Exception as e:
            logger.error(
                f"Failed to load {name} from {absolute_path}: {e}", exc_info=True
            )
            return None
