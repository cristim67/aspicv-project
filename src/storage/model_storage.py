import asyncio
import logging
from pathlib import Path

import joblib

logger = logging.getLogger(__name__)


class ModelStorage:
    def __init__(self, storage_dir="models"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def save_feature_extractor(self, extractor, name="feature_extractor"):
        return await self._save(extractor, name)
    
    async def load_feature_extractor(self, name="feature_extractor"):
        return await self._load(name)
    
    async def save_classifier(self, classifier, name="classifier"):
        return await self._save(classifier, name)
    
    async def load_classifier(self, name="classifier"):
        return await self._load(name)
    
    def exists(self, name="classifier"):
        return (self.storage_dir / f"{name}.joblib").exists()
    
    async def _save(self, obj, name):
        filepath = self.storage_dir / f"{name}.joblib"
        try:
            await asyncio.to_thread(joblib.dump, obj, filepath)
            logger.info(f"Saved {name} to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save {name}: {e}", exc_info=True)
            return False
    
    async def _load(self, name):
        filepath = self.storage_dir / f"{name}.joblib"
        absolute_path = filepath.resolve()
        
        if not filepath.exists():
            logger.warning(f"{name} not found: {absolute_path}")
            return None
        
        try:
            obj = await asyncio.to_thread(joblib.load, filepath)
            logger.info(f"Loaded {name} from {absolute_path}")
            return obj
        except Exception as e:
            logger.error(f"Failed to load {name} from {absolute_path}: {e}", exc_info=True)
            return None

