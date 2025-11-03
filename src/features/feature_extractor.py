import asyncio
import logging

import numpy as np
from sklearn.preprocessing import RobustScaler
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from skimage.feature import hog, local_binary_pattern

    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("scikit-image not available - HOG and LBP features disabled")


def _sanitize_features(X: np.ndarray) -> np.ndarray:
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
    X = np.clip(X, -1e10, 1e10)
    return X


def _normalize_features(X: np.ndarray) -> np.ndarray:
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    return np.clip(X, -10.0, 10.0)


class FeatureExtractor:
    def __init__(self, feature_types: list[str] | None = None) -> None:
        if feature_types is None:
            feature_types = ["hog", "lbp", "raw"]
        self.feature_types = feature_types
        self.scaler = RobustScaler()

    def _extract_hog(self, image: np.ndarray) -> np.ndarray:
        if not SKIMAGE_AVAILABLE:
            return np.array([])
        return hog(
            image,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )

    def _extract_lbp(self, image: np.ndarray) -> np.ndarray:
        if not SKIMAGE_AVAILABLE:
            return np.array([])

        radius = 3
        n_points = 8 * radius

        img_uint8 = (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        lbp = local_binary_pattern(img_uint8, n_points, radius, method="uniform")

        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-7

        return hist

    def _extract_raw(self, image: np.ndarray) -> np.ndarray:
        return image.flatten()

    async def extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image asynchronously."""
        features = []

        # Run CPU-bound operations in thread pool (only for expensive operations)
        if "hog" in self.feature_types:
            hog_feat = await asyncio.to_thread(self._extract_hog, image)
            if len(hog_feat) > 0:
                features.append(hog_feat)

        if "lbp" in self.feature_types:
            lbp_feat = await asyncio.to_thread(self._extract_lbp, image)
            if len(lbp_feat) > 0:
                features.append(lbp_feat)

        if "raw" in self.feature_types:
            # Simple operation, no need for thread pool
            features.append(self._extract_raw(image))

        if not features:
            logger.error("No features extracted")
            return np.array([])

        return np.concatenate(features)

    async def fit_transform(self, images: np.ndarray) -> np.ndarray:
        logger.info("Extracting features from training images...")

        # Process images in parallel
        tasks = [self.extract_features(img) for img in images]
        features_list = []

        with tqdm(total=len(tasks), desc="Processing") as pbar:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Feature extraction failed: {result}")
                    features_list.append(np.array([]))
                else:
                    features_list.append(result)
                pbar.update(1)

        X = np.array(features_list)
        logger.info(f"Extracted features shape: {X.shape}")

        # Simple operations - no need for thread pool
        X = _sanitize_features(X)
        # Scaler operations are CPU-bound, run in thread pool
        X = await asyncio.to_thread(self.scaler.fit_transform, X)
        X = _normalize_features(X)

        return X

    async def transform(
        self, images: np.ndarray, context: str = "images"
    ) -> np.ndarray:
        logger.info(f"Extracting features from {context}...")

        # Process images in parallel
        tasks = [self.extract_features(img) for img in images]
        features_list = []

        with tqdm(total=len(tasks), desc="Processing") as pbar:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Feature extraction failed: {result}")
                    features_list.append(np.array([]))
                else:
                    features_list.append(result)
                pbar.update(1)

        X = np.array(features_list)
        logger.info(f"Extracted features shape: {X.shape}")

        # Simple operations - no need for thread pool
        X = _sanitize_features(X)
        # Scaler operations are CPU-bound, run in thread pool
        X = await asyncio.to_thread(self.scaler.transform, X)
        X = _normalize_features(X)

        return X
