import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, cast

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ImageRepository:
    IMAGE_SIZE = (48, 48)

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    async def _load_image(self, img_path: str) -> Optional[np.ndarray]:
        """Load and preprocess a single image asynchronously."""
        if not Path(img_path).exists():
            return None

        # Run blocking I/O in thread pool
        img = await asyncio.to_thread(cv2.imread, img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None

        if img.shape != self.IMAGE_SIZE:
            img = await asyncio.to_thread(cv2.resize, img, self.IMAGE_SIZE)

        return img.astype(np.float32) / 255.0

    async def _load_image_with_info(
        self, img_id: int, label: Optional[str] = None
    ) -> Tuple[int, Optional[np.ndarray], Optional[str]]:
        """Load image with ID and optional label."""
        img_path = os.path.join(self.dataset_path, f"{img_id}.jpg")
        img = await self._load_image(img_path)
        return img_id, img, label

    async def load_training_data(
        self, csv_path: str, max_images: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        logger.info(f"Loading images from {csv_path}")

        # Read CSV in thread pool
        df = await asyncio.to_thread(pd.read_csv, csv_path)

        if max_images:
            df = df.head(max_images)
            logger.info(f"Limiting to {max_images} images")

        # Create tasks for parallel image loading
        tasks = [
            self._load_image_with_info(row["id"], row["label"])
            for _, row in df.iterrows()
        ]

        images, labels, ids = [], [], []
        failed_count = 0

        # Load images in parallel with progress bar
        with tqdm(total=len(tasks), desc="Loading images") as pbar:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.warning(f"Failed to load image: {result}")
                else:
                    img_id, img, label = cast(
                        Tuple[int, Optional[np.ndarray], Optional[str]], result
                    )
                    if img is not None:
                        images.append(img)
                        labels.append(label)
                        ids.append(img_id)
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to load image {img_id}")
                pbar.update(1)

        if failed_count > 0:
            logger.warning(f"Failed to load {failed_count} images")

        logger.info(f"Successfully loaded {len(images)} images")
        return np.array(images), np.array(labels), np.array(ids)

    async def load_test_data(
        self, start_id: int = 2701, end_id: int = 3000
    ) -> Tuple[np.ndarray, np.ndarray]:
        logger.info(f"Loading test images from {start_id} to {end_id}")

        # Create tasks for parallel image loading
        tasks = [
            self._load_image_with_info(img_id) for img_id in range(start_id, end_id + 1)
        ]

        images, ids = [], []
        failed_count = 0

        # Load images in parallel with progress bar
        with tqdm(total=len(tasks), desc="Loading test images") as pbar:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.warning(f"Failed to load image: {result}")
                else:
                    img_id, img, _ = cast(
                        Tuple[int, Optional[np.ndarray], Optional[str]], result
                    )
                    if img is not None:
                        images.append(img)
                        ids.append(img_id)
                    else:
                        failed_count += 1
                        logger.warning(f"Failed to load image {img_id}")
                pbar.update(1)

        if failed_count > 0:
            logger.warning(f"Failed to load {failed_count} test images")

        logger.info(f"Successfully loaded {len(images)} test images")
        return np.array(images), np.array(ids)
