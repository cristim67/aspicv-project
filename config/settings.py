from pathlib import Path

import torch

# Paths
DATASET_PATH = Path("dataset")
TRAIN_CSV = Path("tags/train.csv")
OUTPUT_CSV = Path("submission.csv")
MODEL_PATH = Path("models/vit_emotion.pt")
LOG_FILE = Path("logs/training.log")

# Emotion Labels
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

EMOTION_EMOJIS = {
    "angry": "😠 Angry",
    "disgust": "🤢 Disgust",
    "fear": "😨 Fear",
    "happy": "😊 Happy",
    "neutral": "😐 Neutral",
    "sad": "😢 Sad",
    "surprise": "😲 Surprise",
}

# Image Config
IMG_SIZE = 224

# Training Configuration
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.05
DROPOUT_RATE = 0.3  # Standardized to 0.3
GRADIENT_CLIP_NORM = 1.0
USE_MIXED_PRECISION = True

# Device Configuration
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Augmentation Config
MIXUP_ALPHA = 0.4
LABEL_SMOOTHING = 0.1
