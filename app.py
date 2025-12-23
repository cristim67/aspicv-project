import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModelForImageClassification

from config.settings import (
    BATCH_SIZE,
    DATASET_PATH,
    DEVICE,
    DROPOUT_RATE,
    EMOTION_LABELS,
    EPOCHS,
    GRADIENT_CLIP_NORM,
    IDX_TO_LABEL,
    IMG_SIZE,
    LABEL_SMOOTHING,
    LABEL_TO_IDX,
    LEARNING_RATE,
    LOG_FILE,
    MIXUP_ALPHA,
    MODEL_PATH,
    OUTPUT_CSV,
    TRAIN_CSV,
    USE_MIXED_PRECISION,
    WEIGHT_DECAY,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w"),
    ],
)
logger = logging.getLogger(__name__)


class EmotionDataset(Dataset):  # type: ignore
    """Dataset for 48x48 grayscale emotion images with proper preprocessing."""

    def __init__(
        self,
        image_ids: list[int],
        labels: list[str] | None,
        dataset_path: Path,
        augment: bool = False,
    ):
        self.image_ids = image_ids
        self.labels = labels
        self.dataset_path = dataset_path
        self.augment = augment

        # Transforms for training with augmentation - simplified back to "sweet spot"
        if augment:
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
                    transforms.Resize(
                        (IMG_SIZE, IMG_SIZE),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(15),  # Moderate rotation
                    transforms.RandomAffine(
                        degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)
                    ),
                    transforms.ColorJitter(
                        brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                    transforms.RandomErasing(
                        p=0.2, scale=(0.02, 0.33)
                    ),  # Mixup usually handles this, but slight erasing helps
                ]
            )
        else:
            # Validation/test transforms - no augmentation
            self.transform = transforms.Compose(
                [
                    transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
                    transforms.Resize(
                        (IMG_SIZE, IMG_SIZE),
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int | torch.Tensor]:
        img_id = self.image_ids[idx]
        img_path = self.dataset_path / f"{img_id}.jpg"

        # Load image
        image = Image.open(img_path)

        # Apply transforms
        image = self.transform(image)

        if self.labels is not None:
            label = LABEL_TO_IDX[self.labels[idx]]
            return image, label
        return image, -1


def load_training_data() -> tuple[list[int], list[str]]:
    """Load training data from CSV."""
    df = pd.read_csv(TRAIN_CSV)
    return df["id"].tolist(), df["label"].tolist()


def make_weighted_sampler(labels: list[str]) -> WeightedRandomSampler:
    """
    Create a WeightedRandomSampler to handle class imbalance.

    This ensures that the model sees roughly the same number of examples
    from each emotion class during an epoch, even if the dataset has
    many more 'happy' images than 'disgust' ones.
    """
    from collections import Counter

    counts = Counter(labels)
    total = len(labels)

    # Calculate weight per class: total / (n_classes * class_count)
    # Using sqrt to dampen the effect slightly (heuristic for very imbalanced data)
    class_weights = {label: np.sqrt(total / count) for label, count in counts.items()}

    # Assign weight to each sample
    sample_weights = [class_weights[label] for label in labels]
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

    # Create sampler
    # replacement=True allows the same minority sample to be picked multiple times in an epoch
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True,
    )
    return sampler


def mixup_data(  # type: ignore
    x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, device: Any = "cpu"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Applies Mixup augmentation to a batch of inputs and targets.

    Returns:
        mixed_x: The blended images.
        y_a, y_b: The two sets of original target labels.
        lam: The mixing coefficient (lambda).
    """
    if alpha > 0:
        # Sample lambda from Beta distribution
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    # Random permutation of indices to pair samples
    index = torch.randperm(batch_size).to(device)

    # Mix the images: new_x = λ * x1 + (1-λ) * x2
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Calculates the loss for Mixed-up inputs.

    Loss = λ * Loss(pred, label1) + (1-λ) * Loss(pred, label2)
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_model(  # type: ignore
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: Any,
    class_weights: torch.Tensor | None = None,
    save_path: Path | None = None,
    model_name: str = "vit",
) -> tuple[nn.Module, float]:
    """Fine-tune the model with proper training loop."""
    model.to(device)

    # Mixed precision scaler for M4
    scaler = (
        torch.cuda.amp.GradScaler()
        if USE_MIXED_PRECISION and device == "cuda"
        else None
    )
    if USE_MIXED_PRECISION and device == "mps":
        # MPS doesn't support AMP yet, but we can still optimize
        scaler = None

    # Optimizer with different learning rates for backbone and head
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if "classifier" in name or "head" in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr * 0.1},  # Lower LR for backbone
            {"params": head_params, "lr": lr},  # Higher LR for head
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # Use ReduceLROnPlateau for adaptive learning rate reduction
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for pixel_values, labels in pbar:
            pixel_values = pixel_values.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # Mixed precision training with Mixup
            # This speeds up training on GPUs (CUDA) by using 16-bit floats for some operations
            if scaler is not None and device == "cuda":
                with torch.cuda.amp.autocast():
                    # Apply Mixup: Blend images and labels
                    inputs, targets_a, targets_b, lam = mixup_data(
                        pixel_values, labels, MIXUP_ALPHA, device
                    )
                    # Forward pass: Get model predictions
                    outputs = model(inputs).logits
                    # Calculate loss using the mixup criterion
                    loss = mixup_criterion(
                        criterion, outputs, targets_a, targets_b, lam
                    )

                # Backward pass with gradient scaling (to prevent underflow in FP16)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRADIENT_CLIP_NORM
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision training (MPS/CPU)

                # Apply Mixup: Blend images and labels
                inputs, targets_a, targets_b, lam = mixup_data(
                    pixel_values, labels, MIXUP_ALPHA, device
                )

                # Forward pass
                outputs = model(inputs).logits

                # Calculate loss
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

                # Backward pass
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=GRADIENT_CLIP_NORM
                )

                # Update weights
                optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100*train_correct/train_total:.1f}%",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
            )

        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for pixel_values, labels in val_loader:
                pixel_values = pixel_values.to(device)
                labels = labels.to(device)

                outputs = model(pixel_values).logits
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)

        # Update learning rate scheduler based on validation loss
        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.1f}% | Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.1f}%"
        )
        logger.info(
            f"   Learning rate: {optimizer.param_groups[0]['lr']:.2e} (backbone), {optimizer.param_groups[1]['lr']:.2e} (head)"
        )

        # Early stopping with best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Save immediately to disk
            if save_path:
                save_path.parent.mkdir(exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "model_name": model_name,
                        "best_val_acc": best_val_acc,
                    },
                    save_path,
                )
                logger.info(
                    f"New best validation accuracy: {val_acc:.1f}% (Saved to {save_path})"
                )
            else:
                logger.info(f"New best validation accuracy: {val_acc:.1f}%")

            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"⏹️ Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val_acc


def predict_with_tta(  # type: ignore
    model: nn.Module, image_ids: list[int], dataset_path: Path, device: Any
) -> list[str]:
    """Predict with Test Time Augmentation for better accuracy."""
    model.eval()
    model.to(device)

    # Define TTA transforms
    tta_transforms = [
        # Original
        transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        # Horizontal flip
        transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(
                    (IMG_SIZE, IMG_SIZE),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        # Slight zoom
        transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize(
                    (int(IMG_SIZE * 1.1), int(IMG_SIZE * 1.1)),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.CenterCrop(IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    ]

    all_predictions = []

    with torch.no_grad():
        for img_id in tqdm(image_ids, desc="Predicting with TTA"):
            img_path = dataset_path / f"{img_id}.jpg"
            image = Image.open(img_path)

            # Accumulate probabilities from all augmentations
            probs_sum = None

            for tta_transform in tta_transforms:
                img_tensor = tta_transform(image).unsqueeze(0).to(device)
                outputs = model(img_tensor).logits
                probs = torch.softmax(outputs, dim=1)

                if probs_sum is None:
                    probs_sum = probs
                else:
                    probs_sum += probs

            # Average and get prediction
            if probs_sum is not None:
                avg_probs = probs_sum / len(tta_transforms)
                _, predicted = torch.max(avg_probs, 1)
                all_predictions.append(IDX_TO_LABEL[predicted.item()])

    return all_predictions


def main() -> None:
    logger.info("=" * 60)
    logger.info("Facial Emotion Recognition with Vision Transformer")
    logger.info("   Optimized for 48x48 grayscale FER-style images")
    logger.info("=" * 60)

    logger.info(f"Using device: {DEVICE}")
    if DEVICE == "mps":
        logger.info("   MPS (Metal) acceleration enabled")
    if USE_MIXED_PRECISION:
        logger.info("   Mixed precision training enabled")

    # Load pre-trained ViT model - using a model good for facial expressions
    logger.info("Loading pre-trained ViT model...")

    # Use base model (small doesn't exist in transformers, but we optimize training)
    # For better results, we'll use base but with optimized training
    model_name = "google/vit-base-patch16-224"

    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS),
        id2label=IDX_TO_LABEL,
        label2id=LABEL_TO_IDX,
        ignore_mismatched_sizes=True,
    )

    # Add dropout to classifier head for regularization
    if hasattr(model, "classifier"):
        # Replace classifier with dropout + linear
        original_classifier = model.classifier
        if isinstance(original_classifier, nn.Linear):
            model.classifier = nn.Sequential(
                nn.Dropout(p=DROPOUT_RATE),
                nn.Linear(
                    original_classifier.in_features, original_classifier.out_features
                ),
            )
            # Copy weights from original classifier
            model.classifier[1].weight.data = original_classifier.weight.data.clone()
            model.classifier[1].bias.data = original_classifier.bias.data.clone()

    logger.info(f"   Model: {model_name}")
    logger.info(
        f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}"
    )

    # Load training data
    logger.info("Loading training data...")
    train_ids, train_labels = load_training_data()
    logger.info(f"   Total training samples: {len(train_ids)}")

    # Show class distribution
    from collections import Counter

    label_counts = Counter(train_labels)
    logger.info("   Class distribution:")
    for label in EMOTION_LABELS:
        logger.info(f"      {label}: {label_counts.get(label, 0)}")

    # Split into train/validation (stratified)
    train_ids_split, val_ids_split, train_labels_split, val_labels_split = (
        train_test_split(
            train_ids,
            train_labels,
            test_size=0.15,
            random_state=42,
            stratify=train_labels,
        )
    )

    logger.info(
        f"   Train split: {len(train_ids_split)}, Validation split: {len(val_ids_split)}"
    )

    # Calculate sampler for balanced training
    logger.info(" Creating WeightedRandomSampler for balanced batches...")
    train_sampler = make_weighted_sampler(train_labels_split)

    # Create datasets
    train_dataset = EmotionDataset(
        train_ids_split, train_labels_split, DATASET_PATH, augment=True
    )
    val_dataset = EmotionDataset(
        val_ids_split, val_labels_split, DATASET_PATH, augment=False
    )

    # Create dataloaders
    # Note: shuffle must be False when using sampling
    # Optimized for MPS/Mac: 4 workers is usually a sweet spot for 8-10 core CPUs
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,  # Use sampler instead of shuffle
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    # Train
    logger.info("Training model...")
    logger.info(f"   Batch size: {BATCH_SIZE}")
    logger.info(f"   Epochs: {EPOCHS}")
    logger.info(f"   Learning rate: {LEARNING_RATE}")
    logger.info(f"   Mixup alpha: {MIXUP_ALPHA}")
    logger.info(f"   Label smoothing: {LABEL_SMOOTHING}")

    model, best_acc = train_model(
        model,
        train_loader,
        val_loader,
        EPOCHS,
        LEARNING_RATE,
        DEVICE,
        class_weights=None,
        save_path=MODEL_PATH,
        model_name=model_name,
    )

    # Model is already saved during training if improvements were found
    logger.info(f"Training completed. Best model saved at {MODEL_PATH}")

    # Predict test images (2701-3000) with TTA
    logger.info("Predicting test images (2701-3000) with TTA...")
    test_ids = list(range(2701, 3001))

    predictions = predict_with_tta(model, test_ids, DATASET_PATH, DEVICE)

    # Create submission
    submission = pd.DataFrame({"id": test_ids, "label": predictions})
    submission.to_csv(OUTPUT_CSV, index=False)

    logger.info(f"Submission saved to {OUTPUT_CSV}")
    logger.info(f"Best validation accuracy: {best_acc:.1f}%")
    logger.info("Prediction distribution:")
    logger.info("\n" + str(submission["label"].value_counts().sort_index()))


if __name__ == "__main__":
    main()
