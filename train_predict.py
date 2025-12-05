"""
Facial Emotion Recognition using Pre-trained Vision Transformer.
Optimized for 48x48 grayscale FER-style images.
"""
import os
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Configuration
DATASET_PATH = Path("dataset")
TRAIN_CSV = Path("tags/train.csv")
OUTPUT_CSV = Path("submission.csv")
MODEL_SAVE_PATH = Path("models/vit_emotion.pt")

# Emotion labels (sorted alphabetically for consistency)
EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
IDX_TO_LABEL = {idx: label for idx, label in enumerate(EMOTION_LABELS)}

# Training config - optimized for better accuracy and reduced overfitting
BATCH_SIZE = 32  # Smaller batch for M4 stability
EPOCHS = 25  # More epochs for better convergence
LEARNING_RATE = 5e-5  # Slightly higher LR
WEIGHT_DECAY = 0.05  # Increased for stronger L2 regularization
DROPOUT_RATE = 0.4  # Dropout rate for classifier head
GRADIENT_CLIP_NORM = 0.5  # Reduced gradient clipping for smaller gradients
USE_MIXED_PRECISION = True  # Faster on M4
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# Image config - original images are 48x48 grayscale
IMG_SIZE = 224  # ViT expects 224x224

# Use smaller model for faster training and better convergence
USE_SMALL_MODEL = True  # ViT-small instead of base

print(f"🔧 Using device: {DEVICE}")
if DEVICE == "mps":
    print("   ⚡ MPS (Metal) acceleration enabled")
if USE_MIXED_PRECISION:
    print("   🚀 Mixed precision training enabled")


class EmotionDataset(Dataset):
    """Dataset for 48x48 grayscale emotion images with proper preprocessing."""
    
    def __init__(self, image_ids: list, labels: list | None, dataset_path: Path, augment: bool = False):
        self.image_ids = image_ids
        self.labels = labels
        self.dataset_path = dataset_path
        self.augment = augment
        
        # Transforms for training with augmentation - improved for better regularization
        if augment:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
                transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(15),  # Increased rotation range
                transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),  # Added scale variation
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.0, hue=0.0),  # Increased variation
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.33)),  # Increased probability and scale
            ])
        else:
            # Validation/test transforms - no augmentation
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
                transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
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


def load_training_data():
    """Load training data from CSV."""
    df = pd.read_csv(TRAIN_CSV)
    return df["id"].tolist(), df["label"].tolist()


def get_class_weights(labels):
    """Calculate class weights for imbalanced dataset."""
    from collections import Counter
    counts = Counter(labels)
    total = len(labels)
    n_classes = len(EMOTION_LABELS)
    
    weights = []
    for label in EMOTION_LABELS:
        count = counts.get(label, 1)
        weight = total / (n_classes * count)
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def train_model(model, train_loader, val_loader, epochs, lr, device, class_weights=None):
    """Fine-tune the model with proper training loop."""
    model.to(device)
    
    # Mixed precision scaler for M4
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION and device == "cuda" else None
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
    
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},  # Lower LR for backbone
        {"params": head_params, "lr": lr}              # Higher LR for head
    ], weight_decay=WEIGHT_DECAY)
    
    # Use ReduceLROnPlateau for adaptive learning rate reduction
    # This reduces LR when validation loss plateaus, helping with overfitting
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-7
    )
    
    # Loss with class weights
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    patience = 6  # Reduced patience for earlier stopping when overfitting occurs
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
            
            # Mixed precision training
            if scaler is not None and device == "cuda":
                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values).logits
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(pixel_values).logits
                loss = criterion(outputs, labels)
                loss.backward()
                # Gradient clipping with reduced norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_NORM)
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "acc": f"{100*train_correct/train_total:.1f}%",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
        
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
        
        print(f"\n📊 Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.1f}% | Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.1f}%")
        print(f"   Learning rate: {optimizer.param_groups[0]['lr']:.2e} (backbone), {optimizer.param_groups[1]['lr']:.2e} (head)")
        
        # Early stopping with best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"✅ New best validation accuracy: {val_acc:.1f}%")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"⏹️ Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, best_val_acc


def predict_with_tta(model, image_ids, dataset_path, device):
    """Predict with Test Time Augmentation for better accuracy."""
    model.eval()
    model.to(device)
    
    # Define TTA transforms
    tta_transforms = [
        # Original
        transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Horizontal flip
        transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        # Slight zoom
        transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((int(IMG_SIZE * 1.1), int(IMG_SIZE * 1.1)), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
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
            avg_probs = probs_sum / len(tta_transforms)
            _, predicted = torch.max(avg_probs, 1)
            all_predictions.append(IDX_TO_LABEL[predicted.item()])
    
    return all_predictions


def main():
    print("=" * 60)
    print("🎭 Facial Emotion Recognition with Vision Transformer")
    print("   Optimized for 48x48 grayscale FER-style images")
    print("=" * 60)
    
    # Load pre-trained ViT model - using a model good for facial expressions
    print("\n📦 Loading pre-trained ViT model...")
    
    # Use base model (small doesn't exist in transformers, but we optimize training)
    # For better results, we'll use base but with optimized training
    model_name = "google/vit-base-patch16-224"
    
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS),
        id2label=IDX_TO_LABEL,
        label2id=LABEL_TO_IDX,
        ignore_mismatched_sizes=True
    )
    
    # Add dropout to classifier head for regularization
    if hasattr(model, 'classifier'):
        # Replace classifier with dropout + linear
        original_classifier = model.classifier
        if isinstance(original_classifier, nn.Linear):
            model.classifier = nn.Sequential(
                nn.Dropout(p=DROPOUT_RATE),
                nn.Linear(original_classifier.in_features, original_classifier.out_features)
            )
            # Copy weights from original classifier
            model.classifier[1].weight.data = original_classifier.weight.data.clone()
            model.classifier[1].bias.data = original_classifier.bias.data.clone()
    
    print(f"   Model: {model_name}")
    print(f"   Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load training data
    print("\n📂 Loading training data...")
    train_ids, train_labels = load_training_data()
    print(f"   Total training samples: {len(train_ids)}")
    
    # Show class distribution
    from collections import Counter
    label_counts = Counter(train_labels)
    print("   Class distribution:")
    for label in EMOTION_LABELS:
        print(f"      {label}: {label_counts.get(label, 0)}")
    
    # Split into train/validation (stratified)
    train_ids_split, val_ids_split, train_labels_split, val_labels_split = train_test_split(
        train_ids, train_labels, test_size=0.15, random_state=42, stratify=train_labels
    )
    
    print(f"\n   Train split: {len(train_ids_split)}, Validation split: {len(val_ids_split)}")
    
    # Calculate class weights
    class_weights = get_class_weights(train_labels_split)
    print(f"   Class weights: {dict(zip(EMOTION_LABELS, class_weights.numpy().round(2)))}")
    
    # Create datasets
    train_dataset = EmotionDataset(train_ids_split, train_labels_split, DATASET_PATH, augment=True)
    val_dataset = EmotionDataset(val_ids_split, val_labels_split, DATASET_PATH, augment=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    # Train
    print("\n🏋️ Training model...")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Learning rate: {LEARNING_RATE}")
    
    model, best_acc = train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, DEVICE, class_weights)
    
    # Save model
    print(f"\n💾 Saving model to {MODEL_SAVE_PATH}...")
    MODEL_SAVE_PATH.parent.mkdir(exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": model_name,
        "best_val_acc": best_acc,
    }, MODEL_SAVE_PATH)
    
    # Predict test images (2701-3000) with TTA
    print("\n🔮 Predicting test images (2701-3000) with TTA...")
    test_ids = list(range(2701, 3001))
    
    predictions = predict_with_tta(model, test_ids, DATASET_PATH, DEVICE)
    
    # Create submission
    submission = pd.DataFrame({
        "id": test_ids,
        "label": predictions
    })
    submission.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✅ Submission saved to {OUTPUT_CSV}")
    print(f"📈 Best validation accuracy: {best_acc:.1f}%")
    print("\n📊 Prediction distribution:")
    print(submission["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
