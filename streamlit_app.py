import logging
from pathlib import Path
from typing import Any

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageClassification

from config.settings import (
    DATASET_PATH,
    DROPOUT_RATE,
    EMOTION_EMOJIS,
    EMOTION_LABELS,
    IDX_TO_LABEL,
    IMG_SIZE,
    LABEL_TO_IDX,
    MODEL_PATH,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Emotion Recognition",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_resource  # type: ignore
def load_model() -> tuple[Any, Any]:  # type: ignore
    """Load the trained ViT model."""
    try:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Load base model architecture
        model_name = "google/vit-base-patch16-224"
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=len(EMOTION_LABELS),
            id2label=IDX_TO_LABEL,
            label2id=LABEL_TO_IDX,
            ignore_mismatched_sizes=True,
        )

        # Apply the same classifier modification as in training
        if hasattr(model, "classifier"):
            original_classifier = model.classifier
            if isinstance(original_classifier, nn.Linear):
                model.classifier = nn.Sequential(
                    nn.Dropout(p=DROPOUT_RATE),
                    nn.Linear(
                        original_classifier.in_features,
                        original_classifier.out_features,
                    ),
                )

        # Load weights
        if MODEL_PATH.exists():
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            st.warning(
                f"Model file not found at {MODEL_PATH}. Using untrained base model."
            )

        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        logger.error(f"Error loading model: {e}", exc_info=True)
        return None, None


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for ViT: Grayscale->RGB, Resize, Normalize."""
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(
                (IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0)  # Add batch dimension


def predict_emotion(  # type: ignore
    model: Any, image_tensor: torch.Tensor, device: Any
) -> tuple[str, dict[str, float]]:
    """Predict emotion from image tensor."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor).logits
        probabilities = torch.softmax(outputs, dim=1)

    prob_dict = {
        IDX_TO_LABEL[idx]: float(prob) for idx, prob in enumerate(probabilities[0])
    }

    predicted_idx = torch.argmax(probabilities, dim=1).item()
    predicted_label = IDX_TO_LABEL[predicted_idx]

    return predicted_label, prob_dict


def main() -> None:
    st.title("🎭 Facial Emotion Recognition")

    model, device = load_model()

    if model is None:
        st.error("Failed to load model.")
        return

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📥 Input")
        input_method = st.radio(
            "Choose input:", ["Upload Image", "From Dataset"], horizontal=True
        )

        display_image = None

        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image...", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                display_image = Image.open(uploaded_file)
        else:
            image_id = st.number_input(
                "Image ID (e.g. from valid/test set)", min_value=1, value=2701
            )
            img_path = DATASET_PATH / f"{image_id}.jpg"
            if img_path.exists():
                display_image = Image.open(img_path)
            else:
                st.error(f"Image {image_id} not found in {DATASET_PATH}")

        if display_image:
            st.image(display_image, caption="Input Image", width=300)

            # Predict
            with st.spinner("Analyzing..."):
                img_tensor = preprocess_image(display_image)
                emotion, probabilities = predict_emotion(model, img_tensor, device)

            with col2:
                st.subheader("📊 Results")

                emoji_label = EMOTION_EMOJIS.get(emotion, emotion)
                st.markdown(
                    f"""
                    <div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px;'>
                        <h2 style='color: #1f77b4; margin:0;'>{emoji_label}</h2>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader("Confidence Scores")

                # Sort probabilities
                sorted_probs = sorted(
                    probabilities.items(), key=lambda x: x[1], reverse=True
                )

                for label, prob in sorted_probs:
                    display_label = EMOTION_EMOJIS.get(label, label)
                    st.progress(prob, text=f"{display_label}: {prob:.1%}")


if __name__ == "__main__":
    main()
