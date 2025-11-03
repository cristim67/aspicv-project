import asyncio
import logging

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import Settings
from src.features import FeatureExtractor
from src.models import EmotionClassifier
from src.repository import ImageRepository
from src.storage import ModelStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Emotion Recognition",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EMOTION_LABELS = {
    "angry": "😠 Angry",
    "disgust": "🤢 Disgust",
    "fear": "😨 Fear",
    "happy": "😊 Happy",
    "neutral": "😐 Neutral",
    "sad": "😢 Sad",
    "surprise": "😲 Surprise",
}

MAX_DISPLAY_WIDTH = 600
MAX_DISPLAY_HEIGHT = 500
MIN_DISPLAY_SIZE = 300


@st.cache_resource  # type: ignore
def load_models() -> tuple[EmotionClassifier | None, FeatureExtractor | None]:
    """Load trained models with async event loop handling."""
    storage = ModelStorage(Settings.STORAGE_DIR)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        classifier = loop.run_until_complete(storage.load_classifier())
        feature_extractor = loop.run_until_complete(storage.load_feature_extractor())
        return (
            (classifier, feature_extractor)
            if classifier and feature_extractor
            else (None, None)
        )
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)
        return None, None


@st.cache_resource  # type: ignore
def load_image_repository() -> ImageRepository:
    return ImageRepository(Settings.DATASET_PATH)


@st.cache_data(ttl=3600)  # type: ignore
def load_image_by_id_sync(
    image_id: int,
) -> tuple[np.ndarray | None, Image.Image | None]:
    """Load image from dataset by ID with caching."""
    repo = load_image_repository()

    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        _, img_array, _ = loop.run_until_complete(repo._load_image_with_info(image_id))

        if img_array is None:
            return None, None

        img_display = (img_array * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_display, mode="L")
        return img_array, pil_image
    except Exception as e:
        logger.error(f"Error loading image {image_id}: {e}", exc_info=True)
        return None, None


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to model input format: grayscale, 48x48, normalized."""
    img_array = np.array(image.convert("L"))
    img_resized = cv2.resize(img_array, (48, 48))
    return img_resized.astype(np.float32) / 255.0


def resize_for_display(image: Image.Image) -> Image.Image:
    """Upscale small images (≤100px) to min size, downscale large ones to max size."""
    if image.width <= 100 or image.height <= 100:
        scale_factor = max(
            MIN_DISPLAY_SIZE / image.width, MIN_DISPLAY_SIZE / image.height
        )
        new_size = (int(image.width * scale_factor), int(image.height * scale_factor))
        image = image.resize(new_size, Image.Resampling.NEAREST)

    if image.width > MAX_DISPLAY_WIDTH or image.height > MAX_DISPLAY_HEIGHT:
        ratio = min(MAX_DISPLAY_WIDTH / image.width, MAX_DISPLAY_HEIGHT / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    return image


async def predict_emotion(
    image: np.ndarray,
    classifier: EmotionClassifier,
    feature_extractor: FeatureExtractor,
) -> tuple[str, dict[str, float]]:
    """Extract features and predict emotion with probabilities."""
    features = await feature_extractor.transform(
        np.array([image]), context="single image"
    )

    prediction = await asyncio.to_thread(classifier.predict, features)
    probabilities = await asyncio.to_thread(classifier.predict_proba, features)

    prob_dict = {
        label: float(prob)
        for label, prob in zip(classifier.idx_to_label.values(), probabilities[0])
    }

    return prediction[0], prob_dict


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Get or create async event loop."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


def main() -> None:
    st.markdown(
        """
        <style>
        .main > div { padding-top: 0.5rem; }
        .stImage { width: 100% !important; display: flex; justify-content: center; }
        .stImage img { max-width: 100% !important; width: auto !important; height: auto !important;
                       max-height: 500px; object-fit: contain; margin: 0 auto; }
        h1 { font-size: 1.75rem; margin-bottom: 0.5rem; }
        h2 { font-size: 1.25rem; margin-top: 0.75rem; margin-bottom: 0.5rem; }
        h3 { font-size: 1rem; margin-top: 0.5rem; margin-bottom: 0.25rem; }
        .stMarkdown { margin-bottom: 0.5rem; }
        .stButton > button { width: 100%; margin-top: 0.75rem; margin-bottom: 0.5rem; }
        [data-testid="stVerticalBlock"] { gap: 0.75rem; }
        [data-testid="column"] { align-items: flex-start; }
        [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap: 0.5rem; }
        .stRadio > div { gap: 0.5rem; }
        .stNumberInput { margin-bottom: 0.5rem; }
        </style>
    """,
        unsafe_allow_html=True,
    )

    with st.spinner("Loading models..."):
        classifier, feature_extractor = load_models()

    if classifier is None or feature_extractor is None:
        st.error(
            "❌ Models not found! Please train the model first by running `python app.py`"
        )
        st.info("The models should be saved in the `models/` directory.")
        return

    if "models_loaded_toast_shown" not in st.session_state:
        st.toast("Models loaded successfully!", icon="✅", duration=3)
        st.session_state.models_loaded_toast_shown = True

    with st.sidebar:
        st.header("Settings")
        st.markdown("---")
        st.subheader("Model Info")
        st.write(f"**Storage Dir:** {Settings.STORAGE_DIR}")
        st.write(f"**Dataset Path:** {Settings.DATASET_PATH}")
        st.write(f"**Image Size:** 48x48 pixels")
        st.write(f"**Feature Types:** HOG, LBP, Raw")

    col1, col2 = st.columns([1, 1])

    # Initialize session state
    if "processed_image" not in st.session_state:
        st.session_state.processed_image = None
    if "input_mode" not in st.session_state:
        st.session_state.input_mode = "load_from_dataset_(id)"
    if "display_image" not in st.session_state:
        st.session_state.display_image = None

    with col1:
        st.subheader("📥 Input Method")

        input_mode = st.radio(
            "Choose input method:",
            ["Upload Image", "Load from Dataset (ID)"],
            key="input_mode_selector",
            help="Select how you want to provide the image",
            horizontal=True,
            index=1,
        )

        st.session_state.input_mode = input_mode.lower().replace(" ", "_")

        if (
            "last_input_mode" not in st.session_state
            or st.session_state.last_input_mode != st.session_state.input_mode
        ):
            st.session_state.display_image = None
            st.session_state.last_input_mode = st.session_state.input_mode

        if input_mode == "Upload Image":
            st.subheader("📤 Upload Image")
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=["jpg", "jpeg", "png"],
                help="Upload a face image to predict the emotion",
            )

            if uploaded_file is not None:
                display_image = Image.open(uploaded_file)
                st.image(
                    resize_for_display(display_image),
                    caption="Uploaded Image",
                    width="stretch",
                )

                st.session_state.processed_image = preprocess_image(
                    Image.open(uploaded_file)
                )

                if (
                    "last_uploaded_file" not in st.session_state
                    or st.session_state.last_uploaded_file != uploaded_file.name
                ):
                    st.session_state.auto_predict = True
                    st.session_state.last_uploaded_file = uploaded_file.name
                    st.rerun()
            else:
                st.session_state.processed_image = None
                if "last_uploaded_file" in st.session_state:
                    del st.session_state.last_uploaded_file

        else:
            st.subheader("🔍 Load from Dataset")
            image_id = st.number_input(
                "Enter Image ID:",
                min_value=1,
                value=1,
                step=1,
                help="Enter the ID of the image from the dataset (e.g., 1, 2, 3, ...)",
            )

            st.session_state.current_image_id = image_id

            if (
                "display_image" in st.session_state
                and st.session_state.display_image is not None
            ):
                st.image(
                    resize_for_display(st.session_state.display_image),
                    caption=f"Image ID: {image_id}",
                    width="stretch",
                )
            elif (
                "last_loaded_id" in st.session_state
                and st.session_state.last_loaded_id == image_id
                and st.session_state.processed_image is not None
            ):
                img_display = (st.session_state.processed_image * 255).astype(np.uint8)
                pil_image = Image.fromarray(img_display, mode="L")
                st.image(
                    resize_for_display(pil_image),
                    caption=f"Image ID: {image_id}",
                    width="stretch",
                )

            if (
                "last_loaded_id" not in st.session_state
                or st.session_state.last_loaded_id != image_id
            ):
                if (
                    st.session_state.processed_image is None
                    or st.session_state.last_loaded_id != image_id
                ):
                    with st.spinner(f"Loading image {image_id} from dataset..."):
                        try:
                            img_array, pil_image = load_image_by_id_sync(image_id)

                            if img_array is not None and pil_image is not None:
                                st.session_state.processed_image = img_array
                                st.session_state.last_loaded_id = image_id
                                st.session_state.display_image = pil_image
                                st.session_state.auto_predict = True
                                st.rerun()
                            else:
                                st.error(
                                    f"❌ Image with ID {image_id} not found in dataset!"
                                )
                                st.info(
                                    f"Please check if the image exists at: {Settings.DATASET_PATH}/{image_id}.jpg"
                                )
                        except Exception as e:
                            st.error(f"❌ Error loading image: {str(e)}")
                            logger.error(
                                f"Error loading image {image_id}: {e}", exc_info=True
                            )

    with col2:
        should_predict = st.session_state.get("auto_predict", False)

        if should_predict:
            st.session_state.auto_predict = False

            if st.session_state.input_mode == "load_from_dataset_(id)":
                if "current_image_id" in st.session_state:
                    image_id = st.session_state.current_image_id
                    if (
                        "last_loaded_id" not in st.session_state
                        or st.session_state.last_loaded_id != image_id
                        or st.session_state.processed_image is None
                    ):
                        with st.spinner(f"Loading image {image_id} from dataset..."):
                            try:
                                img_array, pil_image = load_image_by_id_sync(image_id)

                                if img_array is not None and pil_image is not None:
                                    st.session_state.processed_image = img_array
                                    st.session_state.last_loaded_id = image_id
                                    st.session_state.display_image = pil_image
                                else:
                                    st.error(
                                        f"❌ Image with ID {image_id} not found in dataset!"
                                    )
                                    st.info(
                                        f"Please check if the image exists at: {Settings.DATASET_PATH}/{image_id}.jpg"
                                    )
                                    st.stop()
                            except Exception as e:
                                st.error(f"❌ Error loading image: {str(e)}")
                                logger.error(
                                    f"Error loading image {image_id}: {e}",
                                    exc_info=True,
                                )
                                st.stop()

            if st.session_state.processed_image is not None:
                with st.spinner("Analyzing image..."):
                    try:
                        loop = _get_event_loop()
                        emotion, probabilities = loop.run_until_complete(
                            predict_emotion(
                                st.session_state.processed_image,
                                classifier,
                                feature_extractor,
                            )
                        )

                        st.subheader("📊 Prediction Results")

                        emotion_display = EMOTION_LABELS.get(emotion, f"🎭 {emotion}")
                        st.markdown(
                            f"""
                            <div style='text-align: center; padding: 15px; background-color: #f0f2f6; border-radius: 10px; margin: 0.5rem 0;'>
                                <div style='color: #1f77b4; font-size: 1.5rem; font-weight: bold; margin-bottom: 5px;'>{emotion_display}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        st.subheader("📈 Confidence Scores")

                        sorted_probs = sorted(
                            probabilities.items(), key=lambda x: x[1], reverse=True
                        )
                        labels = [EMOTION_LABELS.get(k, k) for k, _ in sorted_probs]
                        values = [v for _, v in sorted_probs]

                        st.bar_chart(dict(zip(labels, values)), height=220)

                        with st.expander("📋 Detailed Probabilities", expanded=True):
                            for label, prob in sorted_probs:
                                display_label = EMOTION_LABELS.get(label, label)
                                st.progress(prob, text=f"{display_label}: {prob:.2%}")

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        logger.error(f"Prediction error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
