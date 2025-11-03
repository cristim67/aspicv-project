# Facial Expression Recognition System

Facial expression recognition system for the ASPICV project.

## Description

This system uses multiple feature extraction methods (HOG, LBP, raw pixels) and classification algorithms (SVM, MLP, Random Forest) to recognize facial expressions from 48×48 pixel grayscale images.

### Recognized Emotions
- happy
- sad
- angry
- neutral
- fear
- surprise
- disgust

## Project Structure

```
aspicv-project/
├── src/                      # Main source code
│   ├── __init__.py
│   ├── config/              # Configuration
│   │   ├── __init__.py
│   │   └── settings.py      # Settings (paths, IDs, etc.)
│   ├── features/            # Feature extraction
│   │   ├── __init__.py
│   │   └── feature_extractor.py
│   ├── models/             # Classification models
│   │   ├── __init__.py
│   │   └── emotion_classifier.py
│   ├── repository/         # Data access
│   │   ├── __init__.py
│   │   └── image_repository.py
│   ├── services/           # Business logic services
│   │   ├── __init__.py
│   │   ├── training_service.py
│   │   └── prediction_service.py
│   └── storage/            # Model persistence
│       ├── __init__.py
│       └── model_storage.py
├── utils/                  # Utilities
│   ├── __init__.py
│   └── logger.py           # Logging configuration
├── dataset/                # Images (48×48 pixels, 1.jpg - 3000.jpg)
├── tags/
│   └── train.csv           # Labels for images 1-2700
├── models/                 # Trained models (auto-generated)
│   ├── classifier.joblib
│   └── feature_extractor.joblib
├── logs/                   # Logs (auto-generated)
│   └── training.log
├── app.py                  # CLI script for training and prediction
├── streamlit_app.py        # Interactive web application
├── requirements.txt        # Python dependencies
├── mypy.ini                # Type checking configuration
├── .pre-commit-config.yaml # Pre-commit hooks configuration
└── README.md               # This file
```

## Installation

1. Install dependencies:
```bash
pip3 install -r requirements.txt
```

2. Configure pre-commit hooks:
```bash
pre-commit install
```

This will automatically install hooks that run on each commit:
- **isort**: sorts imports
- **mypy**: type checking
- **pre-commit-hooks**: general checks (trailing whitespace, end-of-file, etc.)

## Usage

### 1. Training and Full Prediction (CLI)

To train the model on the complete training set and generate predictions for the test set:

```bash
python3 app.py
```

The script will:
1. Load training images (1-2700)
2. Extract features (HOG, LBP, raw pixels)
3. Train an ensemble of classifiers (SVM + MLP + Random Forest)
4. Generate predictions for test images (2701-3000)
5. Create the `submission.csv` file

#### Caching (Model Cache)

The system supports automatic caching to avoid re-training:
- **First run**: trains and saves the model in `models/`
- **Subsequent runs**: loads the model from cache (much faster!)

```bash
# With cache (default)
python3 app.py

# Without cache (force re-training)
python3 app.py --no-cache
```

Models are saved in the `models/` folder:
- `models/feature_extractor.joblib` - feature extractor with scaler
- `models/classifier.joblib` - trained model

### 2. Interactive Web Application (Streamlit)

To use the interactive web interface for real-time predictions:

```bash
streamlit run streamlit_app.py
```

The application will run at `http://localhost:8501` and provides:

#### Features:
- **Upload Image**: Upload a local image for prediction
- **Load from Dataset**: Select an image from the training/test dataset by ID
- **Automatic Prediction**: Results are displayed automatically when an image is loaded
- **Detailed Visualization**:
  - Predicted emotion with icon
  - Confidence scores chart for all emotions
  - Detailed probabilities for each emotion

#### Characteristics:
- **Auto-load**: On startup, automatically loads the first image from the dataset
- **Auto-predict**: Prediction runs automatically when an image is loaded
- **Smart Caching**: Models and images are cached for optimal performance
- **Responsive Layout**: Interface optimized for different screen sizes

#### Note:
Make sure models have been trained (run `python3 app.py` at least once) before using the Streamlit application.

## Architecture

The system is organized in clear layers:

- **Repository Layer** (`src/repository/`): Data access, image loading
- **Feature Layer** (`src/features/`): Feature extraction (HOG, LBP, raw)
- **Model Layer** (`src/models/`): Emotion classifiers
- **Service Layer** (`src/services/`): Business logic (training, prediction)
- **Storage Layer** (`src/storage/`): Model persistence (save/load)

## Results

The `submission.csv` file will have the format:
```csv
id,label
2701,happy
2702,angry
...
3000,neutral
```

## Main Dependencies

- `numpy` - Numerical computations
- `pandas` - Data manipulation
- `opencv-python` - Image processing
- `scikit-learn` - Machine learning (SVM, MLP, Random Forest)
- `scikit-image` - Feature extraction (HOG, LBP)
- `streamlit` - Interactive web application
- `tqdm` - Progress bars
- `joblib` - Model serialization

## Logging

Logs are saved in:
- Console (stdout)
- File: `logs/training.log`

The logging level can be modified in `app.py`:
```python
logger = setup_logger(name="aspicv", log_level=logging.INFO, ...)
```

## Notes

- Images must be in the `dataset/` folder
- The `tags/train.csv` file must exist
- The test set consists of images 2701-3000 (300 images)
- Trained models are saved in `models/` for reuse
- The Streamlit application requires trained models to function
