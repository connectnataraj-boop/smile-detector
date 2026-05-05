# Smile Detector

A binary image classifier that detects whether a person is smiling, built with Logistic Regression on grayscale face images. Includes a Streamlit web app for real-time predictions.

---

## Demo

Upload any face photo → get a smile score (0–100%) and a smile / no smile prediction.

---

## Project Structure

```
smile_detector/
├── src/
│   ├── train.py   # Core: data loading, training, evaluation, inference
├── data/
│   ├── smile/              # Place smile .jpg images here
│   ├── non_smile/          # Place non-smile .jpg images here
│   └── test/               # Place test .jpg images here
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

Place your face images in the `data/` folder before training:

- `data/smile/` — JPG images of smiling faces (label: 1)
- `data/non_smile/` — JPG images of non-smiling faces (label: 0)

Each image is automatically converted to grayscale and resized to 64×64 pixels, giving a feature vector of 4096 pixel values per image.

A compatible dataset is the [Labeled Faces in the Wild (LFW)](http://vis-www.cs.umass.edu/lfw/) collection, filtered by smile/no-smile labels.

---

## Usage

### 1. Train the model

```bash
python src/train.py
```

What it does:

- Loads all images from `data/smile/` and `data/non_smile/`
- Converts each image to grayscale, resizes to 64×64, flattens to a 1D array
- Shuffles and splits data 80/20 (train/test)
- Scales features with `StandardScaler`
- Trains a baseline `LogisticRegression`
- Runs `GridSearchCV` over `C` and `l1_ratio` to find best hyperparameters
- Trains final model with best params
- Saves `best_model.pkl` and `scaler.pkl` to `models/`
- Runs a single test image prediction

Example output:

```
Baseline accuracy: 0.8423
Best model accuracy: 0.8506

── Saved models ──────────────────────
  Model  → models/best_model.pkl
  Scaler → models/scaler.pkl

Prediction: Smile
```

### 2. Run the web app

```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser, upload a face photo, and get the smile prediction with a confidence score.

---

## Pipeline

| Step | Details |
|---|---|
| Load | Reads `.jpg` images from `data/smile/` and `data/non_smile/` using `glob` |
| Preprocess | Grayscale convert → resize 64×64 → flatten to 4096-dim vector |
| Shuffle | Randomised with `random_state=42` for reproducibility |
| Split | 80% train, 20% test, `random_state=42` |
| Scale | `StandardScaler` fitted on train only, applied to both splits |
| Baseline | `LogisticRegression` with default settings |
| Tune | `GridSearchCV` over `C=[0.01, 0.1, 1, 10, 100]` and `l1_ratio=[0, 1]`, `solver=liblinear`, 5-fold CV |
| Evaluate | Accuracy score on test set |
| Save | Model and scaler pickled to `models/` |

---

## Hyperparameters Searched

| Parameter | Values | Meaning |
|---|---|---|
| `C` | 0.01, 0.1, 1, 10, 100 | Inverse regularization strength — smaller = stronger regularization |
| `l1_ratio` | 0, 1 | 0 = L2 regularization, 1 = L1 regularization |
| `solver` | liblinear | Supports both L1 and L2 penalties |

---

## Requirements

```
Pillow>=9.0.0
numpy>=1.23.0
scikit-learn>=1.2.0
streamlit>=1.20.0
joblib>=1.2.0
```

Install all at once:

```bash
pip install -r requirements.txt
```

---

## Key Design Decisions

- **Grayscale conversion** — colour channels are not useful for smile detection; removing them reduces the feature space from 3×64×64 = 12,288 to 4,096 dimensions
- **StandardScaler on train only** — `fit_transform` on train, `transform` on test — prevents data leakage
- **`liblinear` solver** — chosen because it supports both L1 and L2 penalties, unlike the default `lbfgs` which only supports L2
- **`l1_ratio` instead of `penalty`** — `penalty` parameter is deprecated in sklearn 1.8+ and will be removed in 1.10
- **`@st.cache_resource` in app.py** — loads model and scaler once and reuses across Streamlit reruns, avoiding repeated disk reads

---

## License

MIT
