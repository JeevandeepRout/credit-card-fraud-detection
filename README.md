# Credit Card Fraud Detection System

An interactive Streamlit dashboard that trains, evaluates, and compares
machine-learning models for credit-card fraud detection on a highly
imbalanced dataset.

> **Academic project** — not intended for production banking or payment systems.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Exploratory Data Analysis** | Dataset dimensions, missing values, class imbalance, and per-feature distribution plots |
| **Model Training** | Logistic Regression, Random Forest, and XGBoost with stratified splitting and consistent preprocessing |
| **Model Evaluation** | Precision, Recall, F1-Score, PR-AUC, False-Positive Rate, confusion matrices, and PR curves |
| **Interactive Prediction** | Submit a single transaction (all 30 dataset features) and receive a fraud/legitimate classification with model score |
| **Prediction History** | Session-based log of every prediction with timestamps, scores, and applied thresholds |

## 📂 Project Structure

```
credit-card-fraud-detection/
├── app.py                          # Streamlit entry point (Home page)
├── pages/
│   ├── 1_Data_Exploration.py       # EDA dashboard
│   ├── 2_Model_Training.py         # Train & save models
│   ├── 3_Model_Evaluation.py       # Compare model performance
│   ├── 4_Predict_Transaction.py    # Single-transaction prediction
│   └── 5_Prediction_History.py     # Session prediction log
├── src/
│   ├── __init__.py
│   ├── utils.py                    # Constants, data classes, helpers
│   ├── data_loader.py              # CSV loading & schema validation
│   ├── preprocessing.py            # Scaling, splitting, scaler I/O
│   ├── models.py                   # Training, saving, loading models
│   ├── evaluation.py               # Metrics & comparison plots
│   └── prediction.py               # Input validation & inference
├── artifacts/                      # Saved models & scaler (auto-created)
├── data/                           # Place creditcard.csv here
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Setup

### 1. Clone or download this repository

```bash
cd credit-card-fraud-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download **creditcard.csv** from the
[Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
page and place it inside the `data/` folder:

```
data/creditcard.csv
```

### 5. Run the application

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`.

---

## 🔬 Methodology

### Dataset

The [Kaggle credit-card fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
contains **284,807 European transactions** made over two days in September 2013.
Only **492 (0.172%)** are fraudulent, making this a severely imbalanced binary
classification problem.

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed since the first transaction |
| `V1` – `V28` | PCA-transformed anonymised features |
| `Amount` | Transaction amount |
| `Class` | Target — 0 = Legitimate, 1 = Fraud |

### Preprocessing

1. **Stratified train/test split** (default 80/20) preserves the class ratio.
2. `StandardScaler` is **fitted on the training set only** and applied to both
   splits (and later to prediction inputs). Only `Time` and `Amount` are scaled;
   `V1`–`V28` are already PCA-normalised.
3. All artifacts (scaler, feature ordering, models) are saved to `artifacts/`
   for consistent inference.

### Models

| Model | Imbalance Strategy |
|-------|--------------------|
| **Logistic Regression** | `class_weight="balanced"` |
| **Random Forest** (100 trees) | `class_weight="balanced"` |
| **XGBoost** (100 rounds) | `scale_pos_weight = n_legit / n_fraud` |

### Evaluation

- **Precision, Recall, F1-Score** — per-class and macro
- **PR-AUC** (Average Precision) — better than ROC-AUC for imbalanced data
- **False-Positive Rate** — `FP / (FP + TN)`
- **Confusion matrix** — per-model heatmaps
- Adjustable **classification threshold** (default 0.5)

> Accuracy is intentionally **not** used as a primary metric because a naïve
> "predict all legitimate" classifier achieves 99.83% accuracy.

---

## ⚠️ Disclaimer

- All scores and predictions are model outputs on a specific test split and
  are **NOT calibrated real-world fraud probabilities**.
- This application does **not** collect real card numbers, CVVs, or personal
  information.
- Not suitable for production fraud detection, automatic transaction blocking,
  or customer-facing alerts.

---

## 📄 License

This project is provided for educational purposes.
The dataset is released under the
[Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1-0/).
