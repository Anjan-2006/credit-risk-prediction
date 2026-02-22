# Credit Risk Predictor

A machine learning project that predicts the probability of a borrower defaulting on a loan and maps that probability to an interpretable credit score on a 300-900 scale. The model is served through an interactive Streamlit web application.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Machine Learning Pipeline](#machine-learning-pipeline)
  - [1. Data Merging](#1-data-merging)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Exploratory Data Analysis](#3-exploratory-data-analysis)
  - [4. Feature Engineering](#4-feature-engineering)
  - [5. Feature Selection](#5-feature-selection)
  - [6. Feature Encoding and Scaling](#6-feature-encoding-and-scaling)
  - [7. Handling Class Imbalance](#7-handling-class-imbalance)
  - [8. Model Training and Comparison](#8-model-training-and-comparison)
  - [9. Hyperparameter Tuning with Optuna](#9-hyperparameter-tuning-with-optuna)
  - [10. Model Evaluation](#10-model-evaluation)
  - [11. Model Serialization](#11-model-serialization)
- [Credit Score Calculation](#credit-score-calculation)
- [Streamlit Application](#streamlit-application)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Input Fields and Valid Ranges](#input-fields-and-valid-ranges)
- [Dependencies](#dependencies)

---

## Project Overview

The goal of this project is to assess the credit risk of a borrower by:

1. Predicting the probability that the borrower will default on a loan.
2. Converting that probability into a credit score in the range 300-900.
3. Assigning a human-readable rating: Poor, Average, Good, or Excellent.

The final model is a Logistic Regression classifier tuned using Optuna, trained on SMOTETomek-balanced data, and evaluated with AUC-ROC, Gini coefficient, and KS statistic metrics.

---

## Project Structure

```

├── main.py                  # Streamlit web application (user interface)
├── main.ipynb               # Full model training and experimentation notebook
├── prediction_helper.py     # Inference logic: data preparation and scoring
├── requirements.txt         # Python package dependencies
└── artifacts/
    └── model_data.joblib    # Serialized model, scaler, features, and cols_to_scale
```

---

## Dataset

Three CSV files are used, joined on `cust_id`:

| File | Description |
|---|---|
| `customers.csv` | Demographic and personal information about borrowers |
| `bureau_data.csv` | Credit bureau data including account history and enquiry counts |
| `loans.csv` | Loan-specific data including amount, tenure, DPD, and delinquency information |

The target variable is `default` (binary: 1 = defaulted, 0 = did not default).

---

## Machine Learning Pipeline

The full pipeline is documented and implemented in `main.ipynb`.

### 1. Data Merging

The three datasets are merged on `cust_id` to produce a single unified dataframe. The `default` column is cast from boolean to integer.

### 2. Data Cleaning

- Train-test split (75/25) is performed before any cleaning to prevent data leakage.
- Missing values in `residence_type` are filled with the mode of the training set.
- Duplicate rows are checked and confirmed absent.
- Outliers in `processing_fee` are removed by enforcing the business rule: `processing_fee < loan_amount * 0.03`.
- A known data entry error (`"Personaal"` in `loan_purpose`) is corrected to `"Personal"`.

### 3. Exploratory Data Analysis

KDE plots are generated for all numeric features split by the target variable (`default = 0` vs `default = 1`). Key insights:

- Younger borrowers have higher default rates.
- Lower income borrowers default more frequently.
- Higher delinquency duration and higher DPD values strongly correlate with default.

### 4. Feature Engineering

Three ratio features are engineered to improve predictive power:

| Feature | Formula | Rationale |
|---|---|---|
| `loan_to_income` | `loan_amount / income` | Measures debt burden relative to earning capacity |
| `delinquent_to_loan` | `(delinquent_months / total_loan_months) * 100` | Fraction of loan tenure spent in delinquency |
| `avg_dpd_to_deliquency` | `total_dpd / delinquent_months` (0 if no delinquency) | Average days past due per delinquent month |

### 5. Feature Selection

Two techniques are used to eliminate redundant or weak features:

**Variance Inflation Factor (VIF):** Detects multicollinearity. Features with high VIF that are also collinear with others (`sanction_amount`, `processing_fee`, `gst`, `net_disbursement`, `principal_outstanding`) are dropped.

**Information Value (IV) and Weight of Evidence (WOE):** Measures the predictive strength of each feature with respect to the target. Features with IV below 0.02 are excluded as they carry insufficient information.

### 6. Feature Encoding and Scaling

- Categorical features are one-hot encoded using `pd.get_dummies` with `drop_first=True` to avoid the dummy variable trap.
- Numeric features are scaled using `MinMaxScaler` fitted only on the training data. The scaler is saved alongside the model to ensure consistent transformation at inference time.

### 7. Handling Class Imbalance

The dataset is imbalanced (most borrowers do not default). Three strategies are tested:

| Strategy | Description |
|---|---|
| No resampling | Baseline |
| `RandomUnderSampler` | Randomly reduces majority class |
| `SMOTETomek` | Combines SMOTE oversampling with Tomek link cleaning — used for final model |

### 8. Model Training and Comparison

Three classifiers are trained and compared on the same encoded and scaled data:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

All are evaluated using precision, recall, and macro F1-score from `classification_report`. Logistic Regression with SMOTETomek balancing and Optuna tuning produces the best results and is selected as the final model.

### 9. Hyperparameter Tuning with Optuna

Optuna is used to optimize the Logistic Regression hyperparameters, tuning:

- `C` (regularization strength, log-scale)
- `solver` (lbfgs, liblinear, saga, newton-cg)
- `tol` (convergence tolerance)
- `class_weight` (None or balanced)

The objective metric is macro F1-score evaluated via cross-validation on the SMOTETomek-balanced training data.

A similar Optuna study is run for XGBoost (tuning `lambda`, `alpha`, `subsample`, `colsample_bytree`, etc.), but the tuned Logistic Regression outperforms it on the test set.

### 10. Model Evaluation

The final model is evaluated using:

| Metric | Value |
|---|---|
| AUC-ROC | ~0.98 |
| Gini Coefficient | ~0.96 |
| KS Statistic | Computed via decile analysis |

The ROC curve and decile rank-ordering table confirm that the model has near-perfect discrimination ability and proper monotonic rank ordering across probability deciles. High deciles (top predicted default probabilities) consistently contain the majority of actual defaulters.

### 11. Model Serialization

The following objects are saved together into `artifacts/model_data.joblib` using `joblib.dump`:

```python
{
    "model":        final_model,          # Trained LogisticRegression instance
    "features":     X_train_encoded.columns,  # Feature names in exact order
    "scaler":       scaler,               # Fitted MinMaxScaler
    "cols_to_scale": cols_to_scale        # List of columns that need scaling
}
```

---

## Credit Score Calculation

At inference time, the model outputs a default probability. This is converted to a credit score as follows:

```
non_default_probability = 1 - default_probability
credit_score = 300 + (non_default_probability * 600)
```

This maps scores to the range 300-900. Ratings are assigned as:

| Score Range | Rating |
|---|---|
| 300 - 499 | Poor |
| 500 - 649 | Average |
| 650 - 749 | Good |
| 750 - 900 | Excellent |

---

## Streamlit Application

The web interface is implemented in `main.py`. It collects user inputs for 11 features and calls the `predict` function from `prediction_helper.py`.

The `prediction_helper.py` module:

1. Loads the saved model artifacts from `artifacts/model_data.joblib` at module import time.
2. `prepare_df()` — Constructs a one-row DataFrame from raw inputs, applies MinMaxScaler, clips scaled values to [0, 1] to guard against out-of-range inputs, and selects only the features the model expects.
3. `calculate_credit_score()` — Manually computes the logit, applies sigmoid to get default probability, derives the credit score, and assigns a rating.
4. `predict()` — Orchestrates the above two functions and returns `(default_probability, credit_score, rating)`.

---

## Installation

```bash
# Clone the repository and navigate to the app folder
cd app

# Create and activate a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

---

## Running the Application

```bash
# From inside the app/ directory
streamlit run main.py
```

The app will open in the browser at `http://localhost:8501`.

---

## Input Fields and Valid Ranges

The inputs must stay within the ranges the scaler was trained on to produce reliable predictions. The application enforces these via `min_value` / `max_value` constraints where applicable.

| Field | Type | Notes |
|---|---|---|
| Age | Integer | 18 to 100 |
| Income | Integer | Annual income in rupees |
| Loan Amount | Integer | Total loan amount in rupees |
| Loan Tenure | Integer (months) | Duration of the loan |
| Avg DPD | Integer | Average days past due per delinquent month; max 10 |
| Delinquency Ratio | Integer (0-100) | Percentage of loan tenure spent delinquent |
| Credit Utilization Ratio | Integer (0-100) | Percentage of available credit being used |
| Open Loan Accounts | Integer (1-4) | Number of currently active loan accounts |
| Residence Type | Categorical | Owned, Rented, or Mortgage |
| Loan Purpose | Categorical | Education, Home, Auto, or Personal |
| Loan Type | Categorical | Secured or Unsecured |

---

## Dependencies

All packages and their pinned versions are listed in `requirements.txt`:

```
streamlit==1.54.0
pandas==2.3.3
numpy==2.4.1
scikit-learn==1.8.0
seaborn==0.13.2
matplotlib==3.10.8
statsmodels==0.14.6
xgboost==3.1.3
imbalanced-learn==0.14.1
optuna==4.7.0
joblib==1.5.3
```
