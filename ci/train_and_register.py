# ci/train_and_register.py
# Trains the RF model, logs to MLflow, and registers the model to HF Model Hub.
# Expects env vars: HF_TOKEN, DATASET_REPO, MODEL_REPO, MLFLOW_TRACKING_URI (optional)

import os
import pandas as pd
from joblib import dump
from huggingface_hub import HfApi

# sklearn / mlflow
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
)

import mlflow, mlflow.sklearn

# --------------------
# Config from env
# --------------------
HF_TOKEN      = os.environ.get("HF_TOKEN")
DATASET_REPO  = os.environ.get("DATASET_REPO")   # e.g., gauravguha/visit-with-us-wellness-dataset
MODEL_REPO    = os.environ.get("MODEL_REPO")     # e.g., gauravguha/visit-with-us-wellness-model
TRACKING_URI  = os.environ.get("MLFLOW_TRACKING_URI", "file:mlruns")

assert HF_TOKEN, "HF_TOKEN missing"
assert DATASET_REPO, "DATASET_REPO missing"
assert MODEL_REPO, "MODEL_REPO missing"

# --------------------
# MLflow init
# --------------------
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment("wellness-predictor")
mlflow.sklearn.autolog(silent=True, log_models=True)

# --------------------
# Load train/test from HF dataset (prepared by the data-prep job)
# --------------------
RAW_TRAIN = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/train.csv"
RAW_TEST  = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/test.csv"

train_df = pd.read_csv(RAW_TRAIN)
test_df  = pd.read_csv(RAW_TEST)

TARGET = "ProdTaken"
drop_cols = [c for c in ["CustomerID", TARGET] if c in train_df.columns]

X_train = train_df.drop(columns=drop_cols, errors="ignore")
y_train = train_df[TARGET].astype(int)
X_test  = test_df.drop(columns=drop_cols, errors="ignore")
y_test  = test_df[TARGET].astype(int)

cat_cols = X_train.select_dtypes(include="object").columns.tolist()
num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

rf = RandomForestClassifier(class_weight="balanced", random_state=42)
pipe = Pipeline([("preprocess", preprocess), ("model", rf)])

grid = {
    "model__n_estimators": [200, 400],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2],
    "model__max_depth": [None, 10],
}

# --------------------
# Train + log with MLflow
# --------------------
with mlflow.start_run(run_name="rf-grid"):
    gs = GridSearchCV(pipe, grid, cv=3, scoring="roc_auc", n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred  = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_proba)
    test_acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    # extra metrics + artifact
    mlflow.log_metric("test_auc", float(test_auc))
    mlflow.log_metric("test_accuracy", float(test_acc))
    mlflow.log_metric("test_precision", float(prec))
    mlflow.log_metric("test_recall", float(rec))
    mlflow.log_metric("test_f1", float(f1))

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]).to_csv("confusion_matrix.csv", index=True)
    mlflow.log_artifact("confusion_matrix.csv")

    # Save model artifact locally (also logged by autolog)
    dump(best, "model_pipeline.joblib")

# --------------------
# Push the model to the HF Model Hub
# --------------------
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)
api.upload_file(
    path_or_fileobj="model_pipeline.joblib",
    path_in_repo="model_pipeline.joblib",
    repo_id=MODEL_REPO,
    repo_type="model",
)
print("✅ Model pushed to HF Model Hub:", MODEL_REPO)

