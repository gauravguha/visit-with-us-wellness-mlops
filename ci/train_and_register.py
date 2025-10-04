"""
Train the model from HF dataset train/test, evaluate, and push the model
artifact to the Hugging Face Model Hub.

CI env (GitHub Actions):
- HF_TOKEN     : Write token (GitHub secret)
- MODEL_REPO   : e.g. 'gauravguha/visit-with-us-wellness-model' (optional; default below)
- DATASET_REPO : e.g. 'gauravguha/visit-with-us-wellness-dataset' (optional; default below)
"""

import os, json
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from huggingface_hub import HfApi

DATASET_REPO = os.environ.get("DATASET_REPO", "gauravguha/visit-with-us-wellness-dataset")
MODEL_REPO   = os.environ.get("MODEL_REPO",   "gauravguha/visit-with-us-wellness-model")

TRAIN_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/train.csv"
TEST_URL  = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/test.csv"

def load_data():
    train_df = pd.read_csv(TRAIN_URL)
    test_df  = pd.read_csv(TEST_URL)
    return train_df, test_df

def build_pipeline(train_df):
    TARGET = "ProdTaken"
    DROP = ["CustomerID"]
    X_train = train_df.drop(columns=[TARGET] + DROP, errors="ignore")

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    clf = RandomForestClassifier(
        random_state=42,
        n_estimators=400,
        class_weight="balanced"
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", clf)])
    return pipe

def evaluate(pipe, test_df):
    TARGET = "ProdTaken"
    DROP = ["CustomerID"]
    X_test = test_df.drop(columns=[TARGET] + DROP, errors="ignore")
    y_test = test_df[TARGET].astype(int)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_proba)
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="binary", zero_division=0)

    return {
        "test_auc": round(float(auc), 4),
        "test_accuracy": round(float(acc), 4),
        "test_precision": round(float(prec), 4),
        "test_recall": round(float(rec), 4),
        "test_f1": round(float(f1), 4),
    }

def main():
    print(">> Loading data …")
    train_df, test_df = load_data()

    print(">> Building pipeline …")
    pipe = build_pipeline(train_df)

    print(">> Training …")
    TARGET = "ProdTaken"
    X_train = train_df.drop(columns=[TARGET, "CustomerID"], errors="ignore")
    y_train = train_df[TARGET].astype(int)
    pipe.fit(X_train, y_train)

    print(">> Evaluating …")
    metrics = evaluate(pipe, test_df)
    print("Metrics:", metrics)

    # Save artifacts
    os.makedirs("artifacts", exist_ok=True)
    model_path = "artifacts/model_pipeline.joblib"
    dump(pipe, model_path)
    with open("artifacts/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(">> Uploading model to HF Model Hub …")
    token = os.environ.get("HF_TOKEN")
    assert token, "Missing HF_TOKEN"
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="model_pipeline.joblib",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="CI: update model_pipeline.joblib"
    )

    print("✅ Pushed model to:", MODEL_REPO)
    print("Final metrics:", metrics)

if __name__ == "__main__":
    main()
