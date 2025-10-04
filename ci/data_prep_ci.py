"""
Load raw tourism.csv from your HF Dataset repo, clean/split, and upload train/test
back to the same dataset.

CI env (GitHub Actions):
- HF_TOKEN       : Write token (GitHub secret)
- DATASET_REPO   : e.g. 'gauravguha/visit-with-us-wellness-dataset' (optional; has a default)
"""

import os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

DATASET_REPO = os.environ.get("DATASET_REPO", "gauravguha/visit-with-us-wellness-dataset")
RAW_URL = f"https://huggingface.co/datasets/{DATASET_REPO}/raw/main/tourism.csv"

def load_raw():
    print(">> Loading raw CSV from:", RAW_URL)
    df = pd.read_csv(RAW_URL)
    return df

def clean_split(df: pd.DataFrame):
    df = df.copy()

    # Drop index-like column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Strip string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    # Coerce numerics
    num_like = [
        "Age","CityTier","DurationOfPitch","NumberOfPersonVisiting","NumberOfFollowups",
        "PreferredPropertyStar","NumberOfTrips","PitchSatisfactionScore","NumberOfChildrenVisiting",
        "MonthlyIncome","ProdTaken"
    ]
    for c in num_like:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Target as int 0/1
    df["ProdTaken"] = df["ProdTaken"].fillna(0).astype(int)

    # Guard negatives then impute
    neg_guard = [
        "Age","DurationOfPitch","NumberOfPersonVisiting","NumberOfFollowups",
        "PreferredPropertyStar","NumberOfTrips","PitchSatisfactionScore","NumberOfChildrenVisiting",
        "MonthlyIncome"
    ]
    for c in neg_guard:
        if c in df.columns:
            df.loc[df[c] < 0, c] = np.nan

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Stratified split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["ProdTaken"]
    )

    return train_df, test_df

def save_local(train_df, test_df):
    os.makedirs("artifacts", exist_ok=True)
    train_p = "artifacts/train.csv"
    test_p  = "artifacts/test.csv"
    train_df.to_csv(train_p, index=False)
    test_df.to_csv(test_p, index=False)
    print(">> Saved:", train_p, test_p)
    return train_p, test_p

def push_to_dataset(train_p, test_p):
    token = os.environ.get("HF_TOKEN")
    assert token, "Missing HF_TOKEN in environment"
    api = HfApi(token=token)

    print(">> Uploading to dataset repo:", DATASET_REPO)
    api.upload_file(
        path_or_fileobj=train_p,
        path_in_repo="train.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="CI: update train.csv"
    )
    api.upload_file(
        path_or_fileobj=test_p,
        path_in_repo="test.csv",
        repo_id=DATASET_REPO,
        repo_type="dataset",
        commit_message="CI: update test.csv"
    )
    print("✅ Uploaded train.csv and test.csv")

def main():
    df = load_raw()
    train_df, test_df = clean_split(df)
    train_p, test_p = save_local(train_df, test_df)
    push_to_dataset(train_p, test_p)

if __name__ == "__main__":
    main()
