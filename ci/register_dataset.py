"""
Idempotently ensure the HF Dataset repo exists.
CI env:
  - HF_TOKEN  (GitHub secret)
  - DATASET_REPO (defaults to your dataset)
"""
import os
from huggingface_hub import HfApi

def main():
    token = os.environ.get("HF_TOKEN")
    assert token, "Missing HF_TOKEN"
    dataset_repo = os.environ.get("DATASET_REPO", "gauravguha/visit-with-us-wellness-dataset")

    api = HfApi(token=token)
    api.create_repo(repo_id=dataset_repo, repo_type="dataset", exist_ok=True)
    print("✅ Dataset repo is ready:", dataset_repo)

if __name__ == "__main__":
    main()

