import os
from huggingface_hub import HfApi

# Read Space repo from env (set at top of pipeline.yml)
SPACE_REPO = os.environ.get("SPACE_REPO", "gauravguha/visit-with-us-wellness-app")

# Local → path in Space repo
FILES = {
    "deployment/Dockerfile": "Dockerfile",
    "deployment/app.py": "app.py",
    "deployment/requirements.txt": "requirements.txt",
}

def main():
    token = os.environ.get("HF_TOKEN")
    assert token, "Missing HF_TOKEN environment variable"

    # Sanity check local paths
    missing = [p for p in FILES.keys() if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Local files not found: {missing}. CWD={os.getcwd()}")

    api = HfApi(token=token)
    for local, dest in FILES.items():
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=dest,
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message=f"CI: update {dest}"
        )
    print(f"✅ Pushed deployment files to Space: {SPACE_REPO}")

if __name__ == "__main__":
    main()
