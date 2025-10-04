import os
from huggingface_hub import HfApi

# EDIT if your Space ID is different
SPACE_REPO = "gauravguha/visit-with-us-wellness-app"

# Local -> destination paths in the Space repo
FILES = {
    "tourism_project/deployment/Dockerfile": "Dockerfile",
    "tourism_project/deployment/app.py": "app.py",
    "tourism_project/deployment/requirements.txt": "requirements.txt",
}

def main():
    token = os.environ.get("HF_TOKEN")
    assert token, "Set HF_TOKEN environment variable with a WRITE token (export HF_TOKEN=...)."
    api = HfApi(token=token)
    for local, dest in FILES.items():
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=dest,
            repo_id=SPACE_REPO,
            repo_type="space",
            commit_message=f"Update {dest} via script"
        )
    print("✅ Pushed all deployment files to Space:", SPACE_REPO)

if __name__ == "__main__":
    main()
