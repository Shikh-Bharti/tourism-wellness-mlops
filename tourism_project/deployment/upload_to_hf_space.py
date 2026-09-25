"""
Hosting
--------
Creates (or updates) a Hugging Face Space and pushes the Docker-based
Streamlit app (Dockerfile, app.py, requirements.txt) to it.
"""

import os
from huggingface_hub import HfApi, create_repo

# ---- EDIT THIS LINE FOR YOUR OWN PROJECT ------------------------------------
HF_USERNAME = "Shirit12"                       # <-- your HF username
SPACE_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-app"
# -----------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

create_repo(
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    space_sdk="docker",
    private=False,
    exist_ok=True,
    token=HF_TOKEN,
)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=SPACE_REPO_ID,
    repo_type="space",
    token=HF_TOKEN,
)

print(f"App deployed to: https://huggingface.co/spaces/{SPACE_REPO_ID}")
