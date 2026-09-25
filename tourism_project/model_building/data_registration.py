"""
Data Registration
------------------
Uploads the raw tourism.csv file to a Hugging Face Hub *dataset* repository so
that every later pipeline stage (data prep, training, CI/CD) pulls the data
from one single, versioned source of truth instead of a local file.
"""

import os
from huggingface_hub import HfApi, create_repo

# ---- EDIT THESE TWO LINES FOR YOUR OWN PROJECT -----------------------------
HF_USERNAME = "Shirit12"                       # <-- your HF username
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package"
# -----------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")

api = HfApi(token=HF_TOKEN)

# Create the dataset repo on the Hub if it doesn't already exist
create_repo(
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    private=False,
    exist_ok=True,
    token=HF_TOKEN,
)

api.upload_file(
    path_or_fileobj="tourism_project/data/tourism.csv",
    path_in_repo="tourism.csv",
    repo_id=DATASET_REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
)

print(f"Raw dataset registered at: https://huggingface.co/datasets/{DATASET_REPO_ID}")
