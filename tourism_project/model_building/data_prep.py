"""
Data Preparation
-----------------
Pulls the raw dataset back down from the Hugging Face dataset repo, cleans it,
performs the train/test split, and pushes the four resulting CSVs
(X_train, X_test, y_train, y_test) back up to the SAME dataset repo as new
files, so the training stage always works off a versioned, reproducible split.
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download, HfApi

# ---- EDIT THIS LINE FOR YOUR OWN PROJECT ------------------------------------
HF_USERNAME = "Shirit12"                       # <-- your HF username
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package"
# -----------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)

# 1. Pull the raw file that data_registration.py uploaded
raw_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="tourism.csv", repo_type="dataset")
df = pd.read_csv(raw_path)

# 2. Cleaning
df = df.drop(columns=["Unnamed: 0"], errors="ignore")   # stray index column, if present
df = df.drop(columns=["CustomerID"], errors="ignore")   # identifier, not predictive
df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})          # fix data-entry typo
df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})
df = df.dropna(subset=["ProdTaken"])                    # safety net: drop rows with no target

target = "ProdTaken"
X = df.drop(columns=[target])
y = df[target]

# 3. Stratified train/test split (target is imbalanced: ~19% positive class)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

os.makedirs("tourism_project/data", exist_ok=True)
X_train.to_csv("tourism_project/data/X_train.csv", index=False)
X_test.to_csv("tourism_project/data/X_test.csv", index=False)
y_train.to_csv("tourism_project/data/y_train.csv", index=False)
y_test.to_csv("tourism_project/data/y_test.csv", index=False)

# 4. Push the processed splits back to the Hub
for fname in ["X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv"]:
    api.upload_file(
        path_or_fileobj=f"tourism_project/data/{fname}",
        path_in_repo=fname,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        token=HF_TOKEN,
    )

print(f"Data preparation complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Splits pushed to: https://huggingface.co/datasets/{DATASET_REPO_ID}")
