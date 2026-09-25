"""
Model Training and Registration with Experimentation Tracking
----------------------------------------------------------------
Pulls the prepared train/test splits, builds a preprocessing + model pipeline,
tunes two candidate algorithms with GridSearchCV, logs every run's params and
metrics to MLflow, selects the best model by F1-score, and pushes the winning
pipeline to a Hugging Face *model* repository.
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from huggingface_hub import hf_hub_download, HfApi, create_repo

# ---- EDIT THIS LINE FOR YOUR OWN PROJECT ------------------------------------
HF_USERNAME = "Shirit12"                       # <-- your HF username
DATASET_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package"
MODEL_REPO_ID = f"{HF_USERNAME}/tourism-wellness-package-model"
# -----------------------------------------------------------------------------

HF_TOKEN = os.getenv("HF_TOKEN")
api = HfApi(token=HF_TOKEN)
create_repo(repo_id=MODEL_REPO_ID, repo_type="model", private=False, exist_ok=True, token=HF_TOKEN)

# 1. Pull the prepared splits from the Hub
X_train = pd.read_csv(hf_hub_download(DATASET_REPO_ID, "X_train.csv", repo_type="dataset"))
X_test = pd.read_csv(hf_hub_download(DATASET_REPO_ID, "X_test.csv", repo_type="dataset"))
y_train = pd.read_csv(hf_hub_download(DATASET_REPO_ID, "y_train.csv", repo_type="dataset")).squeeze()
y_test = pd.read_csv(hf_hub_download(DATASET_REPO_ID, "y_test.csv", repo_type="dataset")).squeeze()

num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=[np.number]).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore")),
    ]), cat_cols),
])

# 2. Candidate models + hyperparameter grids
candidates = {
    "RandomForest": (
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        {"model__n_estimators": [100, 200], "model__max_depth": [5, 10, None]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1]},
    ),
}

mlflow.set_experiment("tourism-wellness-package")

best_pipeline, best_score, best_name, best_metrics = None, -1, None, None

for name, (estimator, grid) in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    with mlflow.start_run(run_name=name):
        gs = GridSearchCV(pipe, grid, cv=3, scoring="f1", n_jobs=-1)
        gs.fit(X_train, y_train)

        fitted = gs.best_estimator_
        preds = fitted.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
        }

        mlflow.log_params(gs.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(fitted, "model", serialization_format="pickle")

        print(f"{name}: best_params={gs.best_params_}  metrics={metrics}")

        if metrics["f1"] > best_score:
            best_pipeline, best_score, best_name, best_metrics = fitted, metrics["f1"], name, metrics

print(f"\nBest model: {best_name}  (F1={best_score:.4f})  {best_metrics}")

# 3. Save + register the winning pipeline
os.makedirs("tourism_project/model_building", exist_ok=True)
model_path = "tourism_project/model_building/best_model.joblib"
joblib.dump(best_pipeline, model_path)

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="best_model.joblib",
    repo_id=MODEL_REPO_ID,
    repo_type="model",
    token=HF_TOKEN,
)

print(f"Best model ({best_name}) registered at: https://huggingface.co/{MODEL_REPO_ID}")
