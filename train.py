import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.lead_scoring.preprocessing import LeadPreprocessor
from src.lead_scoring.model import build_stacking_model


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv("data/ExtraaLearn.csv")

X = df.drop(columns=["status"])
y = df["status"]


# --------------------------------------------------
# 2. Train/test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=41,
    stratify=y,
)


# --------------------------------------------------
# 3. Fit preprocessing ONLY on training data
# --------------------------------------------------

preprocessor = LeadPreprocessor()

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


# --------------------------------------------------
# 4. Build model
# --------------------------------------------------

model = build_stacking_model()


# --------------------------------------------------
# 5. Train model
# --------------------------------------------------

print("Training model...")

model.fit(
    X_train_processed,
    y_train,
)
artifacts_dir = Path("artifacts")
artifacts_dir.mkdir(exist_ok=True)

joblib.dump(
    preprocessor,
    artifacts_dir / "preprocessor.pkl",
)

joblib.dump(
    model,
    artifacts_dir / "model.pkl",
)

with open(
    artifacts_dir / "threshold.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        {
            "threshold": 0.411
        },
        file,
        indent=4,
    )

print("\nArtifacts saved:")
print("- artifacts/preprocessor.pkl")
print("- artifacts/model.pkl")
print("- artifacts/threshold.json")

print("Training complete.")


# --------------------------------------------------
# 6. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test_processed)
y_proba = model.predict_proba(X_test_processed)[:, 1]


# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------

print("\nModel Performance")
print("-----------------")

print(
    f"Accuracy : {accuracy_score(y_test, y_pred):.4f}"
)

print(
    f"Precision: {precision_score(y_test, y_pred):.4f}"
)

print(
    f"Recall   : {recall_score(y_test, y_pred):.4f}"
)

print(
    f"F1 Score : {f1_score(y_test, y_pred):.4f}"
)

print(
    f"ROC AUC  : {roc_auc_score(y_test, y_proba):.4f}"
)