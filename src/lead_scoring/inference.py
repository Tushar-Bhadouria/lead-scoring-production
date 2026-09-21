import json
from pathlib import Path

import joblib
import pandas as pd


class LeadPredictor:
    """
    Loads trained ML artifacts and performs lead prediction.
    """

    def __init__(self, artifacts_dir="artifacts"):
        artifacts_dir = Path(artifacts_dir)

        self.preprocessor = joblib.load(
            artifacts_dir / "preprocessor.pkl"
        )

        self.model = joblib.load(
            artifacts_dir / "model.pkl"
        )

        with open(
            artifacts_dir / "threshold.json",
            "r",
            encoding="utf-8",
        ) as file:
            threshold_config = json.load(file)

        self.threshold = threshold_config["threshold"]

    def predict(self, lead):
        """
        Predict whether a lead should be classified as positive.

        Parameters
        ----------
        lead : dict
            Raw lead information.

        Returns
        -------
        dict
            Prediction probability, threshold, and prediction.
        """

        # Convert raw dictionary into DataFrame
        lead_df = pd.DataFrame([lead])

        # Apply the fitted preprocessing pipeline
        processed_lead = self.preprocessor.transform(
            lead_df
        )

        # Generate probability for positive class
        probability = self.model.predict_proba(
            processed_lead
        )[0, 1]

        # Apply business threshold
        prediction = int(
            probability >= self.threshold
        )

        return {
            "conversion_probability": float(probability),
            "threshold": float(self.threshold),
            "prediction": prediction,
        }