"""
Match prediction helper
"""

import os
from typing import Dict

import joblib
import numpy as np


class MatchPredictor:
    def __init__(self, model_path: str):
        model_data = joblib.load(model_path)
        self.model = model_data.get("calibrated_model") or model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_columns = model_data["feature_columns"]

    def predict_match(self, features: Dict) -> Dict:
        row = [features.get(col, 0.0) for col in self.feature_columns]
        X = self.scaler.transform([row])
        probabilities = self.model.predict_proba(X)[0]

        draw_margin = float(os.getenv("DRAW_MARGIN", "0.08"))
        draw_floor = float(os.getenv("DRAW_FLOOR", "0.20"))
        best_idx = int(np.argmax(probabilities))
        draw_prob = float(probabilities[1])
        max_prob = float(np.max(probabilities))
        if draw_prob >= draw_floor and draw_prob >= (max_prob - draw_margin):
            prediction = 1
        else:
            prediction = best_idx

        reverse_mapping = {0: -1, 1: 0, 2: 1}
        return {
            "prediction": reverse_mapping[prediction],
            "probabilities": probabilities.tolist(),
        }
