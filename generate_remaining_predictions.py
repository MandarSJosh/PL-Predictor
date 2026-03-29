"""
Generate predictions for remaining fixtures and save to CSV.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering.features import FeatureEngineer
from src.models.predictor import MatchPredictor
from src.utils.teams import normalize_team_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    data_dir = Path("data")
    remaining_path = data_dir / "remaining_fixtures.csv"
    raw_path = data_dir / "raw_matches.csv"
    model_path = Path("models/best_model.pkl")

    if not remaining_path.exists():
        raise FileNotFoundError("Missing data/remaining_fixtures.csv")
    if not raw_path.exists():
        raise FileNotFoundError("Missing data/raw_matches.csv")
    if not model_path.exists():
        raise FileNotFoundError("Missing models/best_model.pkl")

    remaining = pd.read_csv(remaining_path)
    remaining["home_team"] = remaining["home_team"].astype(str).map(normalize_team_name)
    remaining["away_team"] = remaining["away_team"].astype(str).map(normalize_team_name)

    base_matches = pd.read_csv(raw_path)
    fixtures = remaining[["date", "home_team", "away_team"]].copy()
    fixtures["season"] = remaining.get("season")
    fixtures["home_score"] = np.nan
    fixtures["away_score"] = np.nan
    all_matches = pd.concat([base_matches, fixtures], ignore_index=True)

    feature_engineer = FeatureEngineer()
    features = feature_engineer.create_match_features(
        all_matches, include_h2h=True, drop_na_target=False
    )

    predictor = MatchPredictor(str(model_path))
    fixture_features = features.tail(len(fixtures)).copy()
    feature_cols = predictor.feature_columns
    X = fixture_features[feature_cols].fillna(0).values
    proba = predictor.model.predict_proba(predictor.scaler.transform(X))

    results = []
    reverse_mapping = {0: "Away Win", 1: "Draw", 2: "Home Win"}
    for idx, probs in enumerate(proba):
        draw_prob = float(probs[1])
        max_prob = float(np.max(probs))
        best_idx = int(np.argmax(probs))
        draw_margin = float(os.getenv("DRAW_MARGIN", "0.06"))
        draw_floor = float(os.getenv("DRAW_FLOOR", "0.15"))
        if draw_prob >= draw_floor and draw_prob >= (max_prob - draw_margin):
            pred_idx = 1
        else:
            pred_idx = best_idx

        row = remaining.iloc[idx]
        results.append(
            {
                "fixture_id": row.get("fixture_id"),
                "date": row.get("date"),
                "matchweek": row.get("matchweek"),
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
                "predicted_outcome": reverse_mapping[pred_idx],
                "home_win_prob": float(probs[2]),
                "draw_prob": float(probs[1]),
                "away_win_prob": float(probs[0]),
            }
        )

    output = data_dir / "remaining_predictions.csv"
    pd.DataFrame(results).to_csv(output, index=False)
    logger.info(f"Saved {len(results)} predictions to {output}")


if __name__ == "__main__":
    main()
