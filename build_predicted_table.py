"""
Build predicted league table from remaining fixtures and current table.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.feature_engineering.features import FeatureEngineer
from src.models.predictor import MatchPredictor
from src.utils.teams import load_teams_list, normalize_team_name

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _normalize_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    if "team" not in df.columns:
        for candidate in ["club", "name", "team_name"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "team"})
                break
    if "points" not in df.columns:
        for candidate in ["pts"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "points"})
                break
    if "gd" not in df.columns:
        for candidate in ["goal_diff", "goal_difference"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "gd"})
                break
    df["team"] = df["team"].astype(str).map(normalize_team_name)
    return df


def _apply_result(stats: pd.Series, outcome: int, gf: int, ga: int) -> pd.Series:
    # outcome: 1 home win, 0 draw, -1 away win
    stats["pl"] += 1
    stats["gf"] += gf
    stats["ga"] += ga
    stats["gd"] += gf - ga
    if outcome == 1:
        stats["w"] += 1
        stats["points"] += 3
    elif outcome == -1:
        stats["l"] += 1
    else:
        stats["d"] += 1
        stats["points"] += 1
    return stats


def _zone_label(pos: int) -> str:
    if pos == 1:
        return "Champion"
    if pos <= 4:
        return "UCL"
    if pos == 5:
        return "UEL"
    if pos == 6:
        return "UECL"
    if pos >= 18:
        return "Relegation"
    return "Mid"


def _delta_label(delta: int) -> str:
    if delta > 0:
        return f"▲ {int(delta)}"
    if delta < 0:
        return f"▼ {abs(int(delta))}"
    return "—"


def main() -> None:
    data_dir = Path("data")
    remaining_path = data_dir / "remaining_fixtures.csv"
    current_path = data_dir / "current_table.csv"
    model_path = Path("models/best_model.pkl")
    raw_path = data_dir / "raw_matches.csv"

    if not remaining_path.exists():
        raise FileNotFoundError("Missing data/remaining_fixtures.csv")
    if not current_path.exists():
        raise FileNotFoundError("Missing data/current_table.csv")
    if not model_path.exists():
        raise FileNotFoundError("Missing models/best_model.pkl")
    if not raw_path.exists():
        raise FileNotFoundError("Missing data/raw_matches.csv")

    remaining = pd.read_csv(remaining_path)
    remaining["home_team"] = remaining["home_team"].astype(str).map(normalize_team_name)
    remaining["away_team"] = remaining["away_team"].astype(str).map(normalize_team_name)

    current = _normalize_table(pd.read_csv(current_path))
    teams = load_teams_list()
    current = current[current["team"].isin(teams)]
    current = current.set_index("team")
    if "points" not in current.columns:
        current["points"] = 0.0
    for col in ["pl", "w", "d", "l", "gf", "ga", "gd"]:
        if col not in current.columns:
            current[col] = 0.0

    base_matches = pd.read_csv(raw_path)
    base_matches["home_team"] = base_matches["home_team"].astype(str).map(normalize_team_name)
    base_matches["away_team"] = base_matches["away_team"].astype(str).map(normalize_team_name)
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
    draw_margin = float(os.getenv("DRAW_MARGIN", "0.08"))
    draw_floor = float(os.getenv("DRAW_FLOOR", "0.20"))
    predictions = []
    for probs in proba:
        draw_prob = float(probs[1])
        max_prob = float(np.max(probs))
        best_idx = int(np.argmax(probs))
        if draw_prob >= draw_floor and draw_prob >= (max_prob - draw_margin):
            pred_idx = 1
        else:
            pred_idx = best_idx
        predictions.append(pred_idx)

    reverse_mapping = {0: -1, 1: 0, 2: 1}
    outcomes = [reverse_mapping[p] for p in predictions]

    table = current.copy()
    # Track form from historic + predicted results
    form_map = {team: [] for team in table.index}
    for _, row in base_matches.iterrows():
        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            continue
        home = row["home_team"]
        away = row["away_team"]
        if row["home_score"] == row["away_score"]:
            form_map.get(home, []).append("D")
            form_map.get(away, []).append("D")
        elif row["home_score"] > row["away_score"]:
            form_map.get(home, []).append("W")
            form_map.get(away, []).append("L")
        else:
            form_map.get(home, []).append("L")
            form_map.get(away, []).append("W")

    for (home, away), outcome in zip(
        remaining[["home_team", "away_team"]].values, outcomes
    ):
        # Minimal scoreline proxy for GF/GA to support GD/WDL
        if outcome == 1:
            home_gf, away_gf = 1, 0
        elif outcome == -1:
            home_gf, away_gf = 0, 1
        else:
            home_gf, away_gf = 1, 1

        if home in table.index:
            table.loc[home] = _apply_result(
                table.loc[home], outcome, home_gf, away_gf
            )
            form_map.get(home, []).append("W" if outcome == 1 else "D" if outcome == 0 else "L")
        if away in table.index:
            table.loc[away] = _apply_result(
                table.loc[away], -outcome, away_gf, home_gf
            )
            form_map.get(away, []).append("W" if outcome == -1 else "D" if outcome == 0 else "L")

    table = table.reset_index().rename(columns={"index": "team"})
    table["form_last5"] = table["team"].map(lambda t: "".join(form_map.get(t, [])[-5:]))
    table = table.sort_values(["points", "gd", "gf"], ascending=False).reset_index(drop=True)
    table["position"] = range(1, len(table) + 1)

    current_sorted = (
        current.reset_index()
        .rename(columns={"index": "team"})
        .sort_values(["points", "gd", "gf"], ascending=False)
        .reset_index(drop=True)
    )
    current_sorted["position"] = range(1, len(current_sorted) + 1)
    pos_map = dict(zip(current_sorted["team"], current_sorted["position"]))
    table["pos_change"] = table["team"].map(pos_map).fillna(table["position"]) - table["position"]
    table["delta"] = table["pos_change"].apply(_delta_label)
    table["zone"] = table["position"].apply(lambda p: _zone_label(int(p)))

    output = data_dir / "predicted_table.csv"
    table[
        [
            "position",
            "team",
            "pl",
            "w",
            "d",
            "l",
            "gf",
            "ga",
            "gd",
            "points",
            "form_last5",
            "zone",
            "delta",
        ]
    ].to_csv(output, index=False)
    logger.info(f"Saved predicted table to {output}")


if __name__ == "__main__":
    main()
