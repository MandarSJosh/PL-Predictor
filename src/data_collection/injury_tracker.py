"""
Injury feature stub (lightweight)
"""

import logging
import os
from typing import Optional

import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InjuryTracker:
    def __init__(self, fetch_live: Optional[bool] = None):
        self.fetch_live = fetch_live if fetch_live is not None else True
        self.api_key = os.getenv("API_FOOTBALL_KEY")

    def add_injury_features(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        # If no live key, use defaults
        for col in [
            "home_key_players_injured",
            "away_key_players_injured",
            "home_total_injured",
            "away_total_injured",
            "home_key_players_suspended",
            "away_key_players_suspended",
            "home_injury_severity",
            "away_injury_severity",
            "injury_difference",
        ]:
            if col not in matches_df.columns:
                matches_df[col] = 0.0
        return matches_df
