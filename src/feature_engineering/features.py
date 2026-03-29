"""
Feature Engineering Pipeline for Premier League Match Predictions
Creates comprehensive features from match and team statistics
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features for match prediction"""

    def __init__(self):
        self.team_stats_cache = {}
        self.form_window = 5  # Last 5 matches for form
        self.stats_window = 10  # Last 10 matches for rolling stats

    def create_match_features(
        self,
        matches_df: pd.DataFrame,
        team_stats_df: Optional[pd.DataFrame] = None,
        include_h2h: bool = True,
        drop_na_target: bool = True,
    ) -> pd.DataFrame:
        """
        Create comprehensive features for each match
        """
        logger.info("Starting feature engineering...")
        df = matches_df.copy()

        # Clean obvious bad rows where team columns are actually kickoff times
        if "home_team" in df.columns and "away_team" in df.columns:
            time_pattern = r"^\d{1,2}:\d{2}$"
            home_series = df["home_team"].astype(str).str.strip()
            away_series = df["away_team"].astype(str).str.strip()
            bad_rows = home_series.str.match(time_pattern) | away_series.str.match(
                time_pattern
            )
            if bad_rows.any():
                logger.warning(
                    f"Dropping {bad_rows.sum()} rows with time-like team names"
                )
                df = df[~bad_rows].copy()

        # Ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")

        # Create target variable (home win = 1, draw = 0, away win = -1)
        df["target"] = df.apply(self._create_target, axis=1)

        # Calculate rolling statistics for each team
        df = self._add_rolling_stats(df)

        # Add form features (recent performance)
        df = self._add_form_features(df)

        # Add head-to-head features (optional - can be slow)
        if include_h2h:
            df = self._add_h2h_features(df)
        else:
            logger.info("Skipping head-to-head features (use include_h2h=True to enable)")
            df["h2h_home_wins"] = 0.0
            df["h2h_away_wins"] = 0.0
            df["h2h_draws"] = 0.0
            df["h2h_home_goals"] = 0.0
            df["h2h_away_goals"] = 0.0

        # Add team strength features
        try:
            df = self._add_team_strength_features(df)
        except MemoryError:
            logger.warning("Team strength features skipped due to memory constraints")
            df["home_cumulative_points"] = 0.0
            df["away_cumulative_points"] = 0.0
            df["home_overall_win_rate"] = 0.5
            df["away_overall_win_rate"] = 0.5
            df["home_strength_prior"] = 0.5
            df["away_strength_prior"] = 0.5
            df["strength_prior_diff"] = 0.0

        # Add momentum features
        try:
            df = self._add_momentum_features(df)
        except MemoryError:
            logger.warning("Momentum features skipped due to memory constraints")
            df["home_momentum"] = 0.0
            df["away_momentum"] = 0.0

        # Add positional features (league position)
        try:
            df = self._add_position_features(df)
        except MemoryError:
            logger.warning("Position features skipped due to memory constraints")
            df["home_position"] = 10.0
            df["away_position"] = 10.0
            df["position_diff"] = 0.0

        # Add last-10 form features (overall, includes opponent difficulty)
        try:
            df = self._add_form10_features(df)
        except MemoryError:
            logger.warning("Form-10 features skipped due to memory constraints")
            for prefix in ["home", "away"]:
                df[f"{prefix}_form10_points"] = 0.0
                df[f"{prefix}_form10_ppg"] = 0.0
                df[f"{prefix}_form10_win_rate"] = 0.0
                df[f"{prefix}_form10_draw_rate"] = 0.0
                df[f"{prefix}_form10_loss_rate"] = 0.0
                df[f"{prefix}_form10_goal_diff"] = 0.0
                df[f"{prefix}_form10_gd_per_game"] = 0.0
                df[f"{prefix}_form10_avg_opp_position"] = 10.0
                df[f"{prefix}_form10_avg_opp_strength"] = 0.5

        # Add time-based features
        df = self._add_time_features(df)

        # Add rest/fatigue features (days since last match)
        try:
            df = self._add_rest_features(df)
        except MemoryError:
            logger.warning("Rest features skipped due to memory constraints")
            df["home_days_since_last_match"] = 0.0
            df["away_days_since_last_match"] = 0.0

        # Add advanced features (xG, referee, weather, injuries, manager)
        df = self._add_advanced_features(df)

        # Merge team stats if provided
        if team_stats_df is not None:
            df = self._merge_team_stats(df, team_stats_df)

        # Remove rows with missing critical features
        if drop_na_target:
            df = df.dropna(subset=["home_team", "away_team", "target"])
        else:
            df = df.dropna(subset=["home_team", "away_team"])

        logger.info(f"Feature engineering complete. {len(df)} matches with features.")
        return df

    def _create_target(self, row: pd.Series) -> int:
        """Create target variable: 1 = home win, 0 = draw, -1 = away win"""
        if pd.isna(row.get("home_score")) or pd.isna(row.get("away_score")):
            return np.nan

        if row["home_score"] > row["away_score"]:
            return 1
        if row["home_score"] == row["away_score"]:
            return 0
        return -1

    def _add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling statistics for goals - memory optimized"""
        logger.info("Adding rolling statistics...")

        for prefix in ["home", "away"]:
            df[f"{prefix}_avg_goals_scored"] = 0.0
            df[f"{prefix}_avg_goals_conceded"] = 0.0
            df[f"{prefix}_avg_goal_diff"] = 0.0
            df[f"{prefix}_avg_goals_scored_5"] = 0.0
            df[f"{prefix}_avg_goals_conceded_5"] = 0.0
            df[f"{prefix}_avg_goal_diff_5"] = 0.0

        for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
            teams = df[team_col].unique()
            logger.info(f"Processing {len(teams)} teams for {prefix} rolling stats...")

            for i, team in enumerate(teams):
                if i % 10 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{len(teams)} teams...")

                team_mask = df[team_col] == team
                team_matches = df[team_mask].sort_values("date")
                if len(team_matches) == 0:
                    continue

                if prefix == "home":
                    goals_scored = team_matches["home_score"].fillna(0).values
                    goals_conceded = team_matches["away_score"].fillna(0).values
                else:
                    goals_scored = team_matches["away_score"].fillna(0).values
                    goals_conceded = team_matches["home_score"].fillna(0).values

                goals_scored_series = pd.Series(goals_scored, index=team_matches.index)
                goals_conceded_series = pd.Series(goals_conceded, index=team_matches.index)

                avg_goals_scored = goals_scored_series.shift(1).rolling(
                    window=self.stats_window, min_periods=1
                ).mean()
                avg_goals_conceded = goals_conceded_series.shift(1).rolling(
                    window=self.stats_window, min_periods=1
                ).mean()

                goal_diff = goals_scored - goals_conceded
                goal_diff_series = pd.Series(goal_diff, index=team_matches.index)
                avg_goal_diff = goal_diff_series.shift(1).rolling(
                    window=self.stats_window, min_periods=1
                ).mean()

                avg_goals_scored_5 = goals_scored_series.shift(1).rolling(
                    window=5, min_periods=1
                ).mean()
                avg_goals_conceded_5 = goals_conceded_series.shift(1).rolling(
                    window=5, min_periods=1
                ).mean()
                avg_goal_diff_5 = goal_diff_series.shift(1).rolling(
                    window=5, min_periods=1
                ).mean()

                df.loc[team_matches.index, f"{prefix}_avg_goals_scored"] = avg_goals_scored.fillna(0)
                df.loc[team_matches.index, f"{prefix}_avg_goals_conceded"] = avg_goals_conceded.fillna(0)
                df.loc[team_matches.index, f"{prefix}_avg_goal_diff"] = avg_goal_diff.fillna(0)
                df.loc[team_matches.index, f"{prefix}_avg_goals_scored_5"] = avg_goals_scored_5.fillna(0)
                df.loc[team_matches.index, f"{prefix}_avg_goals_conceded_5"] = avg_goals_conceded_5.fillna(0)
                df.loc[team_matches.index, f"{prefix}_avg_goal_diff_5"] = avg_goal_diff_5.fillna(0)

        return df

    def _add_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add recent form features (last N matches)"""
        logger.info("Adding form features...")

        for prefix in ["home", "away"]:
            df[f"{prefix}_form_points"] = 0.0
            df[f"{prefix}_win_rate"] = 0.0

        for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
            teams = df[team_col].unique()
            logger.info(f"Processing {len(teams)} teams for {prefix} form...")

            for i, team in enumerate(teams):
                if i % 10 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{len(teams)} teams...")

                team_mask = df[team_col] == team
                team_matches = df[team_mask].sort_values("date")
                if len(team_matches) == 0:
                    continue

                is_home = (team_matches["home_team"] == team).values
                h_scores = team_matches["home_score"].fillna(0).values
                a_scores = team_matches["away_score"].fillna(0).values
                valid = team_matches["home_score"].notna().values & team_matches["away_score"].notna().values

                points = np.zeros(len(team_matches))
                home_mask = is_home & valid
                away_mask = ~is_home & valid

                points[home_mask] = np.where(
                    h_scores[home_mask] > a_scores[home_mask], 3,
                    np.where(h_scores[home_mask] == a_scores[home_mask], 1, 0),
                )
                points[away_mask] = np.where(
                    a_scores[away_mask] > h_scores[away_mask], 3,
                    np.where(a_scores[away_mask] == h_scores[away_mask], 1, 0),
                )

                points_series = pd.Series(points, index=team_matches.index)
                form_points = points_series.shift(1).rolling(
                    window=self.form_window, min_periods=1
                ).sum()

                is_win = (points == 3).astype(float)
                is_win_series = pd.Series(is_win, index=team_matches.index)
                win_rate = is_win_series.shift(1).rolling(
                    window=self.form_window, min_periods=1
                ).mean()

                df.loc[team_matches.index, f"{prefix}_form_points"] = form_points.fillna(0)
                df.loc[team_matches.index, f"{prefix}_win_rate"] = win_rate.fillna(0)

        return df

    def _add_form10_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add overall last-10 match form with opponent difficulty"""
        logger.info("Adding last-10 form features...")

        window = 10
        for prefix in ["home", "away"]:
            df[f"{prefix}_form10_points"] = 0.0
            df[f"{prefix}_form10_ppg"] = 0.0
            df[f"{prefix}_form10_win_rate"] = 0.0
            df[f"{prefix}_form10_draw_rate"] = 0.0
            df[f"{prefix}_form10_loss_rate"] = 0.0
            df[f"{prefix}_form10_goal_diff"] = 0.0
            df[f"{prefix}_form10_gd_per_game"] = 0.0
            df[f"{prefix}_form10_avg_opp_position"] = 10.0
            df[f"{prefix}_form10_avg_opp_strength"] = 0.5

        teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
        logger.info(f"Processing {len(teams)} teams for last-10 form...")

        for i, team in enumerate(teams):
            if i % 10 == 0 and i > 0:
                logger.info(f"  Processed {i}/{len(teams)} teams...")

            team_mask = (df["home_team"] == team) | (df["away_team"] == team)
            team_matches = df[team_mask].sort_values("date")
            if len(team_matches) == 0:
                continue

            idx = team_matches.index
            is_home = (team_matches["home_team"] == team).values
            h_scores = team_matches["home_score"].fillna(0).values
            a_scores = team_matches["away_score"].fillna(0).values
            valid = team_matches["home_score"].notna().values & team_matches["away_score"].notna().values

            points = np.zeros(len(team_matches))
            gd = np.zeros(len(team_matches))

            home_mask = is_home & valid
            away_mask = ~is_home & valid

            points[home_mask] = np.where(
                h_scores[home_mask] > a_scores[home_mask], 3,
                np.where(h_scores[home_mask] == a_scores[home_mask], 1, 0),
            )
            points[away_mask] = np.where(
                a_scores[away_mask] > h_scores[away_mask], 3,
                np.where(a_scores[away_mask] == h_scores[away_mask], 1, 0),
            )

            gd[home_mask] = h_scores[home_mask] - a_scores[home_mask]
            gd[away_mask] = a_scores[away_mask] - h_scores[away_mask]

            opp_position = np.where(
                is_home, team_matches["away_position"], team_matches["home_position"]
            )
            opp_position = pd.Series(opp_position, index=idx).fillna(10.0)

            opp_strength = np.where(
                is_home,
                team_matches["away_overall_win_rate"],
                team_matches["home_overall_win_rate"],
            )
            opp_strength = pd.Series(opp_strength, index=idx).fillna(0.5)

            points_series = pd.Series(points, index=idx)
            gd_series = pd.Series(gd, index=idx)
            valid_series = pd.Series(valid.astype(float), index=idx)

            wins = (points_series == 3).astype(float)
            draws = (points_series == 1).astype(float)
            losses = (points_series == 0).astype(float)

            form_points = points_series.shift(1).rolling(window=window, min_periods=1).sum()
            games_played = (
                valid_series.shift(1).rolling(window=window, min_periods=1).sum().replace(0, 1)
            )
            form_ppg = form_points / games_played
            form_win_rate = wins.shift(1).rolling(window=window, min_periods=1).mean()
            form_draw_rate = draws.shift(1).rolling(window=window, min_periods=1).mean()
            form_loss_rate = losses.shift(1).rolling(window=window, min_periods=1).mean()
            form_gd = gd_series.shift(1).rolling(window=window, min_periods=1).sum()
            form_gd_per_game = form_gd / games_played
            form_opp_pos = opp_position.shift(1).rolling(window=window, min_periods=1).mean()
            form_opp_strength = opp_strength.shift(1).rolling(window=window, min_periods=1).mean()

            home_idx = idx[is_home]
            away_idx = idx[~is_home]

            df.loc[home_idx, "home_form10_points"] = form_points.loc[home_idx].fillna(0)
            df.loc[home_idx, "home_form10_ppg"] = form_ppg.loc[home_idx].fillna(0)
            df.loc[home_idx, "home_form10_win_rate"] = form_win_rate.loc[home_idx].fillna(0)
            df.loc[home_idx, "home_form10_draw_rate"] = form_draw_rate.loc[home_idx].fillna(0)
            df.loc[home_idx, "home_form10_loss_rate"] = form_loss_rate.loc[home_idx].fillna(0)
            df.loc[home_idx, "home_form10_goal_diff"] = form_gd.loc[home_idx].fillna(0)
            df.loc[home_idx, "home_form10_gd_per_game"] = form_gd_per_game.loc[home_idx].fillna(0)
            df.loc[home_idx, "home_form10_avg_opp_position"] = form_opp_pos.loc[home_idx].fillna(10.0)
            df.loc[home_idx, "home_form10_avg_opp_strength"] = form_opp_strength.loc[home_idx].fillna(0.5)

            df.loc[away_idx, "away_form10_points"] = form_points.loc[away_idx].fillna(0)
            df.loc[away_idx, "away_form10_ppg"] = form_ppg.loc[away_idx].fillna(0)
            df.loc[away_idx, "away_form10_win_rate"] = form_win_rate.loc[away_idx].fillna(0)
            df.loc[away_idx, "away_form10_draw_rate"] = form_draw_rate.loc[away_idx].fillna(0)
            df.loc[away_idx, "away_form10_loss_rate"] = form_loss_rate.loc[away_idx].fillna(0)
            df.loc[away_idx, "away_form10_goal_diff"] = form_gd.loc[away_idx].fillna(0)
            df.loc[away_idx, "away_form10_gd_per_game"] = form_gd_per_game.loc[away_idx].fillna(0)
            df.loc[away_idx, "away_form10_avg_opp_position"] = form_opp_pos.loc[away_idx].fillna(10.0)
            df.loc[away_idx, "away_form10_avg_opp_strength"] = form_opp_strength.loc[away_idx].fillna(0.5)

        return df

    def _add_h2h_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head historical features (vectorized, last 5 only)"""
        logger.info("Adding head-to-head features (vectorized)...")

        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        df["h2h_home_wins"] = 0.0
        df["h2h_away_wins"] = 0.0
        df["h2h_draws"] = 0.0
        df["h2h_home_goals"] = 0.0
        df["h2h_away_goals"] = 0.0

        df["team_pair"] = df.apply(
            lambda x: tuple(sorted([str(x["home_team"]), str(x["away_team"])])),
            axis=1,
        )

        valid_mask = (
            df["date"].notna()
            & df["home_score"].notna()
            & df["away_score"].notna()
        )
        df_valid = df[valid_mask].copy()
        if len(df_valid) == 0:
            logger.warning("No valid matches for H2H calculation")
            return df

        unique_pairs = df_valid["team_pair"].unique()
        logger.info(
            f"Processing {len(unique_pairs)} unique team pairs (simplified H2H - last 5 matches only)..."
        )

        max_h2h_matches = 5
        for pair_idx, pair in enumerate(unique_pairs):
            if pair_idx % 100 == 0:
                logger.info(f"Processing pair {pair_idx+1}/{len(unique_pairs)}")

            pair_mask = df_valid["team_pair"] == pair
            pair_matches = df_valid[pair_mask].sort_values("date").reset_index(drop=True)
            if len(pair_matches) < 2:
                continue

            for i in range(1, len(pair_matches)):
                prev_matches = pair_matches.iloc[max(0, i - max_h2h_matches) : i]
                if len(prev_matches) == 0:
                    continue

                current_match = pair_matches.iloc[i]
                current_home = current_match["home_team"]
                orig_idx = df_valid[pair_mask].iloc[i].name

                is_home = (prev_matches["home_team"] == current_home).values
                h_scores = prev_matches["home_score"].values
                a_scores = prev_matches["away_score"].values

                h_goals = np.where(is_home, h_scores, a_scores)
                a_goals = np.where(is_home, a_scores, h_scores)

                home_wins = int(np.sum(h_goals > a_goals))
                away_wins = int(np.sum(h_goals < a_goals))
                draws = int(np.sum(h_goals == a_goals))
                total = home_wins + away_wins + draws

                if total > 0:
                    df.at[orig_idx, "h2h_home_wins"] = home_wins / total
                    df.at[orig_idx, "h2h_away_wins"] = away_wins / total
                    df.at[orig_idx, "h2h_draws"] = draws / total
                    df.at[orig_idx, "h2h_home_goals"] = np.sum(h_goals) / total
                    df.at[orig_idx, "h2h_away_goals"] = np.sum(a_goals) / total

        df = df.drop(columns=["team_pair"])
        logger.info("H2H features complete!")
        return df

    def _add_team_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add overall team strength features (rolling windows)"""
        logger.info("Adding team strength features (simplified)...")

        window_size = 10
        prior_window = 20

        for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
            df[f"{prefix}_cumulative_points"] = 0.0
            df[f"{prefix}_overall_win_rate"] = 0.5
            df[f"{prefix}_strength_prior"] = 0.5

            teams = df[team_col].unique()
            logger.info(f"Processing {len(teams)} teams for {prefix} strength...")

            for i, team in enumerate(teams):
                if i % 10 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{len(teams)} teams...")

                team_mask = df[team_col] == team
                team_matches = df[team_mask].sort_values("date")
                if len(team_matches) == 0:
                    continue

                is_home = (team_matches["home_team"] == team).values
                h_scores = team_matches["home_score"].fillna(0).values
                a_scores = team_matches["away_score"].fillna(0).values
                valid = team_matches["home_score"].notna().values & team_matches["away_score"].notna().values

                points = np.full(len(team_matches), 0.0)
                home_mask = is_home & valid
                away_mask = ~is_home & valid

                points[home_mask] = np.where(
                    h_scores[home_mask] > a_scores[home_mask], 3,
                    np.where(h_scores[home_mask] == a_scores[home_mask], 1, 0),
                )
                points[away_mask] = np.where(
                    a_scores[away_mask] > h_scores[away_mask], 3,
                    np.where(a_scores[away_mask] == h_scores[away_mask], 1, 0),
                )

                points_series = pd.Series(points, index=team_matches.index)
                df.loc[team_matches.index, f"{prefix}_cumulative_points"] = (
                    points_series.shift(1).rolling(window=window_size, min_periods=1).sum()
                )
                df.loc[team_matches.index, f"{prefix}_overall_win_rate"] = (
                    (points_series.shift(1) == 3).rolling(window=window_size, min_periods=1).mean()
                )
                df.loc[team_matches.index, f"{prefix}_strength_prior"] = (
                    (points_series.shift(1).rolling(window=prior_window, min_periods=1).mean() / 3.0).fillna(0.5)
                )

        df["strength_prior_diff"] = df["home_strength_prior"] - df["away_strength_prior"]
        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum features (recent trend)"""
        logger.info("Adding momentum features (simplified)...")

        for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
            df[f"{prefix}_momentum"] = 0.0
            teams = df[team_col].unique()
            logger.info(f"Processing {len(teams)} teams for {prefix} momentum...")

            for i, team in enumerate(teams):
                if i % 10 == 0 and i > 0:
                    logger.info(f"  Processed {i}/{len(teams)} teams...")

                team_mask = df[team_col] == team
                team_matches = df[team_mask].sort_values("date")
                if len(team_matches) < 3:
                    continue

                is_home = (team_matches["home_team"] == team).values
                h_scores = team_matches["home_score"].fillna(0).values
                a_scores = team_matches["away_score"].fillna(0).values
                valid = team_matches["home_score"].notna().values & team_matches["away_score"].notna().values

                points = np.zeros(len(team_matches))
                home_mask = is_home & valid
                away_mask = ~is_home & valid

                points[home_mask] = np.where(
                    h_scores[home_mask] > a_scores[home_mask], 3,
                    np.where(h_scores[home_mask] == a_scores[home_mask], 1, 0),
                )
                points[away_mask] = np.where(
                    a_scores[away_mask] > h_scores[away_mask], 3,
                    np.where(a_scores[away_mask] == h_scores[away_mask], 1, 0),
                )

                points_series = pd.Series(points, index=team_matches.index)
                points_3 = points_series.shift(1).rolling(window=3, min_periods=1).sum()
                df.loc[team_matches.index, f"{prefix}_momentum"] = points_3.fillna(0)

        return df

    def _add_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add league position features (simplified - season position)"""
        logger.info("Adding position features (simplified)...")

        df["home_position"] = np.nan
        df["away_position"] = np.nan
        df["position_diff"] = np.nan

        if "season" not in df.columns:
            return df

        for season in df["season"].unique():
            season_mask = df["season"] == season
            season_df = df[season_mask].copy()

            team_points = {}
            valid_matches = season_df[
                season_df["home_score"].notna() & season_df["away_score"].notna()
            ]
            home_wins = valid_matches["home_score"] > valid_matches["away_score"]
            away_wins = valid_matches["away_score"] > valid_matches["home_score"]
            draws = valid_matches["home_score"] == valid_matches["away_score"]

            for team in valid_matches["home_team"].unique():
                home_mask = valid_matches["home_team"] == team
                team_points[team] = home_wins[home_mask].sum() * 3 + draws[home_mask].sum()

            for team in valid_matches["away_team"].unique():
                away_mask = valid_matches["away_team"] == team
                if team not in team_points:
                    team_points[team] = 0
                team_points[team] += away_wins[away_mask].sum() * 3 + draws[away_mask].sum()

            sorted_teams = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
            position_map = {team: pos + 1 for pos, (team, _) in enumerate(sorted_teams)}

            df.loc[season_mask, "home_position"] = df.loc[season_mask, "home_team"].map(position_map)
            df.loc[season_mask, "away_position"] = df.loc[season_mask, "away_team"].map(position_map)

        df["position_diff"] = df["away_position"] - df["home_position"]
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        logger.info("Adding time features...")

        if "date" in df.columns:
            df["month"] = df["date"].dt.month
            df["day_of_week"] = df["date"].dt.dayofweek
            df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

            for season in df["season"].unique():
                season_start = df[df["season"] == season]["date"].min()
                mask = df["season"] == season
                df.loc[mask, "days_since_season_start"] = (
                    (df.loc[mask, "date"] - season_start).dt.days
                )

        return df

    def _add_rest_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rest/fatigue features: days since last match"""
        logger.info("Adding rest features (days since last match)...")

        df["home_days_since_last_match"] = 0.0
        df["away_days_since_last_match"] = 0.0

        teams = pd.unique(pd.concat([df["home_team"], df["away_team"]], ignore_index=True))
        for team in teams:
            team_mask = (df["home_team"] == team) | (df["away_team"] == team)
            team_matches = df[team_mask].sort_values("date")
            if len(team_matches) == 0:
                continue

            dates = team_matches["date"]
            rest_days = dates.diff().dt.days.fillna(0)
            rest_series = pd.Series(rest_days.values, index=team_matches.index)

            home_idx = team_matches.index[team_matches["home_team"] == team]
            away_idx = team_matches.index[team_matches["away_team"] == team]

            df.loc[home_idx, "home_days_since_last_match"] = rest_series.loc[home_idx].fillna(0)
            df.loc[away_idx, "away_days_since_last_match"] = rest_series.loc[away_idx].fillna(0)

        return df

    def _merge_team_stats(self, df: pd.DataFrame, team_stats_df: pd.DataFrame) -> pd.DataFrame:
        """Merge additional team statistics if available"""
        logger.info("Merging team statistics...")
        return df

    def _add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features: xG, referee, weather, injuries, managers, shots"""
        logger.info("Adding advanced features (xG, referee, weather, injuries, managers)...")

        advanced_features = [
            "home_xg",
            "away_xg",
            "xg_difference",
            "referee_strictness",
            "weather_temperature",
            "weather_humidity",
            "weather_wind_speed",
            "weather_is_rainy",
            "weather_is_snowy",
            "weather_is_clear",
            "home_key_players_injured",
            "away_key_players_injured",
            "home_total_injured",
            "away_total_injured",
            "home_key_players_suspended",
            "away_key_players_suspended",
            "home_injury_severity",
            "away_injury_severity",
            "injury_difference",
            "home_manager_tenure_days",
            "away_manager_tenure_days",
            "manager_tenure_diff",
            "home_manager_win_pct",
            "away_manager_win_pct",
            "manager_win_pct_diff",
            "home_manager_trophies",
            "away_manager_trophies",
            "manager_trophies_diff",
            "home_sot_ratio",
            "away_sot_ratio",
        ]

        for feat in advanced_features:
            if feat not in df.columns:
                df[feat] = 0.0

        if "home_xg" in df.columns and "away_xg" in df.columns:
            df["xg_difference"] = df["home_xg"].fillna(0) - df["away_xg"].fillna(0)
            for team_col, prefix in [("home_team", "home"), ("away_team", "away")]:
                xg_col = f"{prefix}_xg"
                if xg_col in df.columns:
                    for team in df[team_col].unique():
                        team_mask = df[team_col] == team
                        team_matches = df[team_mask].sort_values("date")
                        if len(team_matches) > 0:
                            xg_series = pd.Series(
                                team_matches[xg_col].fillna(0).values,
                                index=team_matches.index,
                            )
                            df.loc[team_matches.index, f"{prefix}_avg_xg"] = (
                                xg_series.shift(1)
                                .rolling(window=5, min_periods=1)
                                .mean()
                                .fillna(0)
                            )

        if "referee" in df.columns:
            referee_counts = df["referee"].value_counts().to_dict()
            df["referee_experience"] = df["referee"].map(referee_counts).fillna(0)
            df["referee_strictness"] = 0.5

        weather_cols = [
            "weather_temperature",
            "weather_humidity",
            "weather_wind_speed",
            "weather_is_rainy",
            "weather_is_snowy",
            "weather_is_clear",
        ]
        for col in weather_cols:
            if col not in df.columns:
                df[col] = 0.0

        injury_cols = [
            "home_key_players_injured",
            "away_key_players_injured",
            "home_total_injured",
            "away_total_injured",
            "home_key_players_suspended",
            "away_key_players_suspended",
            "home_injury_severity",
            "away_injury_severity",
            "injury_difference",
        ]
        for col in injury_cols:
            if col not in df.columns:
                df[col] = 0.0

        strength_cols = ["home_strength_prior", "away_strength_prior", "strength_prior_diff"]
        for col in strength_cols:
            if col not in df.columns:
                df[col] = 0.0

        if "home_manager_tenure_days" in df.columns and "away_manager_tenure_days" in df.columns:
            df["manager_tenure_diff"] = df["home_manager_tenure_days"].fillna(0) - df[
                "away_manager_tenure_days"
            ].fillna(0)
        if "home_manager_win_pct" in df.columns and "away_manager_win_pct" in df.columns:
            df["manager_win_pct_diff"] = df["home_manager_win_pct"].fillna(0) - df[
                "away_manager_win_pct"
            ].fillna(0)
        if "home_manager_trophies" in df.columns and "away_manager_trophies" in df.columns:
            df["manager_trophies_diff"] = df["home_manager_trophies"].fillna(0) - df[
                "away_manager_trophies"
            ].fillna(0)

        # Manager stats from data/managers_2025_26.csv if available
        manager_path = Path("data") / "managers_2025_26.csv"
        if manager_path.exists():
            mgr_df = pd.read_csv(manager_path)
            mgr_df.columns = [c.strip().lower() for c in mgr_df.columns]
            if "team" in mgr_df.columns:
                team_map = dict(
                    zip(
                        mgr_df["team"].astype(str).str.strip(),
                        mgr_df.to_dict(orient="records"),
                    )
                )
                home_vals = df["home_team"].map(team_map).dropna()
                away_vals = df["away_team"].map(team_map).dropna()
                if "career_ppg" in mgr_df.columns:
                    df["home_manager_win_pct"] = df["home_team"].map(
                        lambda t: team_map.get(t, {}).get("career_ppg", 0)
                    )
                    df["away_manager_win_pct"] = df["away_team"].map(
                        lambda t: team_map.get(t, {}).get("career_ppg", 0)
                    )
                    df["manager_win_pct_diff"] = df["home_manager_win_pct"].fillna(0) - df[
                        "away_manager_win_pct"
                    ].fillna(0)
                if "major_trophies_top5" in mgr_df.columns:
                    df["home_manager_trophies"] = df["home_team"].map(
                        lambda t: team_map.get(t, {}).get("major_trophies_top5", 0)
                    )
                    df["away_manager_trophies"] = df["away_team"].map(
                        lambda t: team_map.get(t, {}).get("major_trophies_top5", 0)
                    )
                    df["manager_trophies_diff"] = df["home_manager_trophies"].fillna(0) - df[
                        "away_manager_trophies"
                    ].fillna(0)

        # Squad values from Transfermarkt (data/team_values.csv)
        squad_path = Path("data") / "team_values.csv"
        if squad_path.exists():
            values_df = pd.read_csv(squad_path)
            values_df.columns = [c.strip().lower() for c in values_df.columns]
            if "team" in values_df.columns and "squad_value_eur" in values_df.columns:
                value_map = dict(
                    zip(
                        values_df["team"].astype(str).str.strip(),
                        values_df["squad_value_eur"],
                    )
                )
                df["home_squad_value_eur"] = df["home_team"].map(value_map).fillna(0)
                df["away_squad_value_eur"] = df["away_team"].map(value_map).fillna(0)
                df["squad_value_diff"] = df["home_squad_value_eur"] - df[
                    "away_squad_value_eur"
                ]

        # Shots on target ratio if raw columns exist
        if "home_shots_on_target" in df.columns and "home_shots" in df.columns:
            df["home_sot_ratio"] = (
                df["home_shots_on_target"].fillna(0) / df["home_shots"].replace(0, np.nan)
            ).fillna(0)
        if "away_shots_on_target" in df.columns and "away_shots" in df.columns:
            df["away_sot_ratio"] = (
                df["away_shots_on_target"].fillna(0) / df["away_shots"].replace(0, np.nan)
            ).fillna(0)

        logger.info("Advanced features added!")
        return df

    def get_feature_columns(self) -> List[str]:
        """Get list of feature column names"""
        return [
            "home_avg_goals_scored",
            "home_avg_goals_conceded",
            "home_avg_goal_diff",
            "home_avg_goals_scored_5",
            "home_avg_goals_conceded_5",
            "home_avg_goal_diff_5",
            "away_avg_goals_scored",
            "away_avg_goals_conceded",
            "away_avg_goal_diff",
            "away_avg_goals_scored_5",
            "away_avg_goals_conceded_5",
            "away_avg_goal_diff_5",
            "home_form_points",
            "home_win_rate",
            "away_form_points",
            "away_win_rate",
            "home_form10_points",
            "home_form10_ppg",
            "home_form10_win_rate",
            "home_form10_draw_rate",
            "home_form10_loss_rate",
            "home_form10_goal_diff",
            "home_form10_gd_per_game",
            "home_form10_avg_opp_position",
            "home_form10_avg_opp_strength",
            "away_form10_points",
            "away_form10_ppg",
            "away_form10_win_rate",
            "away_form10_draw_rate",
            "away_form10_loss_rate",
            "away_form10_goal_diff",
            "away_form10_gd_per_game",
            "away_form10_avg_opp_position",
            "away_form10_avg_opp_strength",
            "h2h_home_wins",
            "h2h_away_wins",
            "h2h_draws",
            "h2h_home_goals",
            "h2h_away_goals",
            "home_cumulative_points",
            "home_overall_win_rate",
            "home_strength_prior",
            "away_cumulative_points",
            "away_overall_win_rate",
            "away_strength_prior",
            "strength_prior_diff",
            "home_momentum",
            "away_momentum",
            "home_position",
            "away_position",
            "position_diff",
            "month",
            "day_of_week",
            "is_weekend",
            "days_since_season_start",
            "home_days_since_last_match",
            "away_days_since_last_match",
            "home_xg",
            "away_xg",
            "xg_difference",
            "home_avg_xg",
            "away_avg_xg",
            "referee_experience",
            "referee_strictness",
            "weather_temperature",
            "weather_humidity",
            "weather_wind_speed",
            "weather_is_rainy",
            "weather_is_snowy",
            "weather_is_clear",
            "home_key_players_injured",
            "away_key_players_injured",
            "home_total_injured",
            "away_total_injured",
            "home_key_players_suspended",
            "away_key_players_suspended",
            "home_injury_severity",
            "away_injury_severity",
            "injury_difference",
            "home_manager_tenure_days",
            "away_manager_tenure_days",
            "manager_tenure_diff",
            "home_manager_win_pct",
            "away_manager_win_pct",
            "manager_win_pct_diff",
            "home_manager_trophies",
            "away_manager_trophies",
            "manager_trophies_diff",
            "home_sot_ratio",
            "away_sot_ratio",
            "home_squad_value_eur",
            "away_squad_value_eur",
            "squad_value_diff",
        ]
