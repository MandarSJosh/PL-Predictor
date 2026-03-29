"""
Main Pipeline for Data Collection, Feature Engineering, and Model Training
"""

import pandas as pd
import logging
from pathlib import Path
import argparse
from datetime import datetime

from src.data_collection.premierleague_api import PremierLeagueAPIScraper
from src.feature_engineering.features import FeatureEngineer
from src.models.trainer import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_data(start_season: str = "2018-2019", 
                end_season: str = "2024-2025",
                output_dir: str = "data",
                force: bool = False) -> pd.DataFrame:
    """
    Collect data from PremierLeague.com API
    
    Args:
        start_season: First season to collect
        end_season: Last season to collect
        output_dir: Directory to save data
        
    Returns:
        DataFrame with match data
    """
    logger.info("Starting data collection...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    scraper = PremierLeagueAPIScraper()

    # Collect match data
    matches_df = scraper.get_historical_seasons(start_season, end_season, force=force)
    
    if matches_df.empty:
        logger.error("No data collected!")
        return pd.DataFrame()
    
    # Save raw data
    output_path = Path(output_dir) / "raw_matches.csv"
    matches_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(matches_df)} matches to {output_path}")
    
    return matches_df


def engineer_features(matches_df: pd.DataFrame,
                     output_dir: str = "data",
                     include_h2h: bool = False,
                     include_weather: bool = True,
                     include_injuries: bool = True) -> pd.DataFrame:
    """
    Engineer features from match data
    
    Args:
        matches_df: DataFrame with match data
        output_dir: Directory to save processed data
        include_h2h: Include head-to-head features (slower but more accurate)
        include_weather: Include weather features (requires API key)
        include_injuries: Include injury/suspension features
        
    Returns:
        DataFrame with features
    """
    logger.info("Starting feature engineering...")
    
    # Add advanced data sources
    if include_weather:
        try:
            from src.data_collection.weather_api import WeatherAPI
            weather_api = WeatherAPI()
            matches_df = weather_api.add_weather_features(matches_df)
        except Exception as e:
            logger.warning(f"Could not add weather features: {e}")
    
    if include_injuries:
        try:
            from src.data_collection.injury_tracker import InjuryTracker
            injury_tracker = InjuryTracker()
            matches_df = injury_tracker.add_injury_features(matches_df)
        except Exception as e:
            logger.warning(f"Could not add injury features: {e}")
    
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.create_match_features(matches_df, include_h2h=include_h2h)
    
    # Save features
    output_path = Path(output_dir) / "features.csv"
    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved features to {output_path}")
    
    return features_df


def train_models(features_df: pd.DataFrame,
                optimize: bool = True,
                model_dir: str = "models") -> None:
    """
    Train models on features
    
    Args:
        features_df: DataFrame with features
        optimize: Whether to optimize hyperparameters
        model_dir: Directory to save models
    """
    logger.info("Starting model training...")
    
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    trainer = ModelTrainer(experiment_name="premier_league_2025_26")
    
    # Get feature columns
    feature_engineer = FeatureEngineer()
    feature_columns = feature_engineer.get_feature_columns()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features_df,
        feature_columns
    )
    
    # Train all models
    results = trainer.train_all(X_train, y_train, X_test, y_test, optimize)
    
    # Save best model
    model_path = Path(model_dir) / "best_model.pkl"
    trainer.save_model(str(model_path))
    
    # Print results
    logger.info("\n" + "="*50)
    logger.info("MODEL TRAINING RESULTS")
    logger.info("="*50)
    for name, result in results.items():
        logger.info(f"{name.upper()}:")
        logger.info(f"  Accuracy: {result['accuracy']:.4f}")
        logger.info(f"  Log Loss: {result['log_loss']:.4f}")
    logger.info("="*50)


def main():
    """Main pipeline execution"""
    parser = argparse.ArgumentParser(description="Premier League Prediction Pipeline")
    parser.add_argument("--collect", action="store_true", help="Collect data from FBref")
    parser.add_argument("--features", action="store_true", help="Engineer features")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--start-season", type=str, default="2018-2019", help="Start season")
    parser.add_argument("--end-season", type=str, default="2025-2026", help="End season")
    parser.add_argument("--no-optimize", action="store_true", help="Skip hyperparameter optimization")
    parser.add_argument("--no-h2h", action="store_true", help="Disable head-to-head features")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    parser.add_argument("--pl-season", type=str, default="2025-2026", help="PL season for fixtures/teams")
    parser.add_argument("--save-fixtures", action="store_true", help="Save remaining fixtures to data/")
    parser.add_argument("--save-teams", action="store_true", help="Save season team list to data/")
    
    args = parser.parse_args()
    
    # Run full pipeline if --all is specified
    if args.all:
        args.collect = True
        args.features = True
        args.train = True
    
    # Save PL fixtures/teams if requested
    if args.save_fixtures:
        try:
            scraper = PremierLeagueAPIScraper()
            scraper.save_remaining_fixtures(args.pl_season, args.data_dir)
        except Exception as e:
            logger.warning(f"Could not save remaining fixtures: {e}")

    if args.save_teams:
        try:
            scraper = PremierLeagueAPIScraper()
            scraper.save_season_teams(args.pl_season, args.data_dir)
        except Exception as e:
            logger.warning(f"Could not save teams list: {e}")

    # Data collection
    matches_df = None
    if args.collect:
        matches_df = collect_data(args.start_season, args.end_season, args.data_dir)
    else:
        # Try to load existing data
        data_path = Path(args.data_dir) / "raw_matches.csv"
        if data_path.exists():
            logger.info(f"Loading existing data from {data_path}")
            matches_df = pd.read_csv(data_path)
            # Detect bad away_team values (kickoff times)
            if "away_team" in matches_df.columns:
                time_pattern = r"^\d{1,2}:\d{2}$"
                bad_ratio = (
                    matches_df["away_team"].astype(str).str.match(time_pattern).mean()
                )
                empty_ratio = (
                    matches_df["away_team"].isna()
                    | (matches_df["away_team"].astype(str).str.strip() == "")
                ).mean()
                if bad_ratio > 0.2 or empty_ratio > 0.2:
                    logger.warning(
                        "Detected time-like away_team values; re-collecting via PL API."
                    )
                    matches_df = collect_data(
                        args.start_season, args.end_season, args.data_dir, force=True
                    )
        else:
            logger.warning("No existing data found. Run with --collect first.")
            return
    
    # Feature engineering
    features_df = None
    if args.features:
        if matches_df is None:
            logger.error("No match data available for feature engineering")
            return
        features_df = engineer_features(
            matches_df, 
            args.data_dir, 
            include_h2h=not args.no_h2h,
            include_weather=True,
            include_injuries=True
        )
    else:
        # Try to load existing features
        features_path = Path(args.data_dir) / "features.csv"
        if features_path.exists():
            logger.info(f"Loading existing features from {features_path}")
            features_df = pd.read_csv(features_path)
        else:
            logger.warning("No existing features found. Run with --features first.")
            return
    
    # Model training
    if args.train:
        if features_df is None:
            logger.error("No features available for training")
            return
        train_models(features_df, not args.no_optimize, args.model_dir)
    
    logger.info("Pipeline completed successfully!")


if __name__ == "__main__":
    main()

