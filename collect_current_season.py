"""
Collect current season (2024/25) data for predictions
"""

import pandas as pd
import logging
from pathlib import Path
from src.data_collection.fbref_scraper import FBrefScraper
from src.feature_engineering.features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collect_2024_25_season():
    """Collect 2024/25 season data"""
    
    logger.info("="*60)
    logger.info("COLLECTING 2024/25 SEASON DATA")
    logger.info("="*60)
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize scraper
    scraper = FBrefScraper(delay=2.0)
    
    # Collect 2024/25 season
    logger.info("Scraping 2024/25 season matches...")
    matches_df = scraper.get_historical_seasons("2024-2025", "2024-2025")
    
    if matches_df.empty:
        logger.error("No data collected for 2024/25 season!")
        logger.info("This might be because:")
        logger.info("  1. The season hasn't started yet")
        logger.info("  2. FBref structure has changed")
        logger.info("  3. Network issues")
        return None
    
    # Save raw data
    output_path = data_dir / "raw_matches_2024_25.csv"
    matches_df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved {len(matches_df)} matches to {output_path}")
    
    # Engineer features
    logger.info("\nEngineering features for 2024/25 season...")
    feature_engineer = FeatureEngineer()
    
    # Load historical data for context (needed for rolling stats, H2H, etc.)
    historical_path = data_dir / "raw_matches.csv"
    if historical_path.exists():
        logger.info("Loading historical data for feature engineering context...")
        historical_df = pd.read_csv(historical_path)
        # Combine historical and current season
        all_matches = pd.concat([historical_df, matches_df], ignore_index=True)
        all_matches['date'] = pd.to_datetime(all_matches['date'])
        all_matches = all_matches.sort_values('date')
    else:
        logger.warning("No historical data found. Features may be limited.")
        all_matches = matches_df.copy()
        all_matches['date'] = pd.to_datetime(all_matches['date'])
    
    # Create features
    features_df = feature_engineer.create_match_features(
        all_matches, 
        include_h2h=True  # Include H2H for better predictions
    )
    
    # Filter to only 2024/25 season
    features_2024_25 = features_df[
        features_df['date'] >= pd.to_datetime('2024-08-01')
    ].copy()
    
    # Save features
    features_path = data_dir / "features_2024_25.csv"
    features_2024_25.to_csv(features_path, index=False)
    logger.info(f"✅ Saved features for {len(features_2024_25)} matches to {features_path}")
    
    logger.info("\n" + "="*60)
    logger.info("2024/25 SEASON DATA COLLECTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total matches: {len(matches_df)}")
    logger.info(f"Total features: {len(features_2024_25)}")
    
    return features_2024_25


if __name__ == "__main__":
    collect_2024_25_season()

