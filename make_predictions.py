"""
Make predictions for upcoming matches or specific fixtures
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from src.models.predictor import MatchPredictor
from src.feature_engineering.features import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_single_match(home_team: str, away_team: str, 
                        predictor: MatchPredictor,
                        feature_engineer: FeatureEngineer,
                        historical_data: pd.DataFrame) -> dict:
    """
    Predict a single match by creating features from historical data
    
    Note: This is a simplified version. For real predictions, you'd need
    to calculate features based on current team form, standings, etc.
    """
    # Get recent matches for both teams
    home_recent = historical_data[
        (historical_data['home_team'] == home_team) | 
        (historical_data['away_team'] == home_team)
    ].tail(10)
    
    away_recent = historical_data[
        (historical_data['home_team'] == away_team) | 
        (historical_data['away_team'] == away_team)
    ].tail(10)
    
    # Create a dummy match row with average features
    # In production, you'd calculate these properly
    feature_cols = predictor.feature_columns
    
    # Use average features from recent matches as approximation
    if not home_recent.empty and not away_recent.empty:
        # This is simplified - real implementation would calculate
        # proper rolling stats, form, etc.
        features = {}
        for col in feature_cols:
            if col in home_recent.columns:
                home_val = home_recent[col].mean() if col in home_recent.columns else 0
                away_val = away_recent[col].mean() if col in away_recent.columns else 0
                # Use home/away specific columns
                if 'home_' in col:
                    features[col] = home_val
                elif 'away_' in col:
                    features[col] = away_val
                else:
                    features[col] = (home_val + away_val) / 2
            else:
                features[col] = 0
    else:
        # Default features if no history
        features = {col: 0 for col in feature_cols}
    
    # Make prediction
    result = predictor.predict_match(home_team, away_team, features)
    return result


def show_predictions_summary():
    """Show summary of model and how to use it"""
    
    logger.info("="*60)
    logger.info("PREMIER LEAGUE PREDICTION MODEL")
    logger.info("="*60)
    
    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        logger.error("Model not found. Train it first:")
        logger.info("  python3 -m src.pipeline --train")
        return
    
    predictor = MatchPredictor(str(model_path))
    
    logger.info(f"\n✅ Model loaded: {model_path}")
    logger.info(f"✅ Features: {len(predictor.feature_columns)}")
    logger.info(f"✅ Model type: {type(predictor.model).__name__}")
    
    logger.info("\n" + "="*60)
    logger.info("HOW TO USE THIS MODEL")
    logger.info("="*60)
    logger.info("\n1. Test predictions on test set:")
    logger.info("   python3 test_predictions.py")
    logger.info("\n2. For 2025/26 season predictions:")
    logger.info("   - You'll need to collect current season data first")
    logger.info("   - Then generate features for upcoming fixtures")
    logger.info("   - The model can then predict those matches")
    logger.info("\n3. Example prediction (simplified):")
    
    # Try to load historical data for example
    features_path = Path("data/features.csv")
    if features_path.exists():
        historical_data = pd.read_csv(features_path)
        feature_engineer = FeatureEngineer()
        
        # Example: Arsenal vs Man City
        logger.info("\n   Example: Arsenal vs Manchester City")
        try:
            result = predict_single_match(
                "Arsenal", "Manchester City",
                predictor, feature_engineer, historical_data
            )
            logger.info(f"   Predicted: {result['predicted_outcome']}")
            logger.info(f"   Home Win: {result['home_win_prob']:.1%}")
            logger.info(f"   Draw: {result['draw_prob']:.1%}")
            logger.info(f"   Away Win: {result['away_win_prob']:.1%}")
            logger.info(f"   Confidence: {result['confidence']:.1%}")
        except Exception as e:
            logger.warning(f"   Could not make example prediction: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("NEXT STEPS")
    logger.info("="*60)
    logger.info("1. Run test_predictions.py to see model performance")
    logger.info("2. Collect 2024/25 season data (if not already done)")
    logger.info("3. Generate features for 2025/26 fixtures")
    logger.info("4. Make predictions for the new season!")
    logger.info("="*60)


if __name__ == "__main__":
    show_predictions_summary()

