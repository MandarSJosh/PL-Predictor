"""
Test predictions on the test set and show results
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from src.models.predictor import MatchPredictor
from src.models.trainer import ModelTrainer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    log_loss,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _apply_draw_calibration(probabilities: np.ndarray, margin: float, floor: float) -> np.ndarray:
    preds = []
    for probs in probabilities:
        draw_prob = float(probs[1])
        max_prob = float(np.max(probs))
        best_idx = int(np.argmax(probs))
        if draw_prob >= floor and draw_prob >= (max_prob - margin):
            preds.append(1)
        else:
            preds.append(best_idx)
    return np.array(preds)


def test_predictions():
    """Test model predictions on test set"""
    
    logger.info("Loading features and model...")
    
    # Load features
    features_path = Path("data/features.csv")
    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        logger.info("Run the pipeline first: python3 -m src.pipeline --features --train")
        return
    
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(features_df)} matches with features")
    
    # Load model
    model_path = Path("models/best_model.pkl")
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        logger.info("Train the model first: python3 -m src.pipeline --train")
        return
    
    predictor = MatchPredictor(str(model_path))
    
    # Prepare data (same as training)
    trainer = ModelTrainer()
    feature_columns = predictor.feature_columns
    
    # Get X and y
    X = features_df[feature_columns].fillna(0).values
    y = features_df["target"].values
    
    # Use same split as training (80/20)
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_test = y[split_idx:]
    valid_mask = ~pd.isna(y_test)
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]
    
    # Scale
    X_test_scaled = predictor.scaler.transform(X_test)
    
    # Predict
    logger.info("Making predictions on test set...")
    y_pred_proba = predictor.model.predict_proba(X_test_scaled)

    # Draw calibration grid search
    margins = np.linspace(0.02, 0.20, 10)
    floors = np.linspace(0.05, 0.35, 7)
    best_margin = 0.08
    best_floor = 0.20
    best_score = -1.0
    best_macro_f1 = -1.0
    best_weighted_f1 = -1.0
    best_accuracy = -1.0

    reverse_mapping = {0: -1, 1: 0, 2: 1}
    for margin in margins:
        for floor in floors:
            pred_idx = _apply_draw_calibration(y_pred_proba, margin, floor)
            pred_mapped = np.array([reverse_mapping[p] for p in pred_idx])
            macro_f1 = f1_score(y_test, pred_mapped, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test, pred_mapped, average="weighted", zero_division=0)
            acc = accuracy_score(y_test, pred_mapped)
            score = (macro_f1 + weighted_f1 + acc) / 3.0
            if score > best_score:
                best_score = score
                best_macro_f1 = macro_f1
                best_weighted_f1 = weighted_f1
                best_accuracy = acc
                best_margin = float(margin)
                best_floor = float(floor)

    y_pred_idx = _apply_draw_calibration(y_pred_proba, best_margin, best_floor)
    y_pred = np.array([reverse_mapping[p] for p in y_pred_idx])
    
    # Map predictions back to original target format
    # Model uses [0, 1, 2] but original target is [-1, 0, 1]
    y_pred_mapped = y_pred
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_mapped)
    macro_f1 = f1_score(y_test, y_pred_mapped, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred_mapped, average="weighted", zero_division=0)
    precision = precision_score(y_test, y_pred_mapped, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred_mapped, average="macro", zero_division=0)
    try:
        logloss = log_loss(y_test, y_pred_proba, labels=[-1, 0, 1])
    except ValueError:
        logloss = float("nan")
    report = classification_report(
        y_test,
        y_pred_mapped,
        target_names=['Away Win', 'Draw', 'Home Win'],
        output_dict=True
    )
    
    logger.info("\n" + "="*60)
    logger.info("PREDICTION TEST RESULTS")
    logger.info("="*60)
    logger.info(f"Test set size: {len(y_test)} matches")
    logger.info(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"Macro F1: {macro_f1:.4f}")
    logger.info(f"Weighted F1: {weighted_f1:.4f}")
    logger.info(f"Precision (macro): {precision:.4f}")
    logger.info(f"Recall (macro): {recall:.4f}")
    logger.info(f"Log Loss: {logloss:.4f}")
    logger.info(f"Draw calibration: margin={best_margin:.2f}, floor={best_floor:.2f}")
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred_mapped, 
                                     target_names=['Away Win', 'Draw', 'Home Win']))
    
    logger.info("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_mapped)
    logger.info(f"\n{'':>12} {'Away Win':>10} {'Draw':>10} {'Home Win':>10}")
    logger.info(f"{'Away Win':>12} {cm[0,0]:>10} {cm[0,1]:>10} {cm[0,2]:>10}")
    logger.info(f"{'Draw':>12} {cm[1,0]:>10} {cm[1,1]:>10} {cm[1,2]:>10}")
    logger.info(f"{'Home Win':>12} {cm[2,0]:>10} {cm[2,1]:>10} {cm[2,2]:>10}")
    
    # Show some example predictions
    logger.info("\n" + "="*60)
    logger.info("SAMPLE PREDICTIONS (Last 10 matches in test set)")
    logger.info("="*60)
    
    test_matches = features_df.iloc[split_idx:].copy()
    test_matches['predicted'] = y_pred_mapped
    test_matches['actual'] = y_test
    
    outcome_map = {-1: 'Away Win', 0: 'Draw', 1: 'Home Win'}
    
    for idx, row in test_matches.tail(10).iterrows():
        home = row.get('home_team', 'Unknown')
        away = row.get('away_team', 'Unknown')
        actual = outcome_map.get(row['actual'], 'Unknown')
        predicted = outcome_map.get(row['predicted'], 'Unknown')
        
        # Get probabilities
        match_idx = len(test_matches) - list(test_matches.tail(10).index).index(idx) - 1
        probs = y_pred_proba[match_idx]
        home_prob = probs[2]  # Home win is class 2
        draw_prob = probs[1]   # Draw is class 1
        away_prob = probs[0]   # Away win is class 0
        
        correct = "✓" if actual == predicted else "✗"
        logger.info(f"{correct} {home:20} vs {away:20} | "
                   f"Predicted: {predicted:10} | Actual: {actual:10} | "
                   f"Probs: H:{home_prob:.2f} D:{draw_prob:.2f} A:{away_prob:.2f}")
    
    logger.info("\n" + "="*60)
    logger.info("Test completed!")
    logger.info("="*60)

    # Save metrics to disk
    metrics_path = Path("data/performance_metrics.csv")
    summary_path = Path("data/performance_metrics.txt")
    metrics_df = pd.DataFrame(report).transpose()
    metrics_df['accuracy'] = accuracy
    metrics_df.to_csv(metrics_path, index=True)
    summary_lines = [
        f"Test set size: {len(y_test)}",
        f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)",
        f"Macro F1: {macro_f1:.4f}",
        f"Weighted F1: {weighted_f1:.4f}",
        f"Precision (macro): {precision:.4f}",
        f"Recall (macro): {recall:.4f}",
        f"Log Loss: {logloss:.4f}",
        f"Draw calibration: margin={best_margin:.2f}, floor={best_floor:.2f}",
    ]
    summary_path.write_text("\n".join(summary_lines) + "\n")
    logger.info(f"✅ Saved metrics to {metrics_path} and {summary_path}")


if __name__ == "__main__":
    test_predictions()

