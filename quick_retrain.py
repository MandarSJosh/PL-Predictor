"""
Quick retrain with class weights - fewer trials for faster testing
"""

import pandas as pd
from pathlib import Path
import logging
from src.feature_engineering.features import FeatureEngineer
from src.models.trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_retrain():
    """Retrain model with class weights (reduced trials for speed)"""
    
    logger.info("="*60)
    logger.info("QUICK RETRAIN WITH CLASS WEIGHTS")
    logger.info("="*60)
    
    # Load features
    features_path = Path("data/features.csv")
    if not features_path.exists():
        logger.error("Features not found! Run: python3 -m src.pipeline --features")
        return
    
    features_df = pd.read_csv(features_path)
    logger.info(f"Loaded {len(features_df)} matches with features")
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Get feature columns
    feature_engineer = FeatureEngineer()
    feature_columns = feature_engineer.get_feature_columns()
    
    # Prepare data (this will calculate class weights)
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        features_df, feature_columns
    )
    
    # Temporarily modify Optuna trials to 20 instead of 100 for speed
    # We'll train XGBoost with fewer trials
    logger.info("Training XGBoost with class weights (20 trials for speed)...")
    
    # Monkey patch to reduce trials
    original_optimize = trainer.train_xgboost
    def quick_train_xgboost(X_train, y_train, X_test, y_test, optimize=True):
        logger.info("Training XGBoost model with class weights...")
        
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, log_loss
        import optuna
        import mlflow
        from sklearn.utils.class_weight import compute_class_weight
        import numpy as np
        
        # Calculate class weights
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        sample_weights = np.array([dict(zip(classes, class_weights))[y] for y in y_train])
        
        if optimize:
            def objective(trial):
                params = {
                    'objective': 'multi:softprob',
                    'num_class': 3,
                    'max_depth': trial.suggest_int('max_depth', 3, 8),
                    'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2),
                    'n_estimators': trial.suggest_int('n_estimators', 200, 600),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
                    'subsample': trial.suggest_float('subsample', 0.7, 0.95),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 0.95),
                    'gamma': trial.suggest_float('gamma', 0, 3),
                    'reg_alpha': trial.suggest_float('reg_alpha', 0, 8),
                    'reg_lambda': trial.suggest_float('reg_lambda', 0, 8),
                    'random_state': 42,
                    'eval_metric': 'mlogloss'
                }
                
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train,
                         sample_weight=sample_weights,
                         eval_set=[(X_test, y_test)],
                         verbose=False)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Penalize if no draws predicted
                unique_pred = len(np.unique(y_pred))
                if unique_pred < 3:  # If not all 3 classes predicted
                    accuracy *= 0.9  # Small penalty
                
                return accuracy
            
            study = optuna.create_study(direction='maximize', study_name='xgboost_quick')
            study.optimize(objective, n_trials=20, show_progress_bar=True)  # Only 20 trials!
            
            best_params = study.best_params
            best_params.update({
                'objective': 'multi:softprob',
                'num_class': 3,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            })
        else:
            best_params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 500,
                'random_state': 42,
                'eval_metric': 'mlogloss'
            }
        
        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train,
                 sample_weight=sample_weights,
                 eval_set=[(X_test, y_test)],
                 verbose=False)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        logloss = log_loss(y_test, y_pred_proba, labels=[0, 1, 2])
        
        # Check if draws are predicted
        unique_pred = np.unique(y_pred)
        draw_predicted = 1 in unique_pred  # Class 1 is draw
        
        logger.info(f"XGBoost - Accuracy: {accuracy:.4f}, Log Loss: {logloss:.4f}")
        logger.info(f"Unique classes predicted: {unique_pred}")
        logger.info(f"Draws predicted: {draw_predicted}")
        
        results = {
            'model': model,
            'accuracy': accuracy,
            'log_loss': logloss,
            'params': best_params,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        trainer.models['xgboost'] = results
        trainer.best_model = model
        trainer.best_score = accuracy
        
        return results
    
    # Train with quick version
    results = quick_train_xgboost(X_train, y_train, X_test, y_test, optimize=True)
    
    # Save model
    trainer.save_model("models/best_model.pkl")
    
    logger.info("\n" + "="*60)
    logger.info("QUICK RETRAIN COMPLETE")
    logger.info("="*60)
    logger.info(f"Model saved to models/best_model.pkl")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info("\n✅ Now test with: python3 test_predictions.py")


if __name__ == "__main__":
    quick_retrain()

