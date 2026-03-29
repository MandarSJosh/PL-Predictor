"""
Model training utilities
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV

try:
    import xgboost as xgb
    HAS_XGB = True
except Exception:
    HAS_XGB = False
    xgb = None

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    lgb = None

try:
    import optuna
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False
    optuna = None

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _get_optuna_trials() -> int:
    try:
        return int(os.getenv("OPTUNA_TRIALS", "30"))
    except Exception:
        return 30


class ModelTrainer:
    def __init__(self, experiment_name: str = "premier_league_predictions"):
        self.experiment_name = experiment_name
        self.models: Dict[str, Dict] = {}
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.best_model = None
        self.best_score = 0.0
        self.calibrated_model = None

    def prepare_data(
        self, df: pd.DataFrame, feature_columns: List[str], target_col: str = "target"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logger.info("Preparing data for training...")
        available_features = [f for f in feature_columns if f in df.columns]
        X = df[available_features].fillna(0).values
        y = df[target_col].values

        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]

        y_mapped = y.copy().astype(int)
        y_mapped[y == -1] = 0
        y_mapped[y == 0] = 1
        y_mapped[y == 1] = 2

        self.target_mapping = {-1: 0, 0: 1, 1: 2}
        self.reverse_mapping = {0: -1, 1: 0, 2: 1}

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y_mapped[:split_idx], y_mapped[split_idx:]

        classes = np.unique(y_train)
        class_weights = compute_class_weight("balanced", classes=classes, y=y_train)
        self.class_weights = dict(zip(classes, class_weights))
        self.sample_weights = np.array([self.class_weights[y] for y in y_train])

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        self.feature_columns = available_features

        logger.info(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def _metrics(self, y_true, y_pred, y_proba) -> Dict:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "log_loss": log_loss(y_true, y_proba, labels=[0, 1, 2]),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        }

    def train_xgboost(self, X_train, y_train, X_test, y_test, optimize: bool) -> Dict:
        if not HAS_XGB:
            logger.info("XGBoost not installed, skipping.")
            return {}
        logger.info("Training XGBoost model...")

        if optimize and HAS_OPTUNA:
            def objective(trial):
                params = {
                    "objective": "multi:softprob",
                    "num_class": 3,
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "gamma": trial.suggest_float("gamma", 0, 5),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                    "random_state": 42,
                    "eval_metric": "mlogloss",
                }
                model = xgb.XGBClassifier(**params)
                model.fit(X_train, y_train, sample_weight=self.sample_weights, eval_set=[(X_test, y_test)], verbose=False)
                preds = model.predict(X_test)
                return f1_score(y_test, preds, average="macro", zero_division=0)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=_get_optuna_trials(), show_progress_bar=False)
            best_params = study.best_params
            best_params.update({"objective": "multi:softprob", "num_class": 3, "random_state": 42, "eval_metric": "mlogloss"})
        else:
            best_params = {
                "objective": "multi:softprob",
                "num_class": 3,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 400,
                "random_state": 42,
                "eval_metric": "mlogloss",
            }

        model = xgb.XGBClassifier(**best_params)
        model.fit(X_train, y_train, sample_weight=self.sample_weights, eval_set=[(X_test, y_test)], verbose=False)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = self._metrics(y_test, y_pred, y_proba)
        results = {"model": model, **metrics, "params": best_params, "predictions": y_pred, "probabilities": y_proba}
        self.models["xgboost"] = results
        logger.info(f"XGBoost - Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        return results

    def train_lightgbm(self, X_train, y_train, X_test, y_test, optimize: bool) -> Dict:
        if not HAS_LGB:
            logger.info("LightGBM not installed, skipping.")
            return {}
        logger.info("Training LightGBM model...")

        if optimize and HAS_OPTUNA:
            def objective(trial):
                params = {
                    "objective": "multiclass",
                    "num_class": 3,
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 800),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                    "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
                    "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
                    "random_state": 42,
                    "class_weight": self.class_weights,
                    "verbose": -1,
                }
                model = lgb.LGBMClassifier(**params)
                model.fit(X_train, y_train, sample_weight=self.sample_weights, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                preds = model.predict(X_test)
                return f1_score(y_test, preds, average="macro", zero_division=0)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=_get_optuna_trials(), show_progress_bar=False)
            best_params = study.best_params
            best_params.update({"objective": "multiclass", "num_class": 3, "random_state": 42, "class_weight": self.class_weights, "verbose": -1})
        else:
            best_params = {
                "objective": "multiclass",
                "num_class": 3,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_estimators": 400,
                "random_state": 42,
                "class_weight": self.class_weights,
                "verbose": -1,
            }

        model = lgb.LGBMClassifier(**best_params)
        model.fit(X_train, y_train, sample_weight=self.sample_weights, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = self._metrics(y_test, y_pred, y_proba)
        results = {"model": model, **metrics, "params": best_params, "predictions": y_pred, "probabilities": y_proba}
        self.models["lightgbm"] = results
        logger.info(f"LightGBM - Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        return results

    def train_baseline(self, X_train, y_train, X_test, y_test) -> Dict:
        logger.info("Training RandomForest baseline...")
        model = RandomForestClassifier(n_estimators=300, random_state=42, class_weight=self.class_weights)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        metrics = self._metrics(y_test, y_pred, y_proba)
        results = {"model": model, **metrics, "params": {}, "predictions": y_pred, "probabilities": y_proba}
        self.models["rf"] = results
        logger.info(f"RF - Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        return results

    def train_ensemble(self, X_train, y_train, X_test, y_test) -> Dict:
        logger.info("Training ensemble model...")
        estimators = [(name, res["model"]) for name, res in self.models.items() if res]
        if not estimators:
            return {}
        ensemble = VotingClassifier(estimators=estimators, voting="soft")
        ensemble.fit(X_train, y_train)
        y_pred = ensemble.predict(X_test)
        y_proba = ensemble.predict_proba(X_test)
        metrics = self._metrics(y_test, y_pred, y_proba)
        results = {"model": ensemble, **metrics, "predictions": y_pred, "probabilities": y_proba}
        self.models["ensemble"] = results
        logger.info(f"Ensemble - Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        return results

    def train_stacking(self, X_train, y_train, X_test, y_test) -> Dict:
        logger.info("Training stacking ensemble...")
        estimators = [(name, res["model"]) for name, res in self.models.items() if res]
        if not estimators:
            return {}
        stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=300), passthrough=True, cv=3)
        stacking.fit(X_train, y_train)
        y_pred = stacking.predict(X_test)
        y_proba = stacking.predict_proba(X_test)
        metrics = self._metrics(y_test, y_pred, y_proba)
        results = {"model": stacking, **metrics, "predictions": y_pred, "probabilities": y_proba}
        self.models["stacking"] = results
        logger.info(f"Stacking - Accuracy: {metrics['accuracy']:.4f}, Macro F1: {metrics['macro_f1']:.4f}")
        return results

    def train_all(self, X_train, y_train, X_test, y_test, optimize: bool = True) -> Dict:
        logger.info("Training all models...")
        self.train_xgboost(X_train, y_train, X_test, y_test, optimize)
        self.train_lightgbm(X_train, y_train, X_test, y_test, optimize)
        self.train_baseline(X_train, y_train, X_test, y_test)
        self.train_ensemble(X_train, y_train, X_test, y_test)
        if os.getenv("SKIP_STACKING", "").lower() not in {"1", "true", "yes"}:
            self.train_stacking(X_train, y_train, X_test, y_test)

        best_name = max(self.models.items(), key=lambda x: (x[1].get("macro_f1", 0), x[1].get("accuracy", 0)))[0]
        self.best_model = self.models[best_name]["model"]
        self.best_score = self.models[best_name].get("macro_f1", 0)
        logger.info(f"Best model: {best_name} with macro F1: {self.best_score:.4f}")

        if os.getenv("CALIBRATE_MODEL", "1").lower() in {"1", "true", "yes"}:
            method = os.getenv("CALIBRATION_METHOD", "isotonic")
            logger.info(f"Calibrating best model with {method} calibration...")
            calibrator = CalibratedClassifierCV(self.best_model, method=method, cv=5)
            calibrator.fit(X_train, y_train)
            self.calibrated_model = calibrator
            logger.info("Calibration complete.")
        return self.models

    def save_model(self, model_path: str = "models/best_model.pkl"):
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            "model": self.best_model,
            "calibrated_model": self.calibrated_model,
            "scaler": self.scaler,
            "feature_columns": self.feature_columns,
            "models": self.models,
        }
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
