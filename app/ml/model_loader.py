"""
Model Loader Module.
Loads and caches the production scikit-learn model, standard scaler, and configuration JSON.
"""

import json
from typing import Dict, Any, Tuple
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from app.utils.config import MODEL_PATH, SCALER_PATH, MODEL_CONFIG_PATH, DECISION_THRESHOLD
from app.utils.logger import logger

_model: LogisticRegression = None
_scaler: StandardScaler = None
_model_config: Dict[str, Any] = None

def load_artifacts() -> Tuple[LogisticRegression, StandardScaler, Dict[str, Any]]:
    """Loads model, scaler, and config artifacts from disk."""
    global _model, _scaler, _model_config
    
    if _model is not None and _scaler is not None and _model_config is not None:
        return _model, _scaler, _model_config
        
    logger.info("Loading production ML artifacts from disk...")
    
    # 1. Load config
    if MODEL_CONFIG_PATH.exists():
        with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
            _model_config = json.load(f)
        logger.info(f"Loaded model config: {_model_config.get('model')} (threshold={_model_config.get('threshold')})")
    else:
        _model_config = {
            "model": "logistic_regression_balanced",
            "threshold": DECISION_THRESHOLD,
            "trained_on": "features_scaled.csv"
        }
        logger.warning(f"{MODEL_CONFIG_PATH.name} not found. Created fallback config with threshold {DECISION_THRESHOLD}")
        
    # 2. Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Production model artifact not found at {MODEL_PATH}")
    _model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded production model from {MODEL_PATH.name}")
    
    # 3. Load scaler
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Feature scaler artifact not found at {SCALER_PATH}")
    _scaler = joblib.load(SCALER_PATH)
    logger.info(f"Loaded feature scaler from {SCALER_PATH.name}")
    
    return _model, _scaler, _model_config


def get_model() -> LogisticRegression:
    """Returns the cached production model instance."""
    global _model
    if _model is None:
        load_artifacts()
    return _model


def get_scaler() -> StandardScaler:
    """Returns the cached feature scaler instance."""
    global _scaler
    if _scaler is None:
        load_artifacts()
    return _scaler


def get_model_config() -> Dict[str, Any]:
    """Returns the cached model config dict."""
    global _model_config
    if _model_config is None:
        load_artifacts()
    return _model_config
