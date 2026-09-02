"""
Central Configuration Module.
Stores paths to processed data, model artifacts, and dynamically loads model threshold from model_config.json.
"""

import os
import json
from pathlib import Path
from app.utils.logger import logger

# Base project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data paths
DATA_DIR = BASE_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

EMPLOYEE_ATTRITION_PATH = PROCESSED_DATA_DIR / "employee_attrition_processed.csv"
EMPLOYEE_INTELLIGENCE_PATH = PROCESSED_DATA_DIR / "employee_intelligence.csv"
EMPLOYEE_INTELLIGENCE_PARTIAL_PATH = PROCESSED_DATA_DIR / "employee_intelligence_partial.csv"
ROLE_SKILL_PROFILES_PATH = PROCESSED_DATA_DIR / "role_skill_profiles.csv"
EMPLOYEE_SKILL_GAPS_PATH = PROCESSED_DATA_DIR / "employee_skill_gaps.csv"
ORGANIZATION_SKILL_GAPS_PATH = PROCESSED_DATA_DIR / "organization_skill_gaps.csv"
EMPLOYEE_RECOMMENDATIONS_PATH = PROCESSED_DATA_DIR / "employee_recommendations.csv"
JOBROLE_ONET_MAPPING_PATH = EXTERNAL_DATA_DIR / "jobrole_onet_mapping.csv"
PREDICTIONS_DIR = DATA_DIR / "predictions"
PREDICTION_LOG_PATH = PREDICTIONS_DIR / "prediction_log.csv"

# Model paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "attrition_pipeline.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
MODEL_CONFIG_PATH = MODELS_DIR / "model_config.json"
MODEL_REGISTRY_PATH = MODELS_DIR / "model_registry.json"

def load_decision_threshold(default: float = 0.40) -> float:
    """Reads production decision threshold dynamically from model_config.json."""
    if MODEL_CONFIG_PATH.exists():
        try:
            with open(MODEL_CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                val = float(cfg.get("threshold", default))
                logger.info(f"Loaded decision threshold {val} from {MODEL_CONFIG_PATH.name}")
                return val
        except Exception as e:
            logger.warning(f"Error reading {MODEL_CONFIG_PATH.name}: {e}. Defaulting to {default}")
            return default
    logger.warning(f"{MODEL_CONFIG_PATH.name} not found. Using default threshold {default}")
    return default

DECISION_THRESHOLD = load_decision_threshold()
