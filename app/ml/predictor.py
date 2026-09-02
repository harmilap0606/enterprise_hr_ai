"""
Predictor Module.
Executes the exact feature engineering and scaling transformations established in Step 5,
then generates calibrated attrition risk probabilities using the production model and threshold.
"""

from typing import Dict, Any, Union
import pandas as pd
import numpy as np

from app.ml.model_loader import get_model, get_scaler, get_model_config
from app.utils.logger import logger

# Active categorical levels and their reference baseline dropped levels (drop_first=True)
CATEGORICAL_DUMMIES_MAP = {
    'BusinessTravel': ['Travel_Frequently', 'Travel_Rarely'], # 'Non-Travel' dropped
    'Department': ['Research & Development', 'Sales'],        # 'Human Resources' dropped
    'EducationField': [
        'Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'
    ], # 'Human Resources' dropped
    'JobRole': [
        'Human Resources', 'Laboratory Technician', 'Manager',
        'Manufacturing Director', 'Research Director', 'Research Scientist',
        'Sales Executive', 'Sales Representative'
    ], # 'Healthcare Representative' dropped
    'MaritalStatus': ['Married', 'Single'] # 'Divorced' dropped
}

BINARY_MAPS = {
    'Gender': {'Male': 1, 'Female': 0},
    'OverTime': {'Yes': 1, 'No': 0}
}


# Raw input columns required by the feature engineering formulas below.
# This list must stay in sync with the engineered-feature block (Step 5 formulas).
_REQUIRED_RAW_COLUMNS = [
    'MonthlyIncome', 'YearsAtCompany', 'YearsSinceLastPromotion',
    'JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction',
    'TotalWorkingYears',
]


def preprocess_features(raw_data: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> pd.DataFrame:
    """
    Transforms raw employee record(s) into the scaled 48-feature matrix expected by the model.
    Applies the identical pipeline engineered in Step 5.
    """
    if isinstance(raw_data, dict):
        df_input = pd.DataFrame([raw_data])
    elif isinstance(raw_data, pd.Series):
        df_input = pd.DataFrame([raw_data.to_dict()])
    elif isinstance(raw_data, pd.DataFrame):
        df_input = raw_data.copy()
    else:
        raise ValueError(f"Unsupported input type: {type(raw_data)}")

    # Upfront guard: check all raw columns needed by feature engineering BEFORE touching them.
    # This ensures a clear ValueError is raised immediately rather than a raw pandas KeyError
    # buried inside the feature computation loop.
    missing_raw = [c for c in _REQUIRED_RAW_COLUMNS if c not in df_input.columns]
    if missing_raw:
        raise ValueError(
            f"Missing required input columns for feature engineering: {missing_raw}. "
            f"All of the following columns must be present: {_REQUIRED_RAW_COLUMNS}"
        )

    df_feat = df_input.copy()
    
    # 1. Binary Mapping
    for col, mapping in BINARY_MAPS.items():
        if col in df_feat.columns:
            df_feat[col] = df_feat[col].map(mapping).fillna(0).astype(int)
            
    # 2. Engineered Features (Step 5 formula with +1 zero-division protection)
    df_feat['income_per_year_at_company'] = (
        df_feat['MonthlyIncome'] / (df_feat['YearsAtCompany'] + 1)
    )
    df_feat['years_since_promotion_ratio'] = (
        df_feat['YearsSinceLastPromotion'] / (df_feat['YearsAtCompany'] + 1)
    )
    df_feat['overall_satisfaction_score'] = (
        df_feat[['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
    )
    df_feat['experience_ratio'] = (
        df_feat['YearsAtCompany'] / (df_feat['TotalWorkingYears'] + 1)
    )
    
    # 3. Categorical One-Hot Encoding (matching model dummy columns)
    for base_col, dummy_vals in CATEGORICAL_DUMMIES_MAP.items():
        val_series = df_feat[base_col] if base_col in df_feat.columns else pd.Series([None] * len(df_feat))
        for d_val in dummy_vals:
            col_name = f"{base_col}_{d_val}"
            df_feat[col_name] = (val_series == d_val).astype(int)
            
    # 4. Standard Scaling of Continuous Features
    scaler = get_scaler()
    continuous_cols = list(scaler.feature_names_in_)
    
    missing_continuous = [c for c in continuous_cols if c not in df_feat.columns]
    if missing_continuous:
        raise ValueError(f"Missing required continuous features for scaling: {missing_continuous}")
        
    df_feat[continuous_cols] = scaler.transform(df_feat[continuous_cols])
    
    # 5. Exact Feature Alignment with Model Input Matrix (48 columns)
    model = get_model()
    expected_cols = list(model.feature_names_in_)
    
    missing_model_cols = [c for c in expected_cols if c not in df_feat.columns]
    if missing_model_cols:
        raise ValueError(f"Missing expected model input columns: {missing_model_cols}")
        
    X_model = df_feat[expected_cols].copy()
    return X_model


def predict_attrition_risk(raw_data: Union[Dict[str, Any], pd.Series]) -> Dict[str, Any]:
    """
    Computes attrition probability and assigns risk tier using the production model and threshold.
    Returns:
        {
            'probability': float,
            'risk_level': 'HIGH' | 'LOW',
            'threshold': float
        }
    """
    model = get_model()
    config = get_model_config()
    threshold = float(config.get("threshold", 0.40))
    
    X = preprocess_features(raw_data)
    prob = float(model.predict_proba(X)[0, 1])
    risk_level = "HIGH" if prob >= threshold else "LOW"
    
    return {
        "probability": round(prob, 4),
        "risk_level": risk_level,
        "threshold": threshold
    }
