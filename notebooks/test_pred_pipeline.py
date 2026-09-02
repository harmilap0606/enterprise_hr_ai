import pandas as pd
import numpy as np
import joblib, json, os

proc = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
models = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\models'

# Load model and scaler
model = joblib.load(os.path.join(models, 'attrition_pipeline.joblib'))
scaler = joblib.load(os.path.join(models, 'scaler.joblib'))
with open(os.path.join(models, 'model_config.json')) as f:
    cfg = json.load(f)

# Load raw employee row
att = pd.read_csv(os.path.join(proc, 'employee_attrition_processed.csv'))
fs = pd.read_csv(os.path.join(proc, 'features_scaled.csv'))

# Preprocessing function for a single dict or row
def preprocess_employee(raw_dict):
    df_single = pd.DataFrame([raw_dict])
    
    # 1. Binary encoding
    binary_maps = {
        'Gender': {'Male': 1, 'Female': 0},
        'OverTime': {'Yes': 1, 'No': 0}
    }
    for col, mapping in binary_maps.items():
        if col in df_single.columns:
            df_single[col] = df_single[col].map(mapping)
            
    # 2. Engineered features
    df_single['income_per_year_at_company'] = (
        df_single['MonthlyIncome'] / (df_single['YearsAtCompany'] + 1)
    )
    df_single['years_since_promotion_ratio'] = (
        df_single['YearsSinceLastPromotion'] / (df_single['YearsAtCompany'] + 1)
    )
    df_single['overall_satisfaction_score'] = (
        df_single[['JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction']].mean(axis=1)
    )
    df_single['experience_ratio'] = (
        df_single['YearsAtCompany'] / (df_single['TotalWorkingYears'] + 1)
    )
    
    # 3. Categorical one-hot encoding columns matching model expectations
    # Dummy columns to generate if categories match:
    cat_columns_values = {
        'BusinessTravel': ['Travel_Frequently', 'Travel_Rarely'], # Non-Travel dropped
        'Department': ['Research & Development', 'Sales'],        # Human Resources dropped
        'EducationField': ['Life Sciences', 'Marketing', 'Medical', 'Other', 'Technical Degree'], # Human Resources dropped
        'JobRole': [
            'Human Resources', 'Laboratory Technician', 'Manager',
            'Manufacturing Director', 'Research Director', 'Research Scientist',
            'Sales Executive', 'Sales Representative'
        ], # Healthcare Representative dropped
        'MaritalStatus': ['Married', 'Single'] # Divorced dropped
    }
    
    for base_col, active_vals in cat_columns_values.items():
        val = df_single[base_col].iloc[0] if base_col in df_single.columns else None
        for av in active_vals:
            dummy_col = f"{base_col}_{av}"
            df_single[dummy_col] = 1 if val == av else 0
            
    # 4. Scale continuous features
    continuous_cols = list(scaler.feature_names_in_)
    df_single[continuous_cols] = scaler.transform(df_single[continuous_cols])
    
    # 5. Order columns exactly as model expects
    expected_cols = list(model.feature_names_in_)
    X_ready = df_single[expected_cols].copy()
    
    # Predict
    prob = float(model.predict_proba(X_ready)[0, 1])
    risk_level = "HIGH" if prob >= cfg['threshold'] else "LOW"
    
    return {
        'probability': round(prob, 4),
        'risk_level': risk_level,
        'threshold': cfg['threshold']
    }

# Test on first 5 employees
print("Testing prediction on first 5 raw employees:")
for i in range(5):
    raw_row = att.iloc[i].to_dict()
    res = preprocess_employee(raw_row)
    
    # Expected from features_scaled
    exp_prob = model.predict_proba(fs.drop(columns=['Attrition']).iloc[[i]])[0, 1]
    print(f"Emp #{raw_row['EmployeeNumber']}: Pred={res['probability']} ({res['risk_level']}) | Expected={exp_prob:.4f} | Match={abs(res['probability'] - exp_prob) < 1e-4}")
