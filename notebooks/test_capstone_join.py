import pandas as pd
import numpy as np
import joblib, json, os

proc = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
models = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\models'

# Anchor
df_anchor = pd.read_csv(os.path.join(proc, 'employee_attrition_processed.csv'))
print(f"Anchor start: {len(df_anchor)} rows")

# 1. Risk scoring
fs = pd.read_csv(os.path.join(proc, 'features_scaled.csv'))
model = joblib.load(os.path.join(models, 'attrition_pipeline.joblib'))
with open(os.path.join(models, 'model_config.json')) as f:
    cfg = json.load(f)

X = fs.drop(columns=['Attrition'])
probs = model.predict_proba(X)[:, 1]
df_risk = pd.DataFrame({
    'EmployeeNumber': df_anchor['EmployeeNumber'],
    'RiskScore': probs.round(4),
    'RiskLevel': ['HIGH' if p >= cfg['threshold'] else 'LOW' for p in probs]
})

df_merged = df_anchor[['EmployeeNumber', 'Department', 'JobRole']].copy()
df_merged = df_merged.merge(df_risk, on='EmployeeNumber', how='left')
print(f"After Join 1 (Risk Score): {len(df_merged)} rows")

# 2. Engagement survey
df_eng = pd.read_csv(os.path.join(proc, 'employee_intelligence_partial.csv'))
eng_cols = df_eng[['EmployeeNumber', 'Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score']].copy()
eng_cols.rename(columns={
    'Engagement Score': 'EngagementScore',
    'Satisfaction Score': 'SatisfactionScore',
    'Work-Life Balance Score': 'WorkLifeBalanceScore'
}, inplace=True)

df_merged = df_merged.merge(eng_cols, on='EmployeeNumber', how='left')
print(f"After Join 2 (Engagement Survey): {len(df_merged)} rows")

# 3. O*NET role mappings
df_rsp = pd.read_csv(os.path.join(proc, 'role_skill_profiles.csv'))
onet_cols = df_rsp[['ibm_job_role', 'onet_title', 'match_confidence']].copy()
onet_cols.rename(columns={
    'ibm_job_role': 'JobRole',
    'onet_title': 'ONET_Title',
    'match_confidence': 'ONET_Confidence'
}, inplace=True)

df_merged = df_merged.merge(onet_cols, on='JobRole', how='left')
print(f"After Join 3 (O*NET Role Profile): {len(df_merged)} rows")

# 4. Skill gaps
df_gaps = pd.read_csv(os.path.join(proc, 'employee_skill_gaps.csv'), comment='#')
gaps_cols = df_gaps[['EmployeeNumber', 'gap_count', 'severity']].copy()
gaps_cols.rename(columns={
    'gap_count': 'SkillGapCount',
    'severity': 'SkillGapSeverity'
}, inplace=True)

df_merged = df_merged.merge(gaps_cols, on='EmployeeNumber', how='left')
# Fill Manager values
df_merged.loc[df_merged['JobRole'] == 'Manager', 'SkillGapSeverity'] = 'N/A - Manager'
print(f"After Join 4 (Skill Gaps): {len(df_merged)} rows")

# 5. Recommendations
df_recs = pd.read_csv(os.path.join(proc, 'employee_recommendations.csv'), comment='#')
recs_cols = df_recs[['EmployeeNumber', 'top_3_recommendations']].copy()
recs_cols.rename(columns={'top_3_recommendations': 'Top3Recommendations'}, inplace=True)

df_merged = df_merged.merge(recs_cols, on='EmployeeNumber', how='left')
df_merged.loc[df_merged['JobRole'] == 'Manager', 'Top3Recommendations'] = 'N/A - Manager (use Department-level analysis)'
print(f"After Join 5 (Recommendations): {len(df_merged)} rows")

# Check final columns
expected_cols = [
    'EmployeeNumber', 'Department', 'JobRole', 'RiskScore', 'RiskLevel',
    'EngagementScore', 'SatisfactionScore', 'WorkLifeBalanceScore',
    'ONET_Title', 'ONET_Confidence', 'SkillGapCount', 'SkillGapSeverity',
    'Top3Recommendations'
]
df_final = df_merged[expected_cols].copy()
print("\nFinal shape:", df_final.shape)
print("Final columns:", df_final.columns.tolist())
print("\nFirst 3 rows:")
print(df_final.head(3).to_string())

print("\nManager rows sample:")
print(df_final[df_final['JobRole'] == 'Manager'].head(2).to_string())

# Summary stats
print("\n=== SUMMARY STATS ===")
print("\nRiskLevel distribution:")
print(df_final['RiskLevel'].value_counts())

print("\nSkillGapSeverity distribution:")
print(df_final['SkillGapSeverity'].value_counts())

# Completeness check:
# An employee has complete data across all 5 sources if:
# - RiskScore is not null
# - EngagementScore is not null (real survey data)
# - ONET_Title is not null
# - SkillGapCount is not null (not a manager)
# - Top3Recommendations is not null and not Manager
complete_mask = (
    df_final['RiskScore'].notnull() &
    df_final['EngagementScore'].notnull() &
    df_final['ONET_Title'].notnull() &
    df_final['SkillGapCount'].notnull() &
    (df_final['JobRole'] != 'Manager')
)
n_complete = complete_mask.sum()
n_partial = len(df_final) - n_complete
print(f"\nData completeness across all 5 sources:")
print(f"  Complete data (all 5 sources non-null, non-manager): {n_complete} ({n_complete/len(df_final)*100:.1f}%)")
print(f"  Partial data (missing engagement survey and/or Manager): {n_partial} ({n_partial/len(df_final)*100:.1f}%)")

# Detailed reasons for partial:
missing_eng = df_final['EngagementScore'].isnull()
is_mgr = df_final['JobRole'] == 'Manager'
print(f"    - Missing engagement survey: {missing_eng.sum()}")
print(f"    - Manager (excluded from skill gaps): {is_mgr.sum()}")
print(f"    - Both missing engagement AND is Manager: {(missing_eng & is_mgr).sum()}")
