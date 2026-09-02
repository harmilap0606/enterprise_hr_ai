import pandas as pd
import numpy as np
import os

proc = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
skills_path = os.path.join(proc, 'employee_skills_synthetic.csv')
att_path = os.path.join(proc, 'employee_attrition_processed.csv')
rsp_path = os.path.join(proc, 'role_skill_profiles.csv')

skills_df = pd.read_csv(skills_path, comment='#')
att_df = pd.read_csv(att_path)
rsp_df = pd.read_csv(rsp_path)

print(f"skills_df shape: {skills_df.shape}")
print(f"att_df shape: {att_df.shape}")

# Exclude Managers
mgr_ids = set(att_df[att_df['JobRole'] == 'Manager']['EmployeeNumber'])
non_mgr_skills = skills_df[~skills_df['EmployeeNumber'].isin(mgr_ids)].copy()
non_mgr_att = att_df[~att_df['EmployeeNumber'].isin(mgr_ids)].copy()

print(f"Non-manager employees: {non_mgr_att['EmployeeNumber'].nunique()}")
print(f"Non-manager skill records: {len(non_mgr_skills)}")

# Sanity check on sample of 30 random employees
sample_30_ids = non_mgr_att['EmployeeNumber'].sample(n=30, random_state=42).tolist()
sample_att = non_mgr_att[non_mgr_att['EmployeeNumber'].isin(sample_30_ids)].copy()

# Compute theoretical prob: min(0.3 + 0.05 * tenure + 0.05 * training, 0.95)
sample_att['theoretical_prob'] = sample_att.apply(
    lambda r: min(0.30 + 0.05 * r['YearsAtCompany'] + 0.05 * r['TrainingTimesLastYear'], 0.95),
    axis=1
)

# Compute realized rate:
realized = non_mgr_skills[non_mgr_skills['EmployeeNumber'].isin(sample_30_ids)].groupby('EmployeeNumber').agg(
    total_skills=('has_skill', 'count'),
    possessed_skills=('has_skill', 'sum')
).reset_index()
realized['realized_rate'] = realized['possessed_skills'] / realized['total_skills']

merged_sample = sample_att.merge(realized, on='EmployeeNumber')
corr = merged_sample['theoretical_prob'].corr(merged_sample['realized_rate'])
print(f"\n30-employee sample correlation (theoretical vs realized): {corr:.4f}")
print("First 5 of sample:")
for _, r in merged_sample.head(5).iterrows():
    print(f"  Emp #{r['EmployeeNumber']}: JobRole={r['JobRole']}, Tenure={r['YearsAtCompany']}, Training={r['TrainingTimesLastYear']} | Theory={r['theoretical_prob']:.2f}, Realized={r['realized_rate']:.2f} ({r['possessed_skills']}/{r['total_skills']})")

# Full gap computation for all non-manager employees
# Group skills by employee
gap_records = []
emp_jobrole_map = dict(zip(att_df['EmployeeNumber'], att_df['JobRole']))

for emp_id, group in non_mgr_skills.groupby('EmployeeNumber'):
    role = emp_jobrole_map[emp_id]
    total_req = len(group)
    missing = group[group['has_skill'] == 0]['skill_name'].tolist()
    gap_cnt = len(missing)
    gap_pct = gap_cnt / total_req if total_req > 0 else 0.0
    
    if gap_pct >= 0.70:
        severity = 'HIGH'
    elif gap_pct >= 0.40:
        severity = 'MEDIUM'
    else:
        severity = 'LOW'
        
    gap_records.append({
        'EmployeeNumber': emp_id,
        'JobRole': role,
        'missing_skills': '; '.join(missing) if missing else 'None',
        'gap_count': gap_cnt,
        'total_required': total_req,
        'gap_percentage': round(gap_pct, 4),
        'severity': severity
    })

gap_df = pd.DataFrame(gap_records)
print(f"\nTotal gap records (non-managers): {len(gap_df)}")
print("\nSeverity Distribution:")
print(gap_df['severity'].value_counts())
print("\nSeverity Percentage:")
print(gap_df['severity'].value_counts(normalize=True) * 100)

print("\nSample of 3 employee gap profiles:")
for _, r in gap_df.head(3).iterrows():
    print(f"Emp #{r['EmployeeNumber']} ({r['JobRole']}): {r['gap_count']}/{r['total_required']} missing ({r['gap_percentage']*100:.1f}%) -> {r['severity']}")
    print(f"  Missing: {r['missing_skills']}")
