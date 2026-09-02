import pandas as pd
import os

proc = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
gaps_file = os.path.join(proc, 'employee_skill_gaps.csv')

df_gaps = pd.read_csv(gaps_file, comment='#')
print(f"Loaded {len(df_gaps)} employee gap records")
print(df_gaps.head(3))

# Explode missing_skills
records = []
for _, r in df_gaps.iterrows():
    emp_id = r['EmployeeNumber']
    role = r['JobRole']
    ms = r['missing_skills']
    if pd.isna(ms) or ms.strip() == '' or ms.strip() == 'None':
        continue
    skills = [s.strip() for s in ms.split(';') if s.strip()]
    for s in skills:
        records.append({'EmployeeNumber': emp_id, 'JobRole': role, 'skill_name': s})

df_exploded = pd.DataFrame(records)
print(f"Total exploded (employee, skill) pairs: {len(df_exploded)}")

# Roll up
skill_counts = df_exploded.groupby('skill_name').size().reset_index(name='total_missing_count')
skill_counts = skill_counts.sort_values('total_missing_count', ascending=False).reset_index(drop=True)
print(f"Unique missing skills: {len(skill_counts)}")

# Severity rule: >=100 HIGH, >=50 MEDIUM, else LOW
def get_severity(cnt):
    if cnt >= 100:
        return 'HIGH'
    elif cnt >= 50:
        return 'MEDIUM'
    else:
        return 'LOW'

skill_counts['severity'] = skill_counts['total_missing_count'].apply(get_severity)
print("\nSeverity distribution:")
print(skill_counts['severity'].value_counts())

# Role breakdown and concentration flag (threshold: >= 80% from one role)
CONCENTRATION_THRESHOLD = 0.80

top_roles_list = []
is_conc_list = []

for _, r in skill_counts.iterrows():
    s_name = r['skill_name']
    sub = df_exploded[df_exploded['skill_name'] == s_name]
    role_dist = sub['JobRole'].value_counts()
    
    # Format top affected roles string
    role_strs = [f"{role} ({count})" for role, count in role_dist.items()]
    top_roles_str = ', '.join(role_strs)
    top_roles_list.append(top_roles_str)
    
    # Concentration calculation
    max_pct = (role_dist.iloc[0] / len(sub)) if len(sub) > 0 else 0
    is_conc = max_pct >= CONCENTRATION_THRESHOLD
    is_conc_list.append(is_conc)

skill_counts['top_affected_roles'] = top_roles_list
skill_counts['is_role_concentrated'] = is_conc_list

print("\nAll skills ranked:")
for idx, r in skill_counts.iterrows():
    conc_flag = "ROLE-CONCENTRATED" if r['is_role_concentrated'] else "CROSS-CUTTING"
    print(f"{idx+1:2d}. {r['skill_name']:<40} Count: {r['total_missing_count']:3d} | Severity: {r['severity']:<6} | {conc_flag}")
    print(f"    Affected roles: {r['top_affected_roles']}")
