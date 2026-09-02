import pandas as pd
import numpy as np
import os

proc = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
att = pd.read_csv(os.path.join(proc, 'employee_attrition_processed.csv'))
rsp = pd.read_csv(os.path.join(proc, 'role_skill_profiles.csv'))

# Parse role skills
role_skills = {}
for _, r in rsp.iterrows():
    role = r['ibm_job_role']
    if r['match_confidence'] == 'very_low':
        continue
    ess_str = r['top5_essential_skills']
    sw_str = r['top5_software_tools']
    
    ess_list = [s.split('(')[0].strip() for s in ess_str.split('|')]
    sw_list = [s.strip() for s in sw_str.split('|')]
    
    skills = [(s, 'essential') for s in ess_list] + [(s, 'software') for s in sw_list]
    role_skills[role] = skills

print(f"Roles with parsed skills: {len(role_skills)}")
for k, v in role_skills.items():
    print(f"  {k}: {len(v)} skills ({[s[0] for s in v[:3]]}...)")

# Simulation logic
np.random.seed(42)

rows = []
example_records = []

for idx, emp in att.iterrows():
    emp_num = emp['EmployeeNumber']
    role = emp['JobRole']
    tenure = emp['YearsAtCompany']
    training = emp['TrainingTimesLastYear']
    
    if role == 'Manager':
        rows.append({
            'EmployeeNumber': emp_num,
            'skill_name': 'N/A - Department-level analysis only',
            'skill_type': 'N/A',
            'has_skill': 0
        })
        continue
        
    prob = min(0.3 + 0.05 * tenure + 0.05 * training, 0.95)
    skills = role_skills[role]
    
    emp_skills_possessed = []
    for s_name, s_type in skills:
        has = int(np.random.rand() < prob)
        rows.append({
            'EmployeeNumber': emp_num,
            'skill_name': s_name,
            'skill_type': s_type,
            'has_skill': has
        })
        if has:
            emp_skills_possessed.append(s_name)
            
    if len(example_records) < 5:
        example_records.append({
            'EmployeeNumber': emp_num,
            'JobRole': role,
            'YearsAtCompany': tenure,
            'TrainingTimesLastYear': training,
            'possession_probability': round(prob, 4),
            'skills_possessed_count': len(emp_skills_possessed),
            'total_skills': len(skills),
            'skills_possessed': emp_skills_possessed
        })

df_skills = pd.DataFrame(rows)
print(f"\nTotal synthetic rows: {len(df_skills):,}")
print(f"Unique employees: {df_skills['EmployeeNumber'].nunique():,}")
print("\nValue counts of has_skill:")
print(df_skills['has_skill'].value_counts())

print("\nValue counts of skill_type:")
print(df_skills['skill_type'].value_counts())

print("\n5 Example Employees:")
for ex in example_records:
    print(f"\nEmployee #{ex['EmployeeNumber']} ({ex['JobRole']}):")
    print(f"  Tenure: {ex['YearsAtCompany']} yrs | Training: {ex['TrainingTimesLastYear']} times")
    print(f"  Probability: {ex['possession_probability']:.2f}")
    print(f"  Skills Possessed: {ex['skills_possessed_count']}/{ex['total_skills']}")
    print(f"  List: {', '.join(ex['skills_possessed'])}")
