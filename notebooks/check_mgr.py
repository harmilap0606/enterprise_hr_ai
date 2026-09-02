import pandas as pd
import os

data_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed\employee_attrition_processed.csv'
df = pd.read_csv(data_path)
print("Overall shape:", df.shape)

mgrs = df[df['JobRole'] == 'Manager']
print("Manager count:", len(mgrs))
print("\nDepartment counts for Managers:")
print(mgrs['Department'].value_counts())

# Overall attrition rate in dataset:
overall_att = (df['Attrition'] == 'Yes').mean() * 100
print(f"\nOverall company attrition rate: {overall_att:.2f}%")

# Manager overall attrition
mgr_att = (mgrs['Attrition'] == 'Yes').mean() * 100
print(f"Manager overall attrition rate: {mgr_att:.2f}% ({sum(mgrs['Attrition'] == 'Yes')}/{len(mgrs)})")

# Department breakdown for managers
for dept, g in mgrs.groupby('Department'):
    att_rate = (g['Attrition'] == 'Yes').mean() * 100
    att_count = sum(g['Attrition'] == 'Yes')
    ot_rate = (g['OverTime'] == 'Yes').mean() * 100
    print(f"\nDept: {dept} (N={len(g)})")
    print(f"  Attrition: {att_rate:.2f}% ({att_count}/{len(g)})")
    print(f"  MonthlyIncome (mean): {g['MonthlyIncome'].mean():.2f}")
    print(f"  JobSatisfaction (mean): {g['JobSatisfaction'].mean():.2f}")
    print(f"  WorkLifeBalance (mean): {g['WorkLifeBalance'].mean():.2f}")
    print(f"  YearsAtCompany (mean): {g['YearsAtCompany'].mean():.2f}")
    print(f"  OverTime (% Yes): {ot_rate:.2f}%")

# Also check OverTime vs Attrition within Managers
print("\nOvertime vs Attrition within Managers:")
print(pd.crosstab(mgrs['OverTime'], mgrs['Attrition'], margins=True))
