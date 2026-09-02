import pandas as pd
import numpy as np
import os

proc_dir = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
attrition_path = os.path.join(proc_dir, 'employee_attrition_processed.csv')
eng_path = os.path.join(proc_dir, 'engagement_processed.csv')

att = pd.read_csv(attrition_path)
eng = pd.read_csv(eng_path)

print("att shape:", att.shape)
print("eng shape:", eng.shape)

# Check join keys
print("EmployeeNumber in att:", 'EmployeeNumber' in att.columns)
print("Employee ID in eng:", 'Employee ID' in eng.columns)

# Perform left join
joined = att.merge(eng, left_on='EmployeeNumber', right_on='Employee ID', how='left', suffixes=('', '_eng'))
print("joined shape:", joined.shape)
print("joined rows:", len(joined))

non_null = joined['Engagement Score'].notnull().sum()
null_cnt = joined['Engagement Score'].isnull().sum()
print(f"Engagement Score non-null: {non_null}, null: {null_cnt}")

# Subset for analysis
sub = joined[joined['Engagement Score'].notnull()].copy()
print("sub shape:", sub.shape)

# Encode Attrition as binary
sub['Attrition_num'] = (sub['Attrition'] == 'Yes').astype(int)

# Analysis 1: Correlation with Attrition
cols = ['Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score']
for c in cols:
    corr_pearson = sub[c].corr(sub['Attrition_num'], method='pearson')
    corr_spearman = sub[c].corr(sub['Attrition_num'], method='spearman')
    print(f"Corr {c} with Attrition: pearson={corr_pearson:.4f}, spearman={corr_spearman:.4f}")

# Analysis 2: Mean scores for leavers vs stayers
print("\nMean scores leavers vs stayers:")
for c in cols:
    mean_stayer = sub[sub['Attrition'] == 'No'][c].mean()
    mean_leaver = sub[sub['Attrition'] == 'Yes'][c].mean()
    gap = mean_leaver - mean_stayer
    print(f"{c}: Stayers={mean_stayer:.4f}, Leavers={mean_leaver:.4f}, Gap={gap:.4f}")

# Overall leavers in subset:
n_leavers = (sub['Attrition'] == 'Yes').sum()
n_stayers = (sub['Attrition'] == 'No').sum()
print(f"In subset: Stayers={n_stayers}, Leavers={n_leavers} (Attrition rate: {n_leavers/len(sub)*100:.2f}%)")

# Analysis 3: Cross-reference with OverTime
print("\nOverTime cross-ref:")
ot_yes = sub[sub['OverTime'] == 'Yes']
ot_no = sub[sub['OverTime'] == 'No']
print(f"OT Yes (n={len(ot_yes)}), OT No (n={len(ot_no)})")
for c in cols:
    mean_ot = ot_yes[c].mean()
    mean_no_ot = ot_no[c].mean()
    diff = mean_ot - mean_no_ot
    print(f"{c}: OT Yes={mean_ot:.4f}, OT No={mean_no_ot:.4f}, Diff={diff:.4f}")

