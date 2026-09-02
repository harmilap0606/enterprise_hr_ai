import pandas as pd
import os, time, hashlib

proc = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
p_intel = os.path.join(proc, 'employee_intelligence.csv')
p_att = os.path.join(proc, 'employee_attrition_processed.csv')

df_intel = pd.read_csv(p_intel, comment='#')
df_att = pd.read_csv(p_att)

print('=== 1. employee_intelligence.csv ===')
print('len(df_intel):', len(df_intel))
print('df_intel unique EmployeeNumber:', df_intel['EmployeeNumber'].nunique())
print('Does 100 exist in employee_intelligence.csv?:', 100 in df_intel['EmployeeNumber'].values)
print('File size:', os.path.getsize(p_intel), 'bytes')
print('Last modified:', time.ctime(os.path.getmtime(p_intel)))
with open(p_intel, 'rb') as f:
    print('SHA256:', hashlib.sha256(f.read()).hexdigest())

print('\n=== 2. employee_attrition_processed.csv ===')
print('len(df_att):', len(df_att))
print('df_att unique EmployeeNumber:', df_att['EmployeeNumber'].nunique())
print('Does 100 exist in employee_attrition_processed.csv?:', 100 in df_att['EmployeeNumber'].values)

# Check closest IDs around 100
ids = sorted(df_att['EmployeeNumber'].tolist())
closest = [x for x in ids if 90 <= x <= 110]
print('EmployeeNumber values between 90 and 110:', closest)

# Check why 100 worked in previous test
raw_100 = df_att[df_att['EmployeeNumber'] == 100]
print('\nIs raw_100 empty?:', raw_100.empty)
if not raw_100.empty:
    print('Employee 100 row found in raw data:', raw_100[['EmployeeNumber', 'JobRole', 'Department']].to_dict(orient='records'))
