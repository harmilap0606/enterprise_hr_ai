import pandas as pd, os

PROC = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed'
EXT  = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\external'

files = {
    'occupation_master':          os.path.join(PROC, 'occupation_master.csv'),
    'essential_skills_processed': os.path.join(PROC, 'essential_skills_processed.csv'),
    'software_skills_processed':  os.path.join(PROC, 'software_skills_processed.csv'),
    'employee_attrition_processed': os.path.join(PROC, 'employee_attrition_processed.csv'),
    'jobrole_onet_mapping':       os.path.join(EXT,  'jobrole_onet_mapping.csv'),
}

for name, path in files.items():
    if not os.path.exists(path):
        print(f'MISSING: {name}')
        continue
    df = pd.read_csv(path)
    print(f'=== {name} ===')
    print(f'  Shape: {df.shape}')
    print(f'  Columns: {df.columns.tolist()}')
    # Show first 3 rows to understand data
    print(df.head(3).to_string())
    print()
