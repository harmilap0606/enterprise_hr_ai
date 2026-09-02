import pandas as pd

occ = pd.read_csv(r'C:\Users\ASUS\Desktop\enterprise_hr_ai\data\processed\occupation_master.csv')
soc_col   = [c for c in occ.columns if 'soc' in c.lower() or 'code' in c.lower()][0]
title_col = [c for c in occ.columns if 'title' in c.lower()][0]

# Targeted lookups for the roles we need
targeted = [
    ('sales',             ['sales']),
    ('human resources',   ['human resource']),
    ('manager',           ['manager']),
    ('director',          ['director']),
    ('research',          ['research']),
    ('scientist',         ['scientist']),
    ('manufacturing',     ['manufactur']),
    ('production ops',    ['production', 'operations manager']),
    ('computer info research', ['computer and information research']),
]

for label, kws in targeted:
    combined = occ[occ[title_col].str.lower().str.contains('|'.join(kws), na=False, regex=True)]
    print(f'--- {label.upper()} ({len(combined)} rows) ---')
    for _, row in combined.iterrows():
        print(f'  {row[soc_col]}  |  {row[title_col]}')
    print()
