import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\ASUS\Desktop\enterprise_hr_ai\notebooks\05_feature_engineering.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        print(f"=== CELL {i} ===")
        print(src)
        print()
