import sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\docs\data_relationships.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find('Open Issue #1')
if idx == -1:
    idx = content.lower().find('open issue')

print(f'Found "Open Issue #1" at char: {idx}')
# Show chunk safely
chunk = content[idx:idx+600]
print('=== CHUNK START ===')
print(chunk)
print('=== CHUNK END ===')
