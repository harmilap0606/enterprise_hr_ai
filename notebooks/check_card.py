import sys
sys.stdout.reconfigure(encoding='utf-8')

card_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\docs\model_card.md'
with open(card_path, 'r', encoding='utf-8') as f:
    content = f.read()

limitation_idx = content.find('## Known Limitations')
print("Found '## Known Limitations' at char:", limitation_idx)

# Let's inspect the Known Limitations section
chunk = content[limitation_idx:limitation_idx+1500]
print("=== CURRENT KNOWN LIMITATIONS SECTION ===")
print(chunk)
