import sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\docs\data_relationships.md'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# The note to append goes after Open Issue #1 block (before #2)
old = (
    "1. **JobRole \u2194 O*NET Title gap** \u2014 IBM HR job roles do not match O*NET Titles exactly.\n"
    "   Requires a manual/fuzzy mapping table before the role-intelligence step (Day 1 nb 10).\n"
    "   Action: create `data/external/jobrole_onet_mapping.csv` in Step 5."
)
new = (
    "1. **JobRole \u2194 O*NET Title gap** \u2014 IBM HR job roles do not match O*NET Titles exactly.\n"
    "   Requires a manual/fuzzy mapping table before the role-intelligence step (Day 1 nb 10).\n"
    "   Action: create `data/external/jobrole_onet_mapping.csv` in Step 5.\n"
    "   **Update (Step 10):** Healthcare Representative and Sales Representative both map to\n"
    "   O\\*NET code `41-3091.00` (Sales Representatives of Services) \u2014 no better alternative\n"
    "   exists for either in `occupation_master.csv`. Any skill-gap output for these two IBM roles\n"
    "   will be identical at the O\\*NET layer. This is a mapping limitation, not a modeling error,\n"
    "   and must be labeled as such if surfaced in the Day 4 dashboard."
)

if old in content:
    updated = content.replace(old, new, 1)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(updated)
    print('DONE: Appended note under Open Issue #1.')
    print(f'Old length: {len(content)}  New length: {len(updated)}  Delta: +{len(updated)-len(content)} chars')
else:
    # Try to find it and report what we see
    idx = content.find('JobRole')
    print(f'WARNING: exact string not found. First JobRole at char {idx}:')
    print(repr(content[idx:idx+300]))
