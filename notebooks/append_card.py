import sys
sys.stdout.reconfigure(encoding='utf-8')

card_path = r'C:\Users\ASUS\Desktop\enterprise_hr_ai\docs\model_card.md'
with open(card_path, 'r', encoding='utf-8') as f:
    text = f.read()

target = (
    "5. **No causal inference:** SHAP values identify statistical associations, not causal drivers.\n"
    "   For example, 'OverTime' being the top driver does not prove that reducing overtime will\n"
    "   reduce attrition — it means employees who work overtime tend to leave more often.\n"
    "   Interventions should be designed with HR domain expertise, not derived mechanically from\n"
    "   model outputs."
)

addition = (
    "5. **No causal inference:** SHAP values identify statistical associations, not causal drivers.\n"
    "   For example, 'OverTime' being the top driver does not prove that reducing overtime will\n"
    "   reduce attrition — it means employees who work overtime tend to leave more often.\n"
    "   Interventions should be designed with HR domain expertise, not derived mechanically from\n"
    "   model outputs.\n\n"
    "6. **O*NET taxonomy artifact in skill gap rollups:** The top organization-wide skill gaps\n"
    "   (Speaking, Reading Comprehension, Active Listening, Critical Thinking) are partly an artifact\n"
    "   of O*NET's essential-skills taxonomy, where these general skills appear in nearly every\n"
    "   occupation's requirement list -- their high missing-counts reflect breadth of appearance across\n"
    "   roles combined with random synthetic assignment, not necessarily the most business-critical\n"
    "   real-world gap."
)

if target in text:
    updated = text.replace(target, addition, 1)
    with open(card_path, 'w', encoding='utf-8') as f:
        f.write(updated)
    print("SUCCESS: Appended item 6 to docs/model_card.md under Known Limitations.")
else:
    # Try normalizing dashes/quotes
    print("WARNING: exact target not found. Checking chunk:")
    idx = text.find("5. **No causal inference")
    print(repr(text[idx:idx+350]))
