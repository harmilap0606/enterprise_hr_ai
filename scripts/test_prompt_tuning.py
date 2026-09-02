from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=True)

# Test Q1
c1 = """O*NET Occupation: Computer and Information Research Scientists
O*NET-SOC Code: 15-1221.00
Description: Conduct research into fundamental computer and information science as theorists, designers, or inventors. Develop solutions to problems in the field of computer hardware and software."""

q1 = "What does a Research Scientist do?"

# Test Q2
c2 = """Document: docs/data_relationships.md
Section: Open Issues
1. JobRole to O*NET Title gap -- IBM HR job roles do not match O*NET Titles exactly. Requires a manual/fuzzy mapping table before the role-intelligence step.
Document: docs/model_card.md
Section: Known Limitations
JobRole to O*NET taxonomy gap: The JobRole categories in this dataset do not map directly to the O*NET occupational taxonomy used in the reference dataset. Until that mapping exists, O*NET-derived recommendations cannot be reliably attributed to specific job roles."""

q2 = "Why is the Manager role's O*NET mapping unreliable?"

# Test Q3
c3 = """Document: docs/model_card.md
Section: Performance Metrics (Test Set, Threshold = 0.40)
Decision Threshold: 0.40 (stored in models/model_config.json for use by the API layer; not hardcoded in application code).
Recall: 0.7872 - Caught 37 of 47 true leavers (10 missed).
Precision: 0.3426 - 37 true positives out of 108 flagged employees.
Section: Known Limitations
The 84/16 class imbalance means that even at high recall (0.7872), precision is inherently limited (0.3426). This is an expected trade-off given the business priority of catching leavers (Recall primary)."""

q3 = "What is the production model's decision threshold and why was it chosen?"

tests = [(q1, c1), (q2, c2), (q3, c3)]

for q, c in tests:
    prompt = f"""Context:
{c}

Question: {q}
Based on the context, provide a complete and accurate answer:"""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    print("=" * 60)
    print("Q:", q)
    print("A:", tokenizer.decode(outputs[0], skip_special_tokens=True))
