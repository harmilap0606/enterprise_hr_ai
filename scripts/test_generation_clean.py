from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import re

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=True)

# Context 1: Research Scientist
c1 = """O*NET Title: Computer and Information Research Scientists (O*NET-SOC Code: 15-1221.00)
Description: Conduct research into fundamental computer and information science as theorists, designers, or inventors. Develop solutions to problems in the field of computer hardware and software."""

# Context 2: Manager O*NET mapping
c2 = """Document: docs/data_relationships.md (Section: Open Issues)
1. JobRole to O*NET Title gap -- IBM HR job roles do not match O*NET Titles exactly. Requires a manual/fuzzy mapping table before the role-intelligence step.
Document: docs/model_card.md (Section: Known Limitations)
JobRole to O*NET taxonomy gap: The JobRole categories in this dataset do not map directly to the O*NET occupational taxonomy used in the reference dataset. Until that mapping exists, O*NET-derived recommendations cannot be reliably attributed to specific job roles."""

# Context 3: Decision threshold
c3 = """Document: docs/model_card.md (Section: Performance Metrics)
Decision Threshold: 0.40 (stored in models/model_config.json for use by the API layer; not hardcoded in application code).
Recall: 0.7872 - Caught 37 of 47 true leavers (10 missed).
Precision: 0.3426 - 37 true positives out of 108 flagged employees.
Document: docs/model_card.md (Section: Known Limitations)
The 84/16 class imbalance means that even at high recall (0.7872), precision is inherently limited (0.3426). This is an expected trade-off given the business priority of catching leavers (Recall primary). The decision threshold of 0.40 was chosen to prioritize recall on at-risk employees."""

queries = [
    ("What does a Research Scientist do?", c1),
    ("Why is the Manager role's O*NET mapping unreliable?", c2),
    ("What is the production model's decision threshold and why was it chosen?", c3)
]

for q, c in queries:
    prompt = f"""Read the context below and answer the question in 1 or 2 concise sentences based only on the context.

Context:
{c}

Question: {q}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3
    )
    ans = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    print("=" * 60)
    print("Q:", q)
    print("A:", ans)
