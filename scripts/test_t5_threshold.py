from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base", local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", local_files_only=True)

context = """## Performance Metrics (Test Set, Threshold = 0.40)
Decision Threshold: 0.40 (stored in models/model_config.json for use by the API layer; not hardcoded in application code).
Recall: 0.7872 - Caught 37 of 47 true leavers.
Precision: 0.3426 - 37 true positives out of 108 flagged employees.
F1 Score: 0.4774
ROC-AUC: 0.8060
This is an expected trade-off given the business priority of catching leavers (Recall primary). The decision threshold of 0.40 was chosen to prioritize recall on at-risk employees."""

q = "What is the production model's decision threshold and why was it chosen?"
prompt = f"""Read the context and answer the question accurately using only information from the context.

Context:
{context}

Question: {q}
Answer:"""

inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print("Answer:", tokenizer.decode(outputs[0], skip_special_tokens=True))
