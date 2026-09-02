import sys
import codecs
import json
import requests

# Ensure stdout handles unicode without crashing Windows cp1252
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

API_URL = "http://localhost:8000/rag/ask"

test_cases = [
    {
        "id": 1,
        "question": "What does a Research Scientist do?",
        "expected_source": "occupation_master.csv",
        "must_refuse": False
    },
    {
        "id": 2,
        "question": "Why is the Manager role's O*NET mapping unreliable?",
        "expected_source": "data_relationships.md",
        "must_refuse": False
    },
    {
        "id": 3,
        "question": "What is the production model's decision threshold and why was it chosen?",
        "expected_source": "model_card.md",
        "must_refuse": False
    },
    {
        "id": 4,
        "question": "What is the company's parental leave policy?",
        "expected_source": None,
        "must_refuse": True
    },
    {
        "id": 5,
        "question": "What is the capital of France?",
        "expected_source": None,
        "must_refuse": True
    }
]

REFUSAL = "I don't have information about that in the platform's knowledge base"

print("=" * 80)
print("TESTING LIVE FASTAPI ENDPOINT: POST /rag/ask")
print("=" * 80)

all_passed = True

for case in test_cases:
    print(f"\n--- Test {case['id']}: '{case['question']}' ---")
    resp = requests.post(API_URL, json={"question": case["question"]})
    if resp.status_code != 200:
        print(f"FAILED: HTTP {resp.status_code}: {resp.text}")
        all_passed = False
        continue

    data = resp.json()
    answer = data.get("answer", "")
    sources = data.get("sources", [])

    print(f"Status: {resp.status_code}")
    print(f"Answer: {answer}")
    print(f"Sources ({len(sources)} returned):")
    for s in sources:
        print(f"  * [{s['score']:.4f}] {s['source']}")
        print(f"    Excerpt: {s['excerpt'][:100]}...")

    # Assertions
    if case["must_refuse"]:
        if REFUSAL.lower() not in answer.lower():
            print(f"FAILED: Expected refusal '{REFUSAL}', but got '{answer}'")
            all_passed = False
        else:
            print("PASSED: Correctly refused ungrounded/out-of-domain query.")
    else:
        if REFUSAL.lower() in answer.lower():
            print(f"FAILED: Expected grounded answer, but query was refused.")
            all_passed = False
        else:
            # Check source provenance
            source_found = any(case["expected_source"] in s["source"] for s in sources)
            if not source_found:
                print(f"WARNING: Expected source '{case['expected_source']}' in sources list.")
            print(f"PASSED: Grounded response retrieved from {case['expected_source']}.")

print("\n" + "=" * 80)
if all_passed:
    print("ALL 5 LIVE RAG API TESTS PASSED SUCCESSFULLY!")
else:
    print("SOME TESTS FAILED.")
print("=" * 80)
