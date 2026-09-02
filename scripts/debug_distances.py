from chromadb.utils import embedding_functions
import numpy as np

fn = embedding_functions.DefaultEmbeddingFunction()
q = "Why is the Manager role's O*NET mapping unreliable?"

c1 = "O*NET Occupation Title: Surveying and Mapping Technicians\nDescription: Perform surveying and mapping duties."
c2 = "Source: docs/data_relationships.md\nSection: Open Issues\nContent:\n## Open Issues\n1. JobRole to O*NET Title gap -- IBM HR job roles do not match O*NET Titles exactly. Requires a manual fuzzy mapping table before the role-intelligence step."

embs = fn([q, c1, c2])
d1 = 1 - np.dot(embs[0], embs[1]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[1]))
d2 = 1 - np.dot(embs[0], embs[2]) / (np.linalg.norm(embs[0]) * np.linalg.norm(embs[2]))
print(f"Distance c1 (Surveying): {d1:.4f}, Distance c2 (Open Issues): {d2:.4f}")
