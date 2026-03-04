"""Test script for document similarity analysis without Voyage AI (using mock embeddings)."""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

# Load the v3 data with preserved text
FEATURES_DIR = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\moo\llm_summary\data\processed\source_doc_features_v3")

# Load cases
with open(FEATURES_DIR / "all_cases_features.json") as f:
    cases = json.load(f)

# Filter cases with source documents
cases_with_docs = [c for c in cases if c.get("source_document_features")]
print(f"Loaded {len(cases_with_docs)} cases with source documents")

# Check text preservation
text_preserved = 0
for case in cases_with_docs:
    if case["source_document_features"].get("combined_source_text"):
        text_preserved += 1

print(f"Cases with preserved text: {text_preserved}/{len(cases_with_docs)}")

# Show example text content
if cases_with_docs:
    example = cases_with_docs[0]
    text = example["source_document_features"].get("combined_source_text", "")
    print(f"\nExample case {example['case_index']} ({example['source_document_features']['case_folder']}):")
    print(f"Text length: {len(text)} characters")
    print(f"Word count: {example['source_document_features']['combined_source_text_features']['word_count']}")
    if text:
        print(f"First 300 chars:\n{text[:300]}...")

# Create mock embeddings based on text length for testing
print("\nCreating mock embeddings for similarity analysis...")
mock_embeddings = []
for case in cases_with_docs:
    word_count = case["source_document_features"]["combined_source_text_features"]["word_count"]
    # Create a simple embedding based on word count and some randomness
    embedding = np.random.randn(1024)
    embedding[0] = word_count / 1000  # Use first dimension to encode word count
    mock_embeddings.append(embedding)

mock_embeddings = np.array(mock_embeddings)

# Calculate similarity matrix
similarity_matrix = cosine_similarity(mock_embeddings)

# Find most similar pairs
print("\nMost similar case pairs:")
similar_pairs = []
for i in range(len(cases_with_docs)):
    for j in range(i + 1, len(cases_with_docs)):
        sim = similarity_matrix[i, j]
        if sim > 0.8:  # High similarity threshold
            similar_pairs.append((i, j, sim))

similar_pairs.sort(key=lambda x: x[2], reverse=True)

for i, j, sim in similar_pairs[:5]:
    case1 = cases_with_docs[i]
    case2 = cases_with_docs[j]
    acc1 = case1["case_level_summary"]["case_accuracy_ai"]
    acc2 = case2["case_level_summary"]["case_accuracy_ai"]
    print(f"  Cases {i} ({case1['source_document_features']['case_folder']}) and {j} ({case2['source_document_features']['case_folder']}):")
    print(f"    Similarity: {sim:.3f}")
    print(f"    AI accuracies: {acc1:.3f} vs {acc2:.3f}")

# Check for cases with different outcomes but similar features
print("\nCases with different AI outcomes but similar word counts:")
# Group by word count ranges
word_count_groups = {}
for case in cases_with_docs:
    wc = case["source_document_features"]["combined_source_text_features"]["word_count"]
    bucket = (wc // 100) * 100  # Group by 100-word buckets
    if bucket not in word_count_groups:
        word_count_groups[bucket] = []
    word_count_groups[bucket].append(case)

for bucket, group_cases in word_count_groups.items():
    if len(group_cases) > 1:
        accuracies = [c["case_level_summary"]["case_accuracy_ai"] for c in group_cases]
        if max(accuracies) - min(accuracies) > 0.2:  # Significant difference
            print(f"  Word count {bucket}-{bucket+99}: {len(group_cases)} cases")
            for c in group_cases:
                print(f"    {c['source_document_features']['case_folder']}: accuracy={c['case_level_summary']['case_accuracy_ai']:.3f}")
