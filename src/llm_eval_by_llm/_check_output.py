"""Temporary script to inspect Phase 1 output."""
import json
from pathlib import Path

BASE = Path(r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\Research\Projects\moo\llm_summary\data\processed\source_doc_features")

# Load all cases
with open(BASE / "all_cases_features.json") as f:
    data = json.load(f)

matched = [r for r in data if r["source_document_features"] is not None]
unmatched = [r for r in data if r["source_document_features"] is None]

print(f"Total cases: {len(data)}")
print(f"Matched to folders: {len(matched)}")
print(f"Unmatched: {len(unmatched)}")

# Among matched, split by AI accuracy
perfect = [r for r in matched if r["case_level_summary"]["case_accuracy_ai"] == 1.0]
imperfect = [r for r in matched if r["case_level_summary"]["case_accuracy_ai"] is not None and r["case_level_summary"]["case_accuracy_ai"] < 1.0]

print(f"\nMatched - AI perfect: {len(perfect)}")
print(f"Matched - AI imperfect: {len(imperfect)}")

if perfect:
    avg_words_p = sum(r["source_document_features"]["combined_source_text_features"]["word_count"] for r in perfect) / len(perfect)
    avg_docs_p = sum(r["source_document_features"]["n_source_documents"] for r in perfect) / len(perfect)
    print(f"  Perfect: avg {avg_words_p:.0f} words, {avg_docs_p:.1f} source docs")

if imperfect:
    avg_words_i = sum(r["source_document_features"]["combined_source_text_features"]["word_count"] for r in imperfect) / len(imperfect)
    avg_docs_i = sum(r["source_document_features"]["n_source_documents"] for r in imperfect) / len(imperfect)
    print(f"  Imperfect: avg {avg_words_i:.0f} words, {avg_docs_i:.1f} source docs")

    print("\n  Imperfect cases detail:")
    for r in imperfect[:8]:
        acc = r["case_level_summary"]["case_accuracy_ai"]
        fab = r["case_level_summary"]["total_fabrication_ai"]
        miss = r["case_level_summary"]["total_correct_ai"]
        words = r["source_document_features"]["combined_source_text_features"]["word_count"]
        docs = r["source_document_features"]["n_source_documents"]
        folder = r["source_document_features"]["case_folder"]
        print(f"    Case {r['case_index']}: acc={acc:.3f}, fabrications={fab}, words={words}, docs={docs}, folder={folder}")

# Feature correctness analysis
print("\n=== FEATURE-CORRECTNESS ANALYSIS ===")
with open(BASE / "feature_correctness_analysis.json") as f:
    analysis = json.load(f)

el = analysis["element_level"]
print(f"\n{'Element':<45} {'Correct':>8} {'Incorrect':>10} {'Acc':>8}")
print("-" * 75)
for elem_name, d in el.items():
    cn = d["ai_correct"]["n"]
    icn = d["ai_incorrect"]["n"]
    total = cn + icn
    acc = cn / total if total > 0 else 0
    marker = " ***" if acc < 0.95 else ""
    print(f"{elem_name:<45} {cn:>8} {icn:>10} {acc:>8.3f}{marker}")
