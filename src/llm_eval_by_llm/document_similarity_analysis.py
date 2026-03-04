"""
Document Similarity Analysis & Error Pattern Detection
======================================================
Uses embeddings to analyze similar cases with different annotation outcomes
and detect patterns in AI mistakes.

Features:
1. Embeds all source documents using Voyage AI
2. Finds similar cases with different AI annotation outcomes
3. Clusters cases by error patterns
4. Visualizes similarity networks

Usage:
    python document_similarity_analysis.py [--model voyage-3.5]
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Voyage AI for embeddings
try:
    import voyageai
    VOYAGE_AVAILABLE = True
except ImportError:
    VOYAGE_AVAILABLE = False
    print("Voyage AI not installed. Install with: pip install voyageai")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "source_doc_features_v3"  # Use v3 with text preservation
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "similarity_analysis"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Element definitions
# ---------------------------------------------------------------------------
ELEMENTS = [
    "Lesion Size", "Lesion Laterality", "Lesion Location",
    "Calcifications / Asymmetry", "Additional Enhancement (MRI)",
    "Extent", "Accurate Clip Placement", "Workup Recommendation",
    "Lymph Node", "Chronology Preserved", "Biopsy Method",
    "Invasive Component Size (Pathology)", "Histologic Diagnosis",
    "Receptor Status",
]


# ===================================================================
# Initialize Voyage AI client
# ===================================================================
def init_voyage_client():
    """Initialize Voyage AI client with API key from environment."""
    if not VOYAGE_AVAILABLE:
        return None
    
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        log.warning("VOYAGE_API_KEY not found in environment variables")
        return None
    
    return voyageai.Client(api_key=api_key)


# ===================================================================
# Load and prepare data
# ===================================================================
def load_case_data(features_dir: Path) -> List[Dict]:
    """Load case features from JSON files."""
    cases = []
    all_cases_path = features_dir / "all_cases_features.json"
    
    if all_cases_path.exists():
        with open(all_cases_path) as f:
            cases = json.load(f)
    else:
        # Load individual case files
        for json_file in sorted(features_dir.glob("case_*.json")):
            with open(json_file) as f:
                cases.append(json.load(f))
    
    # Filter cases with source documents
    cases_with_docs = [c for c in cases if c.get("source_document_features")]
    log.info(f"Loaded {len(cases)} total cases, {len(cases_with_docs)} with source documents")
    
    return cases_with_docs


# ===================================================================
# Extract text for embedding
# ===================================================================
def extract_text_for_embedding(case: Dict) -> str:
    """Extract combined text from source documents for embedding."""
    if not case.get("source_document_features"):
        return ""
    
    # Use the preserved combined text
    combined_text = case["source_document_features"].get("combined_source_text", "")
    
    # If no combined text, try to reconstruct from file contents
    if not combined_text:
        texts = []
        for file_info in case["source_document_features"].get("files", []):
            if file_info.get("report_type") not in ["hpi_ai", "hpi_human"]:
                file_text = file_info.get("text_content", "")
                if file_text:
                    texts.append(f"--- {file_info['filename']} ---\n{file_text}")
        combined_text = "\n\n".join(texts)
    
    # Truncate if too long (Voyage AI has limits)
    if len(combined_text) > 8000:
        combined_text = combined_text[:8000] + "..."
    
    return combined_text


# ===================================================================
# Embed documents using Voyage AI
# ===================================================================
def embed_documents(cases: List[Dict], client, model: str = "voyage-3.5", batch_size: int = 128) -> np.ndarray:
    """Embed all case documents using Voyage AI."""
    log.info(f"Embedding {len(cases)} documents with model {model}...")
    
    # Extract text for each case
    documents = [extract_text_for_embedding(case) for case in cases]
    
    # Filter out empty documents
    valid_indices = [i for i, doc in enumerate(documents) if doc.strip()]
    valid_documents = [documents[i] for i in valid_indices]
    
    if not valid_documents:
        log.warning("No valid documents to embed")
        return np.array([])
    
    log.info(f"Embedding {len(valid_documents)} valid documents...")
    
    # Embed in batches
    all_embeddings = []
    for i in range(0, len(valid_documents), batch_size):
        batch = valid_documents[i:i+batch_size]
        try:
            result = client.embed(batch, model=model, input_type="document")
            all_embeddings.extend(result.embeddings)
            log.info(f"Embedded batch {i//batch_size + 1}/{(len(valid_documents)-1)//batch_size + 1}")
        except Exception as e:
            log.error(f"Failed to embed batch {i//batch_size + 1}: {e}")
            # Add zero embeddings for failed batch
            all_embeddings.extend([[0.0] * 1024] * len(batch))
    
    # Create full array with zeros for invalid documents
    full_embeddings = np.zeros((len(cases), len(all_embeddings[0]) if all_embeddings else 1024))
    for i, idx in enumerate(valid_indices):
        full_embeddings[idx] = all_embeddings[i]
    
    return full_embeddings


# ===================================================================
# Find similar cases with different outcomes
# ===================================================================
def find_similar_cases_different_outcomes(
    cases: List[Dict], 
    embeddings: np.ndarray, 
    element: str,
    similarity_threshold: float = 0.8
) -> List[Dict]:
    """
    Find pairs of cases that are similar but have different AI annotation outcomes
    for a specific element.
    """
    # Get outcomes for the element
    outcomes = []
    for case in cases:
        elem_data = case["elements"].get(element, {})
        outcomes.append(elem_data.get("ai_correct"))
    
    # Find pairs with different outcomes
    similar_pairs = []
    similarity_matrix = cosine_similarity(embeddings)
    
    for i in range(len(cases)):
        for j in range(i + 1, len(cases)):
            # Skip if outcomes are the same or missing
            if outcomes[i] is None or outcomes[j] is None:
                continue
            if outcomes[i] == outcomes[j]:
                continue
            
            # Check similarity
            sim = similarity_matrix[i, j]
            if sim >= similarity_threshold:
                similar_pairs.append({
                    "case_1_index": i,
                    "case_2_index": j,
                    "similarity": float(sim),
                    "case_1_correct": outcomes[i],
                    "case_2_correct": outcomes[j],
                    "case_1_folder": cases[i]["source_document_features"]["case_folder"],
                    "case_2_folder": cases[j]["source_document_features"]["case_folder"],
                    "element": element,
                })
    
    return similar_pairs


# ===================================================================
# Cluster error patterns
# ===================================================================
def cluster_error_patterns(cases: List[Dict], embeddings: np.ndarray) -> Dict:
    """Cluster cases based on their error patterns using DBSCAN."""
    # Create error pattern vector for each case
    error_patterns = []
    for case in cases:
        pattern = []
        for element in ELEMENTS:
            elem_data = case["elements"].get(element, {})
            if elem_data.get("ai_correct") is False:
                pattern.append(1)  # Error
            elif elem_data.get("ai_correct") is True:
                pattern.append(0)  # Correct
            else:
                pattern.append(-1)  # Not evaluated
        error_patterns.append(pattern)
    
    error_patterns = np.array(error_patterns)
    
    # Cluster using DBSCAN
    clustering = DBSCAN(eps=0.3, min_samples=2, metric="cosine")
    clusters = clustering.fit_predict(error_patterns)
    
    # Analyze clusters
    cluster_analysis = {}
    for cluster_id in set(clusters):
        if cluster_id == -1:  # Noise points
            continue
        
        cluster_cases = [cases[i] for i in range(len(cases)) if clusters[i] == cluster_id]
        
        # Find common errors in this cluster
        error_counts = Counter()
        for case in cluster_cases:
            for element in ELEMENTS:
                elem_data = case["elements"].get(element, {})
                if elem_data.get("ai_correct") is False:
                    error_counts[element] += 1
        
        cluster_analysis[cluster_id] = {
            "size": len(cluster_cases),
            "common_errors": error_counts.most_common(5),
            "case_indices": [i for i in range(len(cases)) if clusters[i] == cluster_id],
        }
    
    return {
        "clusters": cluster_analysis,
        "cluster_labels": clusters.tolist(),
        "n_clusters": len(set(clusters)) - (1 if -1 in clusters else 0),
    }


# ===================================================================
# Visualize similarities
# ===================================================================
def visualize_similarity_network(
    cases: List[Dict], 
    embeddings: np.ndarray, 
    output_dir: Path,
    element: str = None
):
    """Create t-SNE visualization of document similarities."""
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(cases)-1))
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Prepare data for plotting
    plot_data = []
    for i, case in enumerate(cases):
        # Color by overall accuracy
        accuracy = case["case_level_summary"].get("case_accuracy_ai", 0)
        
        # If focusing on specific element, use that
        if element:
            elem_data = case["elements"].get(element, {})
            accuracy = 1.0 if elem_data.get("ai_correct") is True else 0.0 if elem_data.get("ai_correct") is False else None
        
        plot_data.append({
            "x": embeddings_2d[i, 0],
            "y": embeddings_2d[i, 1],
            "case_index": i,
            "case_folder": case["source_document_features"]["case_folder"],
            "accuracy": accuracy,
            "word_count": case["source_document_features"]["combined_source_text_features"]["word_count"],
        })
    
    df = pd.DataFrame(plot_data)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        df["x"], df["y"], 
        c=df["accuracy"], 
        cmap="RdYlGn", 
        s=50 + df["word_count"]/20,  # Size by word count
        alpha=0.7,
        vmin=0, vmax=1
    )
    
    plt.colorbar(scatter, label="AI Accuracy")
    plt.title(f"Document Similarity Map{' (Element: ' + element + ')' if element else ''}")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    
    # Add annotations for outliers
    outliers = df[(df["accuracy"] < 0.5) | (df["accuracy"] > 0.95)]
    for _, row in outliers.iterrows():
        plt.annotate(
            f"{row['case_folder']}",
            (row["x"], row["y"]),
            fontsize=8,
            alpha=0.7
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / f"similarity_map{'_' + element if element else ''}.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    log.info(f"Saved similarity map to {output_dir}")


# ===================================================================
# Generate analysis report
# ===================================================================
def generate_analysis_report(
    cases: List[Dict],
    similar_pairs: List[Dict],
    cluster_analysis: Dict,
    output_dir: Path
):
    """Generate a comprehensive analysis report."""
    report = {
        "summary": {
            "total_cases": len(cases),
            "similar_case_pairs_found": len(similar_pairs),
            "error_clusters_found": cluster_analysis.get("n_clusters", 0),
        },
        "similar_cases_by_element": {},
        "error_clusters": cluster_analysis.get("clusters", {}),
        "recommendations": [],
    }
    
    # Group similar pairs by element
    for element in ELEMENTS:
        element_pairs = [p for p in similar_pairs if p["element"] == element]
        if element_pairs:
            report["similar_cases_by_element"][element] = {
                "n_pairs": len(element_pairs),
                "avg_similarity": np.mean([p["similarity"] for p in element_pairs]),
                "examples": element_pairs[:3],  # Top 3 examples
            }
    
    # Generate recommendations
    if report["summary"]["similar_case_pairs_found"] > 0:
        report["recommendations"].append(
            "Found similar cases with different outcomes - investigate prompt sensitivity"
        )
    
    if cluster_analysis.get("n_clusters", 0) > 0:
        report["recommendations"].append(
            "Identified distinct error patterns - consider element-specific prompt optimization"
        )
    
    # Save report
    with open(output_dir / "similarity_analysis_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    log.info(f"Saved analysis report to {output_dir / 'similarity_analysis_report.json'}")
    
    # Print summary
    print("\n=== SIMILARITY ANALYSIS SUMMARY ===")
    print(f"Total cases analyzed: {report['summary']['total_cases']}")
    print(f"Similar case pairs found: {report['summary']['similar_case_pairs_found']}")
    print(f"Error clusters identified: {report['summary']['error_clusters_found']}")
    
    print("\nTop elements with similar cases (different outcomes):")
    for element, data in sorted(
        report["similar_cases_by_element"].items(), 
        key=lambda x: x[1]["n_pairs"], 
        reverse=True
    )[:5]:
        print(f"  {element}: {data['n_pairs']} pairs (avg sim: {data['avg_similarity']:.3f})")
    
    print("\nRecommendations:")
    for rec in report["recommendations"]:
        print(f"  • {rec}")


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Document Similarity Analysis")
    parser.add_argument("--model", type=str, default="voyage-3.5", 
                        help="Voyage AI model to use")
    parser.add_argument("--similarity-threshold", type=float, default=0.8,
                        help="Similarity threshold for finding similar cases")
    parser.add_argument("--skip-embedding", action="store_true",
                        help="Skip embedding generation (use cached embeddings)")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize Voyage AI
    client = init_voyage_client()
    if not client:
        log.error("Failed to initialize Voyage AI client")
        return
    
    # Load case data
    cases = load_case_data(FEATURES_DIR)
    if not cases:
        log.error("No case data found")
        return
    
    # Generate or load embeddings
    embeddings_path = OUTPUT_DIR / f"embeddings_{args.model.replace('-', '_')}.npy"
    if args.skip_embedding and embeddings_path.exists():
        log.info(f"Loading cached embeddings from {embeddings_path}")
        embeddings = np.load(embeddings_path)
    else:
        embeddings = embed_documents(cases, client, model=args.model)
        np.save(embeddings_path, embeddings)
        log.info(f"Saved embeddings to {embeddings_path}")
    
    # Find similar cases with different outcomes for each element
    all_similar_pairs = []
    for element in ELEMENTS:
        pairs = find_similar_cases_different_outcomes(
            cases, embeddings, element, args.similarity_threshold
        )
        all_similar_pairs.extend(pairs)
        log.info(f"Found {len(pairs)} similar case pairs for {element}")
    
    # Save similar pairs
    with open(OUTPUT_DIR / "similar_case_pairs.json", "w") as f:
        json.dump(all_similar_pairs, f, indent=2, default=str)
    
    # Cluster error patterns
    cluster_analysis = cluster_error_patterns(cases, embeddings)
    with open(OUTPUT_DIR / "error_clusters.json", "w") as f:
        json.dump(cluster_analysis, f, indent=2, default=str)
    
    # Generate visualizations
    log.info("Generating visualizations...")
    visualize_similarity_network(cases, embeddings, OUTPUT_DIR)
    
    # Generate per-element visualizations for top problematic elements
    element_error_rates = {}
    for element in ELEMENTS:
        errors = sum(1 for case in cases 
                    if case["elements"].get(element, {}).get("ai_correct") is False)
        total = sum(1 for case in cases 
                   if case["elements"].get(element, {}).get("ai_correct") is not None)
        if total > 0:
            element_error_rates[element] = errors / total
    
    top_elements = sorted(element_error_rates.items(), key=lambda x: x[1], reverse=True)[:3]
    for element, _ in top_elements:
        visualize_similarity_network(cases, embeddings, OUTPUT_DIR, element)
    
    # Generate report
    generate_analysis_report(cases, all_similar_pairs, cluster_analysis, OUTPUT_DIR)
    
    log.info("Analysis complete. Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
