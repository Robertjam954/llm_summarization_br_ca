"""
Run End-to-End Evaluation Using EHR Deidentification Test Dataset
Demonstrates the complete workflow with the hierarchical multi-agent system
"""

import os
import sys
import asyncio
from pathlib import Path
import logging

from ehr_dataset_adapter import EHRDatasetAdapter, create_test_cases_from_ehr_data
from hierarchical_agents import HierarchicalAgentSystem
from evaluation_metrics import MetricEvaluator
from main import ClinicalSummarizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_deidentification_evaluation():
    """Run deidentification evaluation on EHR test dataset"""
    logger.info("=" * 70)
    logger.info("DEIDENTIFICATION EVALUATION - EHR TEST DATASET")
    logger.info("=" * 70)
    
    # Initialize adapter
    adapter = EHRDatasetAdapter()
    
    # Load and build dataset
    logger.info("\nLoading EHR test data...")
    samples = adapter.load_samples()
    logger.info(f"Loaded {len(samples)} samples")
    
    # Build deidentification dataset
    logger.info("\nBuilding deidentification dataset...")
    dataset = adapter.build_evaluation_dataset(task="deidentification")
    adapter.save_dataset(dataset, "ehr_deidentification")
    
    # Initialize agent system
    logger.info("\nInitializing hierarchical agent system...")
    agent_system = HierarchicalAgentSystem()
    
    # Create test cases by running agent on goldens
    logger.info("\nRunning deidentification agent on test cases...")
    
    def deidentify_text(input_text: str) -> str:
        """Wrapper function for deidentification"""
        result = agent_system.invoke(input_text)
        return result["messages"][-1].content
    
    test_cases = create_test_cases_from_ehr_data(
        llm_app_fn=deidentify_text,
        task="deidentification"
    )
    
    logger.info(f"Created {len(test_cases)} test cases")
    
    # Evaluate with metrics
    logger.info("\nEvaluating with metrics...")
    evaluator = MetricEvaluator()
    
    results_summary = {
        "total_cases": len(test_cases),
        "passed_cases": 0,
        "failed_cases": 0,
        "metric_scores": []
    }
    
    for idx, test_case in enumerate(test_cases[:5]):  # Limit to 5 for demo
        logger.info(f"\nEvaluating test case {idx + 1}/{min(5, len(test_cases))}...")
        
        # Run async evaluation
        results = asyncio.run(
            evaluator.evaluate_async(
                test_case=test_case,
                metric_categories=["text_based", "safety"]
            )
        )
        
        # Check if passed
        passed = all(r.passed for r in results)
        if passed:
            results_summary["passed_cases"] += 1
        else:
            results_summary["failed_cases"] += 1
        
        # Log results
        for result in results:
            logger.info(
                f"  {result.metric_name}: {result.score:.3f} "
                f"({'PASS' if result.passed else 'FAIL'})"
            )
            results_summary["metric_scores"].append({
                "metric": result.metric_name,
                "score": result.score,
                "passed": result.passed
            })
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total test cases: {results_summary['total_cases']}")
    logger.info(f"Evaluated: {min(5, len(test_cases))}")
    logger.info(f"Passed: {results_summary['passed_cases']}")
    logger.info(f"Failed: {results_summary['failed_cases']}")
    
    if results_summary['metric_scores']:
        avg_score = sum(
            m['score'] for m in results_summary['metric_scores']
        ) / len(results_summary['metric_scores'])
        logger.info(f"Average score: {avg_score:.3f}")
    
    return results_summary


def run_summarization_evaluation():
    """Run summarization evaluation on EHR test dataset"""
    logger.info("=" * 70)
    logger.info("SUMMARIZATION EVALUATION - EHR TEST DATASET")
    logger.info("=" * 70)
    
    # Initialize adapter
    adapter = EHRDatasetAdapter()
    
    # Build summarization dataset
    logger.info("\nBuilding summarization dataset...")
    dataset = adapter.build_evaluation_dataset(task="summarization")
    adapter.save_dataset(dataset, "ehr_summarization")
    
    # Initialize agent system
    logger.info("\nInitializing hierarchical agent system...")
    agent_system = HierarchicalAgentSystem()
    
    # Create test cases
    logger.info("\nRunning summarization agent on test cases...")
    
    def summarize_text(input_text: str) -> str:
        """Wrapper function for summarization"""
        result = agent_system.invoke(input_text)
        return result["messages"][-1].content
    
    test_cases = create_test_cases_from_ehr_data(
        llm_app_fn=summarize_text,
        task="summarization"
    )
    
    logger.info(f"Created {len(test_cases)} test cases")
    
    # Evaluate with metrics
    logger.info("\nEvaluating with metrics...")
    evaluator = MetricEvaluator()
    
    results_summary = {
        "total_cases": len(test_cases),
        "passed_cases": 0,
        "failed_cases": 0,
        "metric_scores": []
    }
    
    for idx, test_case in enumerate(test_cases[:3]):  # Limit to 3 for demo
        logger.info(f"\nEvaluating test case {idx + 1}/{min(3, len(test_cases))}...")
        
        # Run async evaluation
        results = asyncio.run(
            evaluator.evaluate_async(
                test_case=test_case,
                metric_categories=["summarization", "text_based"]
            )
        )
        
        # Check if passed
        passed = all(r.passed for r in results)
        if passed:
            results_summary["passed_cases"] += 1
        else:
            results_summary["failed_cases"] += 1
        
        # Log results
        for result in results:
            logger.info(
                f"  {result.metric_name}: {result.score:.3f} "
                f"({'PASS' if result.passed else 'FAIL'})"
            )
            results_summary["metric_scores"].append({
                "metric": result.metric_name,
                "score": result.score,
                "passed": result.passed
            })
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total test cases: {results_summary['total_cases']}")
    logger.info(f"Evaluated: {min(3, len(test_cases))}")
    logger.info(f"Passed: {results_summary['passed_cases']}")
    logger.info(f"Failed: {results_summary['failed_cases']}")
    
    if results_summary['metric_scores']:
        avg_score = sum(
            m['score'] for m in results_summary['metric_scores']
        ) / len(results_summary['metric_scores'])
        logger.info(f"Average score: {avg_score:.3f}")
    
    return results_summary


def explore_dataset():
    """Explore the EHR test dataset"""
    logger.info("=" * 70)
    logger.info("EHR TEST DATASET EXPLORATION")
    logger.info("=" * 70)
    
    adapter = EHRDatasetAdapter()
    samples = adapter.load_samples()
    
    logger.info(f"\nTotal samples: {len(samples)}")
    
    # Analyze PHI distribution
    phi_counts = [len(s.get_phi_entities()) for s in samples]
    logger.info(f"Total PHI entities: {sum(phi_counts)}")
    logger.info(f"Average PHI per sample: {sum(phi_counts)/len(samples):.2f}")
    logger.info(f"Max PHI in single sample: {max(phi_counts)}")
    logger.info(f"Min PHI in single sample: {min(phi_counts)}")
    
    # Analyze note distribution
    note_ids = set(s.get_note_id() for s in samples)
    logger.info(f"\nUnique notes: {len(note_ids)}")
    
    # Show sample
    logger.info("\n" + "=" * 70)
    logger.info("SAMPLE EXAMPLE")
    logger.info("=" * 70)
    
    sample_with_phi = next((s for s in samples if len(s.get_phi_entities()) > 0), None)
    if sample_with_phi:
        logger.info(f"\nText: {sample_with_phi.get_text()[:200]}...")
        logger.info(f"\nPHI entities found:")
        for entity in sample_with_phi.get_phi_entities()[:5]:
            logger.info(f"  - {entity['label']}: {entity['text']}")
    
    # Export summary
    adapter.export_summary_csv("ehr_test_dataset")
    
    return {
        "total_samples": len(samples),
        "total_phi": sum(phi_counts),
        "unique_notes": len(note_ids)
    }


def main():
    """Main entry point"""
    
    # Check environment
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set. Please set it before running.")
        sys.exit(1)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run EHR dataset evaluation")
    parser.add_argument(
        "--task",
        choices=["explore", "deidentify", "summarize", "all"],
        default="explore",
        help="Task to run"
    )
    args = parser.parse_args()
    
    if args.task == "explore":
        explore_dataset()
    
    elif args.task == "deidentify":
        run_deidentification_evaluation()
    
    elif args.task == "summarize":
        run_summarization_evaluation()
    
    elif args.task == "all":
        explore_dataset()
        logger.info("\n\n")
        run_deidentification_evaluation()
        logger.info("\n\n")
        run_summarization_evaluation()
    
    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
