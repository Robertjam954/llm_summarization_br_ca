"""
Main Orchestration Script for Hierarchical Multi-Agent System
Production-ready with tracing, evaluation, and monitoring
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

from deepeval import evaluate
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase

from config import get_config, METRIC_CATEGORIES
from hierarchical_agents import HierarchicalAgentSystem
from evaluation_metrics import MetricEvaluator, MetricResult
from dataset_builder import DatasetBuilder

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

config = get_config()


class ClinicalSummarizationPipeline:
    """End-to-end clinical summarization pipeline with evaluation"""
    
    def __init__(self):
        self.config = config
        self.agent_system = HierarchicalAgentSystem()
        self.metric_evaluator = MetricEvaluator()
        self.dataset_builder = DatasetBuilder()
        
        logger.info("Clinical Summarization Pipeline initialized")
        logger.info(f"Model: {self.config.model.primary_model}")
        logger.info(f"Tracing enabled: {self.config.tracing.langsmith_enabled}")
    
    @observe(type="pipeline")
    def process_patient(self, patient_id: str, source_documents: List[str]) -> Dict[str, Any]:
        """Process a single patient through the pipeline"""
        logger.info(f"Processing patient: {patient_id}")
        
        query = f"Extract and validate clinical features for patient {patient_id}"
        
        result = self.agent_system.invoke(
            query=query,
            metadata={
                "patient_id": patient_id,
                "source_documents": source_documents
            }
        )
        
        update_current_span(
            input=query,
            output=result["messages"][-1].content,
            metadata={"patient_id": patient_id}
        )
        
        logger.info(f"Patient {patient_id} processed successfully")
        return result
    
    async def evaluate_patient_async(
        self,
        test_case: LLMTestCase,
        metric_categories: List[str]
    ) -> List[MetricResult]:
        """Evaluate patient summary asynchronously"""
        logger.info(f"Evaluating test case with metrics: {metric_categories}")
        
        results = await self.metric_evaluator.evaluate_async(
            test_case=test_case,
            metric_categories=metric_categories
        )
        
        passed = sum(1 for r in results if r.passed)
        total = len(results)
        logger.info(f"Evaluation complete: {passed}/{total} metrics passed")
        
        return results
    
    def evaluate_batch(
        self,
        test_cases: List[LLMTestCase],
        metric_categories: List[str]
    ) -> Dict[str, Any]:
        """Evaluate batch of test cases"""
        logger.info(f"Evaluating batch of {len(test_cases)} test cases")
        
        all_results = []
        for test_case in test_cases:
            results = asyncio.run(
                self.evaluate_patient_async(test_case, metric_categories)
            )
            all_results.append({
                "test_case": test_case,
                "results": results
            })
        
        summary = {
            "total_cases": len(test_cases),
            "total_metrics": len(all_results[0]["results"]) if all_results else 0,
            "passed_cases": sum(
                1 for r in all_results
                if all(metric.passed for metric in r["results"])
            ),
            "results": all_results
        }
        
        logger.info(f"Batch evaluation complete: {summary['passed_cases']}/{summary['total_cases']} cases passed")
        return summary
    
    def run_end_to_end_evaluation(self, dataset_name: str = "clinical_summarization"):
        """Run end-to-end evaluation on dataset"""
        logger.info(f"Starting end-to-end evaluation on dataset: {dataset_name}")
        
        try:
            dataset = self.dataset_builder.load_dataset(dataset_name)
        except FileNotFoundError:
            logger.info("Dataset not found, building new dataset...")
            dataset = self.dataset_builder.build_dataset(dataset_name)
            self.dataset_builder.save_dataset(dataset_name)
        
        test_cases = self.dataset_builder.create_test_cases(
            dataset_name=dataset_name,
            llm_app_fn=lambda query: self.agent_system.invoke(query)["messages"][-1].content
        )
        
        logger.info(f"Created {len(test_cases)} test cases")
        
        metric_categories = ["rag_retriever", "rag_generator", "summarization", "agentic"]
        
        results = self.evaluate_batch(test_cases, metric_categories)
        
        output_path = self.config.data.output_dir / f"{dataset_name}_evaluation_results.json"
        import json
        with open(output_path, 'w') as f:
            json.dump({
                "summary": {
                    "total_cases": results["total_cases"],
                    "passed_cases": results["passed_cases"],
                    "pass_rate": results["passed_cases"] / results["total_cases"]
                },
                "metric_categories": metric_categories
            }, f, indent=2)
        
        logger.info(f"Evaluation results saved to {output_path}")
        return results
    
    def monitor_production(self, patient_id: str, actual_output: str, expected_output: str):
        """Monitor production outputs with metrics"""
        test_case = LLMTestCase(
            input=f"Process patient {patient_id}",
            actual_output=actual_output,
            expected_output=expected_output
        )
        
        results = asyncio.run(
            self.evaluate_patient_async(
                test_case,
                metric_categories=["rag_generator", "summarization", "safety"]
            )
        )
        
        failed_metrics = [r for r in results if not r.passed]
        
        if failed_metrics:
            logger.warning(f"Patient {patient_id} failed {len(failed_metrics)} metrics:")
            for metric in failed_metrics:
                logger.warning(f"  - {metric.metric_name}: {metric.score:.3f} < {metric.threshold}")
        else:
            logger.info(f"Patient {patient_id} passed all quality checks")
        
        return results


@observe(type="application")
def main():
    """Main entry point"""
    logger.info("=" * 70)
    logger.info("HIERARCHICAL MULTI-AGENT CLINICAL SUMMARIZATION SYSTEM")
    logger.info("=" * 70)
    
    pipeline = ClinicalSummarizationPipeline()
    
    if os.getenv("RUN_EVALUATION", "false").lower() == "true":
        logger.info("\nRunning full evaluation...")
        results = pipeline.run_end_to_end_evaluation()
        logger.info(f"\nEvaluation complete: {results['passed_cases']}/{results['total_cases']} passed")
    
    else:
        logger.info("\nProcessing sample patient...")
        
        sample_patient_id = "BF_38138814"
        sample_docs = ["pathology_report.pdf", "imaging_report.pdf"]
        
        result = pipeline.process_patient(sample_patient_id, sample_docs)
        
        logger.info(f"\nResult: {result['messages'][-1].content[:300]}...")
        
        logger.info("\nMonitoring output quality...")
        pipeline.monitor_production(
            patient_id=sample_patient_id,
            actual_output=result['messages'][-1].content,
            expected_output="Expected clinical summary with all features"
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("Pipeline execution complete")
    logger.info("=" * 70)


if __name__ == "__main__":
    required_env_vars = ["OPENAI_API_KEY"]
    missing = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        logger.error("Please set them before running the pipeline")
        exit(1)
    
    main()
