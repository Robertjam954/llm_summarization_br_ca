"""HCAT Unified Evaluation Pipeline for Medical LLM Summarization

Integrates all HCAT framework components:
- Document Quality Features
- Embedding-Based Evaluation Metrics  
- Patient Safety Metrics
- Human-Machine Calibration

Provides a single interface for comprehensive evaluation.

Author: Generated for MSKCC Goel Lab project
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import HCAT components
from patient_safety_metrics import (
    PatientSafetyEvaluator, SafetyReport, create_safety_summary_df
)
from embedding_evaluation_metrics import (
    EmbeddingEvaluator, EmbeddingMetricsReport, create_metrics_summary_df
)
from document_quality_features import (
    DocumentQualityEvaluator, DocumentQualityReport, create_quality_summary_df
)
from human_machine_calibration import (
    HumanMachineCalibrator, MultiMetricCalibrator, CalibrationReport,
    compute_calibration_summary
)

logger = logging.getLogger(__name__)


@dataclass
class HCATEvaluationReport:
    """Complete HCAT evaluation results for a single case."""
    
    case_id: str
    
    # Component reports
    document_quality: DocumentQualityReport
    embedding_metrics: EmbeddingMetricsReport
    safety_metrics: SafetyReport
    
    # Calibration results (if available)
    calibration: Optional[CalibrationReport] = None
    
    # Aggregated scores
    overall_quality_score: float = 0.0
    overall_safety_score: float = 0.0
    overall_trust_score: float = 0.0
    
    # Decision flags
    requires_human_review: bool = False
    review_reasons: List[str] = None
    
    def __post_init__(self):
        if self.review_reasons is None:
            self.review_reasons = []
    
    def to_dict(self) -> Dict:
        return {
            'case_id': self.case_id,
            'doc_quality_score': self.document_quality.overall_quality_score,
            'doc_quality_tier': self.document_quality.quality_tier,
            'context_relevancy': self.embedding_metrics.context_relevancy,
            'groundedness': self.embedding_metrics.groundedness,
            'completeness': self.embedding_metrics.completeness,
            'answer_relevancy': self.embedding_metrics.answer_relevancy,
            'safety_risk_level': self.safety_metrics.risk_level,
            'pii_risk_score': self.safety_metrics.pii_risk_score,
            'toxicity_detected': self.safety_metrics.toxicity_detected,
            'overall_quality_score': self.overall_quality_score,
            'overall_safety_score': self.overall_safety_score,
            'overall_trust_score': self.overall_trust_score,
            'requires_human_review': self.requires_human_review,
            'review_reasons': ','.join(self.review_reasons)
        }


class HCATEvaluator:
    """Unified HCAT evaluation pipeline.
    
    Implements the complete HCAT framework:
    1. Document Quality Assessment (input quality)
    2. Embedding-Based Metrics (output quality - 4 dimensions)
    3. Patient Safety Metrics (safety & privacy)
    4. Human-Machine Calibration (uncertainty quantification)
    
    Example:
        evaluator = HCATEvaluator()
        
        report = evaluator.evaluate(
            case_id="CASE_001",
            query="What is the diagnosis?",
            answer="Patient has stage II breast cancer.",
            reference="Stage II invasive ductal carcinoma",
            source_text="Full clinical document text...",
            source_documents=["doc1", "doc2"]
        )
    """
    
    def __init__(
        self,
        embedding_model: str = "all-mpnet-base-v2",
        enable_safety: bool = True,
        enable_calibration: bool = False,
        calibration_data: Optional[Dict] = None
    ):
        """
        Args:
            embedding_model: Name of sentence-transformers model
            enable_safety: Enable PII/toxicity/adversarial checks
            enable_calibration: Enable human-machine calibration
            calibration_data: Dict with 'machine_scores' and 'human_labels' for calibration
        """
        self.doc_quality_evaluator = DocumentQualityEvaluator()
        self.embedding_evaluator = EmbeddingEvaluator(embedding_model)
        self.safety_evaluator = PatientSafetyEvaluator() if enable_safety else None
        
        # Initialize calibrator if data provided
        self.calibrator = None
        if enable_calibration and calibration_data:
            self.calibrator = HumanMachineCalibrator()
            self.calibrator.fit(
                calibration_data['machine_scores'],
                calibration_data['human_labels']
            )
    
    def evaluate(
        self,
        case_id: str,
        query: str,
        answer: str,
        reference: Optional[str] = None,
        source_text: Optional[str] = None,
        source_documents: Optional[List[str]] = None,
        retrieved_documents: Optional[List[str]] = None
    ) -> HCATEvaluationReport:
        """Run complete HCAT evaluation on a single case.
        
        Args:
            case_id: Unique identifier for the case
            query: The user's question/prompt
            answer: The LLM's generated summary/answer
            reference: Ground truth reference (optional, for completeness)
            source_text: Original source document text (optional, for quality)
            source_documents: List of documents provided to LLM
            retrieved_documents: List of documents retrieved by RAG
        """
        
        # 1. Document Quality Assessment
        if source_text:
            doc_quality = self.doc_quality_evaluator.evaluate(source_text)
        else:
            # Create default if no source text
            from document_quality_features import DocumentQualityReport
            doc_quality = DocumentQualityReport(
                image_quality_score=0.5, blur_score=0.5, contrast_score=0.5,
                skew_score=0.5, structure_score=0.5, section_completeness=0.5,
                temporal_consistency=0.5, medical_content_score=0.5,
                information_density=0.5, redundancy_score=0.5,
                negation_clarity=0.5, overall_quality_score=0.5,
                quality_tier='unknown', concerns=[]
            )
        
        # 2. Embedding-Based Metrics (HCAT 4 dimensions)
        embedding_metrics = self.embedding_evaluator.evaluate(
            query=query,
            answer=answer,
            reference=reference,
            retrieved_documents=retrieved_documents,
            source_documents=source_documents
        )
        
        # 3. Patient Safety Metrics
        if self.safety_evaluator:
            safety = self.safety_evaluator.evaluate(answer, context=source_text)
        else:
            from patient_safety_metrics import SafetyReport
            safety = SafetyReport(
                pii_detected=False, pii_entities=[], pii_risk_score=0.0,
                toxicity_detected=False, toxicity_scores={}, bias_flags=[],
                adversarial_indicators=[], contradiction_detected=False,
                is_safe=True, requires_human_review=False, risk_level='low'
            )
        
        # 4. Calibration (if enabled)
        calibration = None
        if self.calibrator:
            # Use average of embedding metrics as the score to calibrate
            avg_score = np.mean([
                embedding_metrics.context_relevancy,
                embedding_metrics.groundedness,
                embedding_metrics.completeness,
                embedding_metrics.answer_relevancy
            ])
            calibration = self.calibrator.calibrate(np.array([avg_score]))
        
        # Aggregate scores
        quality_score = doc_quality.overall_quality_score
        safety_score = 1.0 - safety.pii_risk_score  # Higher = safer
        
        # Trust score combines all dimensions
        trust_components = [
            embedding_metrics.context_relevancy * 0.2,
            embedding_metrics.groundedness * 0.3,
            embedding_metrics.completeness * 0.2,
            embedding_metrics.answer_relevancy * 0.2,
            safety_score * 0.1
        ]
        trust_score = sum(trust_components)
        
        # Determine if human review needed
        review_reasons = []
        
        if doc_quality.quality_tier in ['poor', 'fair']:
            review_reasons.append('poor_document_quality')
        if embedding_metrics.requires_human_review:
            review_reasons.append('low_embedding_confidence')
        if safety.requires_human_review:
            review_reasons.append('safety_concern')
        if calibration and np.any(calibration.uncertain_predictions):
            review_reasons.append('calibration_uncertainty')
        
        requires_review = len(review_reasons) > 0
        
        return HCATEvaluationReport(
            case_id=case_id,
            document_quality=doc_quality,
            embedding_metrics=embedding_metrics,
            safety_metrics=safety,
            calibration=calibration,
            overall_quality_score=quality_score,
            overall_safety_score=safety_score,
            overall_trust_score=trust_score,
            requires_human_review=requires_review,
            review_reasons=review_reasons
        )
    
    def evaluate_batch(
        self,
        cases: List[Dict[str, Any]],
        show_progress: bool = True
    ) -> List[HCATEvaluationReport]:
        """Evaluate multiple cases.
        
        Args:
            cases: List of dicts with keys matching evaluate() parameters
            show_progress: Show tqdm progress bar
        
        Returns:
            List of HCATEvaluationReport objects
        """
        reports = []
        
        iterator = tqdm(cases) if show_progress else cases
        
        for case in iterator:
            try:
                report = self.evaluate(**case)
                reports.append(report)
            except Exception as e:
                logger.error(f"Evaluation failed for case {case.get('case_id', 'unknown')}: {e}")
                # Create a failed report
                from document_quality_features import DocumentQualityReport
                from embedding_evaluation_metrics import EmbeddingMetricsReport
                from patient_safety_metrics import SafetyReport
                
                failed_report = HCATEvaluationReport(
                    case_id=case.get('case_id', 'ERROR'),
                    document_quality=DocumentQualityReport(
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 'error', ['evaluation_failed']
                    ),
                    embedding_metrics=EmbeddingMetricsReport(
                        0, 0, 0, 0, 0, 0, 0, True, ['evaluation_failed']
                    ),
                    safety_metrics=SafetyReport(
                        False, [], 0, False, {}, [], [], False, False, True, 'unknown'
                    ),
                    review_reasons=['evaluation_error']
                )
                reports.append(failed_report)
        
        return reports


class HCATResultsAggregator:
    """Aggregate and analyze HCAT evaluation results."""
    
    @staticmethod
    def to_dataframe(reports: List[HCATEvaluationReport]) -> pd.DataFrame:
        """Convert reports to a summary DataFrame."""
        return pd.DataFrame([r.to_dict() for r in reports])
    
    @staticmethod
    def compute_summary_stats(reports: List[HCATEvaluationReport]) -> Dict:
        """Compute aggregate statistics across all reports."""
        df = HCATResultsAggregator.to_dataframe(reports)
        
        return {
            'n_cases': len(reports),
            'mean_trust_score': df['overall_trust_score'].mean(),
            'mean_quality_score': df['overall_quality_score'].mean(),
            'mean_safety_score': df['overall_safety_score'].mean(),
            'pct_requiring_review': df['requires_human_review'].mean() * 100,
            'pct_poor_quality': (df['doc_quality_tier'] == 'poor').sum() / len(df) * 100,
            'pct_high_safety_risk': (df['safety_risk_level'].isin(['high', 'critical'])).sum() / len(df) * 100,
            'avg_context_relevancy': df['context_relevancy'].mean(),
            'avg_groundedness': df['groundedness'].mean(),
            'avg_completeness': df['completeness'].mean(),
            'avg_answer_relevancy': df['answer_relevancy'].mean()
        }
    
    @staticmethod
    def get_review_queue(
        reports: List[HCATEvaluationReport]
    ) -> List[Tuple[str, List[str]]]:
        """Get list of cases requiring human review with reasons."""
        queue = []
        for report in reports:
            if report.requires_human_review:
                queue.append((report.case_id, report.review_reasons))
        return queue
    
    @staticmethod
    def export_detailed_report(
        reports: List[HCATEvaluationReport],
        output_path: Path
    ):
        """Export detailed JSON report for all cases."""
        detailed = []
        for r in reports:
            entry = {
                'case_id': r.case_id,
                'document_quality': asdict(r.document_quality),
                'embedding_metrics': asdict(r.embedding_metrics),
                'safety_metrics': asdict(r.safety_metrics),
                'overall': r.to_dict()
            }
            detailed.append(entry)
        
        with open(output_path, 'w') as f:
            json.dump(detailed, f, indent=2, default=str)
        
        logger.info(f"Exported detailed report to {output_path}")


def create_hcat_evaluation_pipeline(
    embedding_model: str = "all-mpnet-base-v2",
    calibration_data_path: Optional[Path] = None
) -> HCATEvaluator:
    """Factory function to create a configured HCAT evaluator.
    
    Args:
        embedding_model: Sentence-transformers model name
        calibration_data_path: Path to CSV with 'machine_score' and 'human_label' columns
    
    Returns:
        Configured HCATEvaluator instance
    """
    calibration_data = None
    enable_calibration = False
    
    if calibration_data_path and calibration_data_path.exists():
        df = pd.read_csv(calibration_data_path)
        calibration_data = {
            'machine_scores': df['machine_score'].values,
            'human_labels': df['human_label'].values
        }
        enable_calibration = True
        logger.info(f"Loaded calibration data from {calibration_data_path}")
    
    return HCATEvaluator(
        embedding_model=embedding_model,
        enable_safety=True,
        enable_calibration=enable_calibration,
        calibration_data=calibration_data
    )


# Example usage
if __name__ == "__main__":
    # Create evaluator
    evaluator = HCATEvaluator()
    
    # Single evaluation
    report = evaluator.evaluate(
        case_id="TEST_001",
        query="Summarize the patient's cancer diagnosis",
        answer="The patient has stage II breast cancer, ER positive, PR positive, HER2 negative.",
        reference="Stage II invasive ductal carcinoma of the left breast. ER/PR positive, HER2 negative.",
        source_text="Full pathology report with diagnosis details..."
    )
    
    print(f"Case: {report.case_id}")
    print(f"Trust Score: {report.overall_trust_score:.3f}")
    print(f"Safety Risk: {report.safety_metrics.risk_level}")
    print(f"Requires Review: {report.requires_human_review}")
    if report.requires_human_review:
        print(f"Reasons: {report.review_reasons}")
