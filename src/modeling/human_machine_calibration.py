"""Human-Machine Calibration for Medical LLM Evaluation

Implements HCAT framework's Double-Calibration:
- Probability Calibration: Maps machine scores to human-aligned probabilities
- Conformal Prediction: Provides confidence intervals for uncertainty quantification

Author: Generated for MSKCC Goel Lab project
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class CalibrationReport:
    """Container for calibration results."""
    
    # Original and calibrated scores
    original_scores: np.ndarray
    calibrated_scores: np.ndarray
    
    # Calibration metrics
    expected_calibration_error: float
    maximum_calibration_error: float
    brier_score: float
    
    # Conformal prediction bounds
    confidence_intervals: Optional[np.ndarray]  # (n_samples, 2) for [lower, upper]
    coverage_rate: Optional[float]
    
    # Flags for human review
    uncertain_predictions: np.ndarray  # Boolean array
    
    def to_dict(self) -> Dict:
        return {
            'mean_original_score': float(np.mean(self.original_scores)),
            'mean_calibrated_score': float(np.mean(self.calibrated_scores)),
            'expected_calibration_error': self.expected_calibration_error,
            'maximum_calibration_error': self.maximum_calibration_error,
            'brier_score': self.brier_score,
            'uncertain_count': int(np.sum(self.uncertain_predictions)),
            'uncertain_fraction': float(np.mean(self.uncertain_predictions)),
            'coverage_rate': self.coverage_rate
        }


class ProbabilityCalibrator:
    """Calibrate machine scores to human-interpretable probabilities.
    
    Uses isotonic regression or Platt scaling to align model outputs
    with actual human judgment probabilities.
    """
    
    def __init__(self, method: str = 'isotonic'):
        """
        Args:
            method: 'isotonic' for Isotonic Regression, 'platt' for Platt scaling
        """
        self.method = method
        self.calibrator = None
        self.is_fitted = False
    
    def fit(
        self,
        scores: np.ndarray,
        human_labels: np.ndarray
    ) -> 'ProbabilityCalibrator':
        """Fit the calibrator using paired machine scores and human labels.
        
        Args:
            scores: Raw machine scores (0-1)
            human_labels: Binary human judgments (0 or 1)
        """
        scores = np.asarray(scores).reshape(-1, 1)
        human_labels = np.asarray(human_labels)
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(scores.flatten()) | np.isnan(human_labels))
        scores_clean = scores[valid_mask].flatten()
        labels_clean = human_labels[valid_mask]
        
        if len(scores_clean) < 10:
            logger.warning("Too few samples for calibration, using identity")
            self.calibrator = None
            self.is_fitted = True
            return self
        
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(scores_clean, labels_clean)
        elif self.method == 'platt':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(scores_clean.reshape(-1, 1), labels_clean)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
        
        self.is_fitted = True
        logger.info(f"Calibrator fitted with {len(scores_clean)} samples")
        return self
    
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Transform raw scores to calibrated probabilities."""
        if not self.is_fitted:
            logger.warning("Calibrator not fitted, returning original scores")
            return np.asarray(scores)
        
        scores = np.asarray(scores)
        
        if self.calibrator is None:
            # Identity calibration (pass-through)
            return np.clip(scores, 0, 1)
        
        if self.method == 'isotonic':
            return self.calibrator.transform(scores)
        else:  # platt
            return self.calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]
    
    def fit_transform(
        self,
        scores: np.ndarray,
        human_labels: np.ndarray
    ) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(scores, human_labels).transform(scores)


class ConformalPredictor:
    """Conformal prediction for uncertainty quantification.
    
    Provides prediction sets/confidence intervals with guaranteed coverage.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: Significance level (1 - coverage). Default 0.1 for 90% coverage.
        """
        self.alpha = alpha
        self.q_hat = None  # Quantile threshold
        self.is_fitted = False
    
    def fit(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> 'ConformalPredictor':
        """Fit conformal predictor on calibration set.
        
        Args:
            scores: Predicted scores/probabilities
            labels: True binary labels
        """
        scores = np.asarray(scores)
        labels = np.asarray(labels)
        
        # Compute non-conformity scores
        # For binary classification: |score - label|
        non_conformity = np.abs(scores - labels)
        
        # Compute quantile threshold
        n = len(non_conformity)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_hat = np.quantile(non_conformity, quantile_level, method='higher')
        
        self.is_fitted = True
        logger.info(f"Conformal predictor fitted: q_hat={self.q_hat:.4f}")
        return self
    
    def predict_interval(
        self,
        scores: np.ndarray
    ) -> np.ndarray:
        """Predict confidence intervals for new scores.
        
        Returns:
            Array of shape (n_samples, 2) with [lower, upper] bounds
        """
        if not self.is_fitted:
            raise RuntimeError("Conformal predictor not fitted")
        
        scores = np.asarray(scores)
        
        # Build prediction intervals
        lower = np.clip(scores - self.q_hat, 0, 1)
        upper = np.clip(scores + self.q_hat, 0, 1)
        
        return np.column_stack([lower, upper])
    
    def predict_uncertainty(
        self,
        scores: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """Identify uncertain predictions.
        
        Predictions are uncertain if the confidence interval crosses
        the decision threshold (default 0.5).
        
        Returns:
            Boolean array indicating uncertain predictions
        """
        if threshold is None:
            threshold = 0.5
        
        intervals = self.predict_interval(scores)
        
        # Uncertain if interval crosses threshold
        uncertain = (intervals[:, 0] < threshold) & (intervals[:, 1] > threshold)
        
        return uncertain


class HumanMachineCalibrator:
    """Main entry point for HCAT double-calibration.
    
    Combines probability calibration with conformal prediction
    to align machine scores with human judgment and flag uncertain cases.
    """
    
    def __init__(
        self,
        calibration_method: str = 'isotonic',
        conformal_alpha: float = 0.1,
        uncertainty_threshold: float = 0.5
    ):
        self.probability_calibrator = ProbabilityCalibrator(calibration_method)
        self.conformal_predictor = ConformalPredictor(conformal_alpha)
        self.uncertainty_threshold = uncertainty_threshold
    
    def fit(
        self,
        machine_scores: np.ndarray,
        human_labels: np.ndarray
    ) -> 'HumanMachineCalibrator':
        """Fit both calibrators on training data.
        
        Args:
            machine_scores: Raw machine confidence scores
            human_labels: Binary human judgments (ground truth)
        """
        machine_scores = np.asarray(machine_scores)
        human_labels = np.asarray(human_labels)
        
        # Split data for proper conformal calibration
        # Use 80% for probability calibration, 20% for conformal
        if len(machine_scores) >= 100:
            idx = np.random.RandomState(42).permutation(len(machine_scores))
            split = int(0.8 * len(machine_scores))
            
            prob_idx, conf_idx = idx[:split], idx[split:]
            
            # Fit probability calibrator
            self.probability_calibrator.fit(
                machine_scores[prob_idx],
                human_labels[prob_idx]
            )
            
            # Transform calibration set
            cal_scores = self.probability_calibrator.transform(machine_scores[conf_idx])
            
            # Fit conformal predictor on calibrated scores
            self.conformal_predictor.fit(cal_scores, human_labels[conf_idx])
        else:
            # Small dataset: use all data for both (approximation)
            self.probability_calibrator.fit(machine_scores, human_labels)
            cal_scores = self.probability_calibrator.transform(machine_scores)
            self.conformal_predictor.fit(cal_scores, human_labels)
        
        return self
    
    def calibrate(
        self,
        scores: np.ndarray,
        return_intervals: bool = True
    ) -> CalibrationReport:
        """Calibrate scores and generate report.
        
        Args:
            scores: Raw machine scores to calibrate
            return_intervals: Whether to compute confidence intervals
        
        Returns:
            CalibrationReport with calibrated scores and uncertainty flags
        """
        scores = np.asarray(scores)
        
        # Step 1: Probability calibration
        calibrated = self.probability_calibrator.transform(scores)
        
        # Step 2: Compute calibration metrics (using empirical approach)
        ece = self._compute_ece(scores, calibrated)
        mce = self._compute_mce(scores, calibrated)
        brier = self._compute_brier(calibrated, scores > 0.5)
        
        # Step 3: Conformal prediction intervals
        intervals = None
        coverage = None
        if return_intervals and self.conformal_predictor.is_fitted:
            intervals = self.conformal_predictor.predict_interval(calibrated)
            uncertain = self.conformal_predictor.predict_uncertainty(
                calibrated, self.uncertainty_threshold
            )
        else:
            # Simple heuristic: flag scores near threshold
            uncertain = np.abs(calibrated - self.uncertainty_threshold) < 0.1
        
        return CalibrationReport(
            original_scores=scores,
            calibrated_scores=calibrated,
            expected_calibration_error=ece,
            maximum_calibration_error=mce,
            brier_score=brier,
            confidence_intervals=intervals,
            coverage_rate=coverage,
            uncertain_predictions=uncertain
        )
    
    def _compute_ece(
        self,
        original: np.ndarray,
        calibrated: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Expected Calibration Error."""
        # Bin the calibrated scores
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        
        for i in range(n_bins):
            mask = (calibrated >= bin_edges[i]) & (calibrated < bin_edges[i+1])
            if i == n_bins - 1:  # Include right edge for last bin
                mask = (calibrated >= bin_edges[i]) & (calibrated <= bin_edges[i+1])
            
            if np.sum(mask) > 0:
                avg_confidence = np.mean(calibrated[mask])
                avg_original = np.mean(original[mask])
                ece += np.abs(avg_confidence - avg_original) * np.sum(mask)
        
        return ece / len(calibrated) if len(calibrated) > 0 else 0.0
    
    def _compute_mce(
        self,
        original: np.ndarray,
        calibrated: np.ndarray,
        n_bins: int = 10
    ) -> float:
        """Compute Maximum Calibration Error."""
        bin_edges = np.linspace(0, 1, n_bins + 1)
        max_error = 0.0
        
        for i in range(n_bins):
            mask = (calibrated >= bin_edges[i]) & (calibrated < bin_edges[i+1])
            if i == n_bins - 1:
                mask = (calibrated >= bin_edges[i]) & (calibrated <= bin_edges[i+1])
            
            if np.sum(mask) > 0:
                error = np.abs(np.mean(calibrated[mask]) - np.mean(original[mask]))
                max_error = max(max_error, error)
        
        return max_error
    
    def _compute_brier(
        self,
        calibrated: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute Brier score."""
        return np.mean((calibrated - labels) ** 2)
    
    def get_uncertain_indices(self, report: CalibrationReport) -> np.ndarray:
        """Get indices of predictions flagged for human review."""
        return np.where(report.uncertain_predictions)[0]


class MultiMetricCalibrator:
    """Calibrate multiple evaluation metrics simultaneously.
    
    Useful for HCAT's four dimensions (context relevancy, groundedness,
    completeness, answer_relevancy) that each need calibration.
    """
    
    def __init__(
        self,
        metric_names: List[str],
        calibration_method: str = 'isotonic',
        conformal_alpha: float = 0.1
    ):
        self.metric_names = metric_names
        self.calibrators = {
            name: HumanMachineCalibrator(calibration_method, conformal_alpha)
            for name in metric_names
        }
    
    def fit(
        self,
        machine_scores: Dict[str, np.ndarray],
        human_labels: Dict[str, np.ndarray]
    ) -> 'MultiMetricCalibrator':
        """Fit calibrators for all metrics."""
        for name in self.metric_names:
            if name in machine_scores and name in human_labels:
                self.calibrators[name].fit(
                    machine_scores[name],
                    human_labels[name]
                )
                logger.info(f"Fitted calibrator for {name}")
        return self
    
    def calibrate(
        self,
        scores: Dict[str, np.ndarray],
        return_intervals: bool = True
    ) -> Dict[str, CalibrationReport]:
        """Calibrate all metrics."""
        reports = {}
        for name, calibrator in self.calibrators.items():
            if name in scores:
                reports[name] = calibrator.calibrate(scores[name], return_intervals)
        return reports
    
    def get_all_uncertain(
        self,
        reports: Dict[str, CalibrationReport]
    ) -> np.ndarray:
        """Get indices where ANY metric is uncertain."""
        uncertain_masks = [
            r.uncertain_predictions for r in reports.values()
        ]
        return np.any(uncertain_masks, axis=0)


def compute_calibration_summary(
    reports: Dict[str, CalibrationReport]
) -> pd.DataFrame:
    """Create summary DataFrame from multiple calibration reports."""
    summaries = []
    for metric_name, report in reports.items():
        summary = report.to_dict()
        summary['metric'] = metric_name
        summaries.append(summary)
    return pd.DataFrame(summaries)


# Simple usage example
if __name__ == "__main__":
    # Generate synthetic calibration data
    np.random.seed(42)
    n_samples = 200
    
    # Simulated machine scores (slightly overconfident)
    machine_scores = np.random.beta(2, 2, n_samples)
    
    # Simulated human labels (noisy ground truth)
    human_labels = (machine_scores + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)
    
    # Calibrate
    calibrator = HumanMachineCalibrator()
    calibrator.fit(machine_scores, human_labels)
    
    # Test on new scores
    test_scores = np.array([0.3, 0.45, 0.5, 0.6, 0.85])
    report = calibrator.calibrate(test_scores)
    
    print("Calibration Results:")
    print(f"Original: {report.original_scores}")
    print(f"Calibrated: {report.calibrated_scores}")
    print(f"Uncertain flags: {report.uncertain_predictions}")
    print(f"ECE: {report.expected_calibration_error:.4f}")
