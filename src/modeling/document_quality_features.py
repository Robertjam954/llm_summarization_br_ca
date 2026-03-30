"""Document Quality Features for Medical LLM Summarization

Implements HCAT-inspired document quality assessment:
- OCR/Image quality metrics
- Text structure and readability
- Medical content quality indicators
- Information density and redundancy

Author: Generated for MSKCC Goel Lab project
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import re
from collections import Counter

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Optional CV dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    FITZ_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class DocumentQualityReport:
    """Container for document quality assessment."""
    
    # Image/OCR quality (0-1 scale)
    image_quality_score: float
    blur_score: float
    contrast_score: float
    skew_score: float
    
    # Text structure quality
    structure_score: float
    section_completeness: float
    temporal_consistency: float
    
    # Medical content quality
    medical_content_score: float
    information_density: float
    redundancy_score: float
    negation_clarity: float
    
    # Overall quality
    overall_quality_score: float
    quality_tier: str  # 'excellent', 'good', 'fair', 'poor'
    concerns: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'image_quality_score': self.image_quality_score,
            'blur_score': self.blur_score,
            'contrast_score': self.contrast_score,
            'skew_score': self.skew_score,
            'structure_score': self.structure_score,
            'section_completeness': self.section_completeness,
            'temporal_consistency': self.temporal_consistency,
            'medical_content_score': self.medical_content_score,
            'information_density': self.information_density,
            'redundancy_score': self.redundancy_score,
            'negation_clarity': self.negation_clarity,
            'overall_quality_score': self.overall_quality_score,
            'quality_tier': self.quality_tier,
            'concerns_count': len(self.concerns)
        }


class ImageQualityAnalyzer:
    """Analyze image/OCR quality of scanned documents."""
    
    def __init__(self):
        self.available = CV2_AVAILABLE and PIL_AVAILABLE
    
    def analyze_page(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Analyze quality metrics for a single page."""
        if not self.available or gray_image is None:
            return self._default_scores()
        
        try:
            # Blur detection (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
            # Normalize: higher is sharper
            blur_score = min(laplacian_var / 1000, 1.0)
            
            # Contrast (RMS)
            rms_contrast = float(gray_image.astype(float).std())
            contrast_score = min(rms_contrast / 80, 1.0)
            
            # Brightness spread
            p5 = np.percentile(gray_image, 5)
            p95 = np.percentile(gray_image, 95)
            spread_score = min((p95 - p5) / 200, 1.0)
            
            # Skew detection
            edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)
            
            if lines is not None and len(lines) > 0:
                angles = []
                for rho, theta in lines[:, 0]:
                    angle = (theta * 180 / np.pi) - 90
                    if -45 < angle < 45:
                        angles.append(abs(angle))
                skew_angle = np.median(angles) if angles else 0.0
            else:
                skew_angle = 0.0
            
            skew_score = max(0, 1.0 - (skew_angle / 10.0))
            
            return {
                'blur_score': blur_score,
                'contrast_score': contrast_score,
                'spread_score': spread_score,
                'skew_score': skew_score,
                'laplacian_var': laplacian_var,
                'rms_contrast': rms_contrast,
                'skew_angle': skew_angle
            }
            
        except Exception as e:
            logger.warning(f"Image analysis failed: {e}")
            return self._default_scores()
    
    def _default_scores(self) -> Dict[str, float]:
        return {
            'blur_score': 0.5,
            'contrast_score': 0.5,
            'spread_score': 0.5,
            'skew_score': 0.5,
            'laplacian_var': 0.0,
            'rms_contrast': 0.0,
            'skew_angle': 0.0
        }


class TextStructureAnalyzer:
    """Analyze structure and organization of medical text."""
    
    # Expected sections in clinical documents
    CLINICAL_SECTIONS = [
        'chief complaint', 'history', 'physical examination', 'vitals',
        'assessment', 'plan', 'diagnosis', 'medications', 'allergies',
        'past medical history', 'social history', 'family history',
        'review of systems', 'laboratory', 'imaging', 'procedures',
        'discharge', 'follow up', 'instructions'
    ]
    
    def __init__(self):
        self.section_pattern = re.compile(
            r'\n\s*([A-Z][A-Za-z\s]+?):\s*\n|\n\s*(\d+\.\s+[A-Z][A-Za-z\s]+?)\s*\n',
            re.IGNORECASE
        )
    
    def analyze_structure(self, text: str) -> Dict[str, float]:
        """Analyze document structure."""
        text_lower = text.lower()
        
        # Section detection
        detected_sections = []
        for section in self.CLINICAL_SECTIONS:
            if section in text_lower:
                detected_sections.append(section)
        
        section_completeness = len(detected_sections) / len(self.CLINICAL_SECTIONS)
        
        # Header structure score
        header_matches = len(self.section_pattern.findall(text))
        structure_score = min(header_matches / 5, 1.0)
        
        # Temporal consistency (check for date/time patterns)
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY
            r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
        ]
        dates_found = sum(len(re.findall(p, text_lower)) for p in date_patterns)
        temporal_consistency = min(dates_found / 3, 1.0)
        
        # Paragraph/sentence structure
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Optimal sentence length: 10-25 words
        if 10 <= avg_sentence_length <= 25:
            sentence_structure_score = 1.0
        elif avg_sentence_length < 5:
            sentence_structure_score = 0.3
        else:
            sentence_structure_score = max(0, 1.0 - abs(avg_sentence_length - 17.5) / 50)
        
        return {
            'section_completeness': section_completeness,
            'structure_score': structure_score,
            'temporal_consistency': temporal_consistency,
            'sentence_structure_score': sentence_structure_score,
            'detected_sections': detected_sections
        }


class MedicalContentAnalyzer:
    """Analyze quality of medical content in text."""
    
    # Medical indicators
    MEDICAL_TERMS = {
        'anatomy': ['lung', 'liver', 'heart', 'kidney', 'brain', 'abdomen', 'chest'],
        'procedures': ['biopsy', 'resection', 'excision', 'aspiration', 'catheterization'],
        'medications': ['mg', 'mcg', 'units', 'tablet', 'capsule', 'injection', 'infusion'],
        'diagnoses': ['carcinoma', 'sarcoma', 'lymphoma', 'leukemia', 'metastasis'],
        'labs': ['hemoglobin', 'wbc', 'plt', 'creatinine', 'sodium', 'potassium', 'glucose']
    }
    
    # Negation words (for clarity check)
    NEGATION_WORDS = {
        'no', 'not', 'none', 'negative', 'absent', 'without', 
        'unremarkable', 'normal', 'deny', 'denied', 'non-'
    }
    
    def __init__(self):
        pass
    
    def analyze_content(self, text: str) -> Dict[str, float]:
        """Analyze medical content quality."""
        text_lower = text.lower()
        tokens = re.findall(r'\b\w+\b', text_lower)
        
        # Information density
        medical_term_count = 0
        for category, terms in self.MEDICAL_TERMS.items():
            medical_term_count += sum(1 for t in tokens if t in terms)
        
        information_density = min(medical_term_count / max(len(tokens) / 100, 1), 1.0)
        
        # Redundancy detection (repeated phrases)
        bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
        bigram_counts = Counter(bigrams)
        repeated_bigrams = sum(1 for count in bigram_counts.values() if count > 3)
        redundancy_score = max(0, 1.0 - (repeated_bigrams / max(len(bigrams) / 100, 1)))
        
        # Negation clarity (negations should be near what they negate)
        negation_positions = [i for i, t in enumerate(tokens) if t in self.NEGATION_WORDS]
        
        if negation_positions:
            # Check if negations are well-distributed (not all clustered)
            if len(negation_positions) > 1:
                gaps = [negation_positions[i+1] - negation_positions[i] 
                       for i in range(len(negation_positions)-1)]
                avg_gap = np.mean(gaps) if gaps else 0
                negation_clarity = min(avg_gap / 50, 1.0)
            else:
                negation_clarity = 1.0
        else:
            negation_clarity = 1.0  # No negations = clear
        
        # Medical content score
        total_medical_indicators = sum(
            1 for t in tokens 
            if any(t in terms for terms in self.MEDICAL_TERMS.values())
        )
        medical_content_score = min(total_medical_indicators / 20, 1.0)
        
        return {
            'information_density': information_density,
            'redundancy_score': redundancy_score,
            'negation_clarity': negation_clarity,
            'medical_content_score': medical_content_score,
            'medical_term_count': medical_term_count
        }


class DocumentQualityEvaluator:
    """Main entry point for HCAT-inspired document quality assessment.
    
    Combines image quality, text structure, and medical content analysis.
    """
    
    def __init__(self):
        self.image_analyzer = ImageQualityAnalyzer()
        self.structure_analyzer = TextStructureAnalyzer()
        self.content_analyzer = MedicalContentAnalyzer()
    
    def evaluate(
        self,
        text: str,
        gray_image: Optional[np.ndarray] = None
    ) -> DocumentQualityReport:
        """Evaluate document quality comprehensively."""
        
        # Image/OCR quality
        if gray_image is not None:
            image_metrics = self.image_analyzer.analyze_page(gray_image)
        else:
            image_metrics = self.image_analyzer._default_scores()
        
        # Aggregate image quality
        image_quality_score = np.mean([
            image_metrics['blur_score'],
            image_metrics['contrast_score'],
            image_metrics['skew_score']
        ])
        
        # Text structure
        structure_metrics = self.structure_analyzer.analyze_structure(text)
        
        # Medical content
        content_metrics = self.content_analyzer.analyze_content(text)
        
        # Calculate overall score (weighted combination)
        weights = {
            'image': 0.25,
            'structure': 0.25,
            'content': 0.50
        }
        
        overall_score = (
            weights['image'] * image_quality_score +
            weights['structure'] * structure_metrics['structure_score'] +
            weights['content'] * content_metrics['medical_content_score']
        )
        
        # Determine quality tier
        if overall_score >= 0.8:
            quality_tier = 'excellent'
        elif overall_score >= 0.6:
            quality_tier = 'good'
        elif overall_score >= 0.4:
            quality_tier = 'fair'
        else:
            quality_tier = 'poor'
        
        # Identify concerns
        concerns = []
        if image_metrics['blur_score'] < 0.5:
            concerns.append('low_image_clarity')
        if image_metrics['skew_score'] < 0.7:
            concerns.append('document_skew')
        if structure_metrics['section_completeness'] < 0.3:
            concerns.append('incomplete_sections')
        if content_metrics['redundancy_score'] < 0.5:
            concerns.append('high_redundancy')
        if content_metrics['information_density'] < 0.2:
            concerns.append('low_information_density')
        
        return DocumentQualityReport(
            image_quality_score=image_quality_score,
            blur_score=image_metrics['blur_score'],
            contrast_score=image_metrics['contrast_score'],
            skew_score=image_metrics['skew_score'],
            structure_score=structure_metrics['structure_score'],
            section_completeness=structure_metrics['section_completeness'],
            temporal_consistency=structure_metrics['temporal_consistency'],
            medical_content_score=content_metrics['medical_content_score'],
            information_density=content_metrics['information_density'],
            redundancy_score=content_metrics['redundancy_score'],
            negation_clarity=content_metrics['negation_clarity'],
            overall_quality_score=overall_score,
            quality_tier=quality_tier,
            concerns=concerns
        )
    
    def evaluate_from_pdf(
        self,
        pdf_path: Path,
        page_idx: int = 0,
        dpi: int = 300
    ) -> Tuple[DocumentQualityReport, str]:
        """Evaluate a specific page from a PDF."""
        
        if not FITZ_AVAILABLE:
            raise ImportError("PyMuPDF (fitz) required for PDF processing")
        
        doc = fitz.open(pdf_path)
        try:
            page = doc.load_page(page_idx)
            
            # Extract text
            text = page.get_text()
            
            # Render to image for quality analysis
            zoom = dpi / 72.0
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if CV2_AVAILABLE else None
            
            report = self.evaluate(text, gray)
            return report, text
            
        finally:
            doc.close()
    
    def evaluate_batch_texts(
        self,
        texts: List[str]
    ) -> List[DocumentQualityReport]:
        """Evaluate a batch of text documents."""
        return [self.evaluate(t) for t in texts]


def create_quality_summary_df(reports: List[DocumentQualityReport]) -> pd.DataFrame:
    """Convert list of DocumentQualityReports to a DataFrame."""
    return pd.DataFrame([r.to_dict() for r in reports])


# Simple usage
if __name__ == "__main__":
    evaluator = DocumentQualityEvaluator()
    
    test_text = """
    CHIEF COMPLAINT: Chest pain
    
    HISTORY: Patient presents with 3-day history of chest pain.
    
    PHYSICAL EXAMINATION: Vital signs stable. Heart rate 72.
    
    ASSESSMENT: Possible angina. Plan for stress test.
    
    DIAGNOSIS: Stable angina pectoris.
    """
    
    report = evaluator.evaluate(test_text)
    print(f"Overall Quality: {report.overall_quality_score:.3f}")
    print(f"Quality Tier: {report.quality_tier}")
    print(f"Concerns: {report.concerns}")
