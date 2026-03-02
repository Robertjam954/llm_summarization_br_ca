"""Embedding-Based Evaluation Metrics for Medical LLM Summarization

Implements HCAT framework's Explainable Evaluation Metrics:
- Context Relevancy: Does retrieved document help answer the query?
- Groundedness: Is answer based only on provided documents?
- Completeness: Did AI mention all key points from source?
- Answer Relevancy: Did AI actually answer the user's question?

Author: Generated for MSKCC Goel Lab project
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import re

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

# Optional sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingMetricsReport:
    """Container for embedding-based evaluation results."""
    
    # HCAT four dimensions
    context_relevancy: float  # 0-1, higher = retrieved docs are relevant
    groundedness: float  # 0-1, higher = answer grounded in context
    completeness: float  # 0-1, higher = all key points covered
    answer_relevancy: float  # 0-1, higher = answers the actual question
    
    # Component scores
    semantic_similarity: float
    token_overlap_f1: float
    key_phrase_coverage: float
    
    # Thresholds for human review
    requires_human_review: bool
    low_confidence_dimensions: List[str]
    
    def to_dict(self) -> Dict:
        return {
            'context_relevancy': self.context_relevancy,
            'groundedness': self.groundedness,
            'completeness': self.completeness,
            'answer_relevancy': self.answer_relevancy,
            'semantic_similarity': self.semantic_similarity,
            'token_overlap_f1': self.token_overlap_f1,
            'key_phrase_coverage': self.key_phrase_coverage,
            'requires_human_review': self.requires_human_review,
            'low_confidence_dimensions': self.low_confidence_dimensions
        }


class EmbeddingModelWrapper:
    """Wrapper for embedding models with fallback options."""
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.model = None
        self.model_name = model_name
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load {model_name}: {e}")
    
    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.model is None:
            # Fallback: simple bag-of-characters embedding
            return self._fallback_encode(texts)
        
        try:
            # Truncate long texts
            truncated = [t[:4000] for t in texts]
            return self.model.encode(truncated, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            logger.warning(f"Model encoding failed: {e}, using fallback")
            return self._fallback_encode(texts)
    
    def _fallback_encode(self, texts: List[str]) -> np.ndarray:
        """Simple character n-gram based fallback embedding."""
        # Create character n-gram frequency vectors
        embeddings = []
        for text in texts:
            text_lower = text.lower()
            # Use character trigrams
            ngrams = [text_lower[i:i+3] for i in range(len(text_lower)-2)]
            # Create a simple frequency vector (256 buckets)
            vec = np.zeros(256)
            for ng in ngrams:
                idx = sum(ord(c) for c in ng[:3]) % 256
                vec[idx] += 1
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)


class TextPreprocessor:
    """Preprocess text for evaluation metrics."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s.,;:!?-]', '', text)
        return text.strip().lower()
    
    @staticmethod
    def extract_key_phrases(text: str, min_length: int = 3) -> List[str]:
        """Extract key noun phrases and medical terms."""
        # Simple extraction: noun phrases and terms with medical suffixes
        text_lower = text.lower()
        
        # Medical term patterns
        medical_suffixes = r'(?:itis|osis|oma|emia|pathy|algia|ectasia|plasia|megaly)'
        medical_terms = re.findall(rf'\b\w+{medical_suffixes}\b', text_lower)
        
        # Noun phrases (simplified: consecutive capitalized words or medical terms)
        noun_phrases = re.findall(r'\b(?:[A-Z][a-z]+\s+){1,3}[a-z]+\b', text)
        
        # Numbers and measurements
        numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:mm|cm|mg|ml|%|mmol|units)\b', text_lower)
        
        # Combine and filter
        all_phrases = medical_terms + [p.lower() for p in noun_phrases] + numbers
        filtered = [p for p in all_phrases if len(p) >= min_length]
        
        return list(set(filtered))
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class EmbeddingEvaluator:
    """Compute HCAT embedding-based evaluation metrics.
    
    Implements the four 'Functionality' dimensions from HCAT:
    1. Context Relevancy: Retrieved documents match the query
    2. Groundedness: Answer is based only on provided documents
    3. Completeness: All key points from source are mentioned
    4. Answer Relevancy: Answer addresses the user's question
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self.embedding_model = EmbeddingModelWrapper(model_name)
        self.preprocessor = TextPreprocessor()
    
    def compute_context_relevancy(
        self,
        query: str,
        retrieved_documents: List[str]
    ) -> float:
        """Measure if retrieved documents are relevant to query.
        
        Computes max semantic similarity between query and each document.
        """
        if not retrieved_documents:
            return 0.0
        
        all_texts = [query] + retrieved_documents
        embeddings = self.embedding_model.encode(all_texts)
        
        query_emb = embeddings[0].reshape(1, -1)
        doc_embs = embeddings[1:]
        
        similarities = cosine_similarity(query_emb, doc_embs)[0]
        
        # Return max similarity (at least one doc should be highly relevant)
        # and average similarity (overall quality)
        max_sim = float(np.max(similarities))
        avg_sim = float(np.mean(similarities))
        
        # Weighted combination
        return 0.7 * max_sim + 0.3 * avg_sim
    
    def compute_groundedness(
        self,
        answer: str,
        source_documents: List[str],
        claim_threshold: float = 0.65
    ) -> float:
        """Measure if answer is grounded in source documents.
        
        Breaks answer into sentences, checks each against source.
        """
        if not source_documents or not answer:
            return 0.0
        
        # Combine source documents
        source_combined = " ".join(source_documents)
        
        # Split answer into claims/sentences
        answer_sentences = self.preprocessor.split_sentences(answer)
        
        if not answer_sentences:
            return 0.0
        
        # Encode source and answer sentences
        all_texts = [source_combined] + answer_sentences
        embeddings = self.embedding_model.encode(all_texts)
        
        source_emb = embeddings[0].reshape(1, -1)
        answer_embs = embeddings[1:]
        
        # Check each answer sentence against source
        similarities = cosine_similarity(answer_embs, source_emb)
        
        # Count grounded claims (above threshold)
        grounded_count = np.sum(similarities.flatten() >= claim_threshold)
        
        return grounded_count / len(answer_sentences)
    
    def compute_completeness(
        self,
        answer: str,
        reference: str,
        use_key_phrases: bool = True
    ) -> float:
        """Measure if all key points from reference are in answer.
        
        Uses key phrase coverage and semantic similarity.
        """
        if not reference or not answer:
            return 0.0
        
        # Method 1: Key phrase coverage
        if use_key_phrases:
            ref_phrases = self.preprocessor.extract_key_phrases(reference)
            answer_lower = answer.lower()
            
            if not ref_phrases:
                phrase_coverage = 1.0
            else:
                matched = sum(1 for p in ref_phrases if p in answer_lower)
                phrase_coverage = matched / len(ref_phrases)
        else:
            phrase_coverage = 0.5  # Default if disabled
        
        # Method 2: Semantic similarity between answer and reference
        embeddings = self.embedding_model.encode([reference, answer])
        semantic_sim = float(cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0, 0])
        
        # Combine: phrase coverage ensures specific details, semantic ensures overall meaning
        return 0.6 * phrase_coverage + 0.4 * semantic_sim
    
    def compute_answer_relevancy(
        self,
        query: str,
        answer: str,
        context: Optional[str] = None
    ) -> float:
        """Measure if answer actually addresses the query.
        
        Uses semantic similarity and query term presence.
        """
        if not query or not answer:
            return 0.0
        
        # Semantic similarity between query and answer
        embeddings = self.embedding_model.encode([query, answer])
        semantic_sim = float(cosine_similarity(
            embeddings[0].reshape(1, -1),
            embeddings[1].reshape(1, -1)
        )[0, 0])
        
        # Query term coverage
        query_terms = set(re.findall(r'\b\w+\b', query.lower()))
        query_terms = {t for t in query_terms if len(t) > 2}  # Filter short words
        answer_lower = answer.lower()
        
        if not query_terms:
            term_coverage = 1.0
        else:
            matched = sum(1 for t in query_terms if t in answer_lower)
            term_coverage = matched / len(query_terms)
        
        # Weight semantic similarity more (catches paraphrased answers)
        return 0.7 * semantic_sim + 0.3 * term_coverage
    
    def compute_token_overlap(
        self,
        reference: str,
        answer: str
    ) -> float:
        """Compute F1 score of token overlap."""
        ref_tokens = set(self.preprocessor.clean_text(reference).split())
        ans_tokens = set(self.preprocessor.clean_text(answer).split())
        
        if not ref_tokens or not ans_tokens:
            return 0.0
        
        intersection = len(ref_tokens & ans_tokens)
        
        precision = intersection / len(ans_tokens) if ans_tokens else 0
        recall = intersection / len(ref_tokens) if ref_tokens else 0
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate(
        self,
        query: str,
        answer: str,
        reference: Optional[str] = None,
        retrieved_documents: Optional[List[str]] = None,
        source_documents: Optional[List[str]] = None
    ) -> EmbeddingMetricsReport:
        """Compute all HCAT embedding-based metrics.
        
        Args:
            query: The user's question/prompt
            answer: The LLM's generated answer
            reference: Ground truth reference (for completeness)
            retrieved_documents: Documents retrieved by RAG (for context relevancy)
            source_documents: Documents provided to LLM (for groundedness)
        """
        
        # Context Relevancy
        if retrieved_documents:
            context_relevancy = self.compute_context_relevancy(query, retrieved_documents)
        else:
            context_relevancy = 1.0  # Assume perfect if not provided
        
        # Groundedness
        if source_documents:
            groundedness = self.compute_groundedness(answer, source_documents)
        else:
            groundedness = 0.5  # Unknown - middle value
        
        # Completeness
        if reference:
            completeness = self.compute_completeness(answer, reference)
        else:
            completeness = 0.5  # Unknown - middle value
        
        # Answer Relevancy
        answer_relevancy = self.compute_answer_relevancy(query, answer)
        
        # Semantic similarity (answer vs reference)
        if reference:
            embeddings = self.embedding_model.encode([reference, answer])
            semantic_similarity = float(cosine_similarity(
                embeddings[0].reshape(1, -1),
                embeddings[1].reshape(1, -1)
            )[0, 0])
        else:
            semantic_similarity = answer_relevancy  # Proxy
        
        # Token overlap
        if reference:
            token_overlap_f1 = self.compute_token_overlap(reference, answer)
        else:
            token_overlap_f1 = 0.5
        
        # Key phrase coverage
        if reference:
            ref_phrases = self.preprocessor.extract_key_phrases(reference)
            answer_lower = answer.lower()
            if ref_phrases:
                matched = sum(1 for p in ref_phrases if p in answer_lower)
                key_phrase_coverage = matched / len(ref_phrases)
            else:
                key_phrase_coverage = 1.0
        else:
            key_phrase_coverage = 0.5
        
        # Determine if human review needed
        low_threshold = 0.5
        low_confidence = []
        
        if context_relevancy < low_threshold:
            low_confidence.append('context_relevancy')
        if groundedness < low_threshold:
            low_confidence.append('groundedness')
        if completeness < low_threshold:
            low_confidence.append('completeness')
        if answer_relevancy < low_threshold:
            low_confidence.append('answer_relevancy')
        
        requires_review = len(low_confidence) >= 2 or any([
            context_relevancy < 0.3,
            groundedness < 0.3,
            answer_relevancy < 0.3
        ])
        
        return EmbeddingMetricsReport(
            context_relevancy=context_relevancy,
            groundedness=groundedness,
            completeness=completeness,
            answer_relevancy=answer_relevancy,
            semantic_similarity=semantic_similarity,
            token_overlap_f1=token_overlap_f1,
            key_phrase_coverage=key_phrase_coverage,
            requires_human_review=requires_review,
            low_confidence_dimensions=low_confidence
        )
    
    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        references: Optional[List[str]] = None,
        retrieved_documents: Optional[List[List[str]]] = None,
        source_documents: Optional[List[List[str]]] = None
    ) -> List[EmbeddingMetricsReport]:
        """Evaluate a batch of query-answer pairs."""
        results = []
        
        for i in range(len(queries)):
            ref = references[i] if references and i < len(references) else None
            retr = retrieved_documents[i] if retrieved_documents and i < len(retrieved_documents) else None
            src = source_documents[i] if source_documents and i < len(source_documents) else None
            
            result = self.evaluate(
                query=queries[i],
                answer=answers[i],
                reference=ref,
                retrieved_documents=retr,
                source_documents=src
            )
            results.append(result)
        
        return results


def create_metrics_summary_df(reports: List[EmbeddingMetricsReport]) -> pd.DataFrame:
    """Convert list of EmbeddingMetricsReports to a DataFrame."""
    return pd.DataFrame([r.to_dict() for r in reports])


# Simple usage example
if __name__ == "__main__":
    evaluator = EmbeddingEvaluator()
    
    query = "What is the patient's diagnosis?"
    answer = "The patient has stage II breast cancer with positive hormone receptors."
    reference = "Patient diagnosed with stage II invasive ductal carcinoma, ER/PR positive, HER2 negative."
    
    report = evaluator.evaluate(query, answer, reference)
    print(f"Context Relevancy: {report.context_relevancy:.3f}")
    print(f"Groundedness: {report.groundedness:.3f}")
    print(f"Completeness: {report.completeness:.3f}")
    print(f"Answer Relevancy: {report.answer_relevancy:.3f}")
