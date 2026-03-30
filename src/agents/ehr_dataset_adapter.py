"""
EHR Deidentification Dataset Adapter
Converts NER test.jsonl data into evaluation-ready format for the multi-agent system
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
import pandas as pd

from config import get_config

config = get_config()


@dataclass
class NERSample:
    """Single NER sample from test.jsonl"""
    tokens: List[str]
    labels: List[str]
    current_sent_info: List[Dict[str, Any]]
    note_sent_info: Dict[str, Any]
    
    def get_text(self) -> str:
        """Reconstruct text from tokens"""
        return " ".join(self.tokens)
    
    def get_phi_entities(self) -> List[Dict[str, str]]:
        """Extract PHI entities (labels != 'O' and != 'NA')"""
        entities = []
        for token_info in self.current_sent_info:
            if token_info.get("label") not in ["O", "NA"]:
                entities.append({
                    "text": token_info["text"],
                    "label": token_info["label"],
                    "start": token_info["start"],
                    "end": token_info["end"]
                })
        return entities
    
    def get_note_id(self) -> str:
        """Get note ID"""
        return self.note_sent_info.get("note_id", "unknown")


class EHRDatasetAdapter:
    """Adapter to convert EHR NER dataset to evaluation format"""
    
    def __init__(self, test_jsonl_path: Optional[Path] = None):
        self.test_path = test_jsonl_path or Path(
            r"C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca\ehr_deidentification\data\ner_datasets\test.jsonl"
        )
        self.samples: List[NERSample] = []
        self.goldens_dir = config.data.goldens_dir
        
    def load_samples(self) -> List[NERSample]:
        """Load all samples from test.jsonl"""
        samples = []
        
        with open(self.test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    sample = NERSample(
                        tokens=data["tokens"],
                        labels=data["labels"],
                        current_sent_info=data.get("current_sent_info", []),
                        note_sent_info=data.get("note_sent_info", {})
                    )
                    samples.append(sample)
        
        self.samples = samples
        print(f"Loaded {len(samples)} samples from {self.test_path}")
        return samples
    
    def create_deidentification_goldens(self) -> List[Golden]:
        """Create golden test cases for deidentification task"""
        if not self.samples:
            self.load_samples()
        
        goldens = []
        
        for idx, sample in enumerate(self.samples):
            text = sample.get_text()
            phi_entities = sample.get_phi_entities()
            note_id = sample.get_note_id()
            
            # Skip samples with no meaningful text (all NA labels)
            if all(label == "NA" for label in sample.labels):
                continue
            
            # Input: original text with PHI
            input_text = f"Deidentify the following clinical text:\n\n{text}"
            
            # Expected output: text with PHI entities marked
            expected_output = self._create_expected_deidentified_text(text, phi_entities)
            
            # Context: list of PHI entities that should be removed
            context = [
                f"{entity['label']}: {entity['text']}" 
                for entity in phi_entities
            ]
            
            golden = Golden(
                input=input_text,
                expected_output=expected_output,
                context=context,
                additional_metadata={
                    "sample_id": idx,
                    "note_id": note_id,
                    "phi_count": len(phi_entities),
                    "task": "deidentification"
                }
            )
            
            goldens.append(golden)
        
        print(f"Created {len(goldens)} deidentification goldens")
        return goldens
    
    def create_summarization_goldens(self) -> List[Golden]:
        """Create golden test cases for clinical summarization task"""
        if not self.samples:
            self.load_samples()
        
        goldens = []
        
        # Group samples by note_id to create full documents
        notes = {}
        for sample in self.samples:
            note_id = sample.get_note_id()
            if note_id not in notes:
                notes[note_id] = []
            notes[note_id].append(sample)
        
        for note_id, note_samples in notes.items():
            # Combine all text from the note
            full_text = " ".join([s.get_text() for s in note_samples])
            
            # Skip very short notes
            if len(full_text.split()) < 20:
                continue
            
            input_text = f"Summarize the following clinical note:\n\n{full_text}"
            
            # Expected output: key clinical information
            expected_output = self._extract_clinical_summary(note_samples)
            
            # Context: original text segments
            context = [s.get_text() for s in note_samples if s.get_text().strip()]
            
            golden = Golden(
                input=input_text,
                expected_output=expected_output,
                context=context[:10],  # Limit context to first 10 segments
                additional_metadata={
                    "note_id": note_id,
                    "segment_count": len(note_samples),
                    "task": "summarization"
                }
            )
            
            goldens.append(golden)
        
        print(f"Created {len(goldens)} summarization goldens from {len(notes)} notes")
        return goldens
    
    def _create_expected_deidentified_text(self, text: str, phi_entities: List[Dict]) -> str:
        """Create expected deidentified text by replacing PHI with placeholders"""
        deidentified = text
        
        # Sort entities by start position (reverse to avoid offset issues)
        sorted_entities = sorted(phi_entities, key=lambda x: x["start"], reverse=True)
        
        for entity in sorted_entities:
            label = entity["label"]
            placeholder = f"[{label}]"
            # Simple replacement (in production, would use proper span replacement)
            deidentified = deidentified.replace(entity["text"], placeholder)
        
        return deidentified
    
    def _extract_clinical_summary(self, note_samples: List[NERSample]) -> str:
        """Extract key clinical information for expected summary"""
        full_text = " ".join([s.get_text() for s in note_samples])
        
        # Simple extraction of key sections
        summary_parts = []
        
        if "HISTORY OF PRESENT ILLNESS" in full_text:
            summary_parts.append("Patient history documented")
        
        if "HOSPITAL COURSE" in full_text or "TREATMENT" in full_text:
            summary_parts.append("Hospital course and treatment documented")
        
        if "SOCIAL HISTORY" in full_text:
            summary_parts.append("Social history documented")
        
        if "Discharge" in full_text:
            summary_parts.append("Discharge information documented")
        
        return "; ".join(summary_parts) if summary_parts else "Clinical note summary"
    
    def build_evaluation_dataset(
        self, 
        task: str = "deidentification",
        name: str = "ehr_test_dataset"
    ) -> EvaluationDataset:
        """Build evaluation dataset for specified task"""
        
        if task == "deidentification":
            goldens = self.create_deidentification_goldens()
        elif task == "summarization":
            goldens = self.create_summarization_goldens()
        else:
            raise ValueError(f"Unknown task: {task}. Choose 'deidentification' or 'summarization'")
        
        dataset = EvaluationDataset(goldens=goldens)
        
        return dataset
    
    def save_dataset(self, dataset: EvaluationDataset, name: str):
        """Save dataset to JSON file"""
        output_path = self.goldens_dir / f"{name}_goldens.json"
        
        goldens_data = [
            {
                "input": g.input,
                "expected_output": g.expected_output,
                "context": g.context,
                "metadata": g.additional_metadata
            }
            for g in dataset.goldens
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(goldens_data, f, indent=2)
        
        print(f"Dataset saved to {output_path}")
        print(f"Total goldens: {len(dataset.goldens)}")
        
        return output_path
    
    def export_summary_csv(self, name: str = "ehr_test_dataset") -> Path:
        """Export dataset summary to CSV"""
        if not self.samples:
            self.load_samples()
        
        rows = []
        for idx, sample in enumerate(self.samples):
            phi_entities = sample.get_phi_entities()
            rows.append({
                "sample_id": idx,
                "note_id": sample.get_note_id(),
                "token_count": len(sample.tokens),
                "phi_count": len(phi_entities),
                "text_preview": sample.get_text()[:100]
            })
        
        df = pd.DataFrame(rows)
        output_path = self.goldens_dir / f"{name}_summary.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Summary exported to {output_path}")
        return output_path


def create_test_cases_from_ehr_data(
    llm_app_fn: callable,
    task: str = "deidentification"
) -> List[LLMTestCase]:
    """Create test cases by running LLM app on EHR test data"""
    
    adapter = EHRDatasetAdapter()
    dataset = adapter.build_evaluation_dataset(task=task)
    
    test_cases = []
    
    for golden in dataset.goldens:
        try:
            actual_output = llm_app_fn(golden.input)
            
            test_case = LLMTestCase(
                input=golden.input,
                actual_output=actual_output,
                expected_output=golden.expected_output,
                context=golden.context,
                retrieval_context=golden.context
            )
            
            test_cases.append(test_case)
            
        except Exception as e:
            print(f"Error creating test case: {e}")
            continue
    
    return test_cases


if __name__ == "__main__":
    print("=" * 70)
    print("EHR DEIDENTIFICATION DATASET ADAPTER")
    print("=" * 70)
    
    adapter = EHRDatasetAdapter()
    
    # Load samples
    samples = adapter.load_samples()
    print(f"\nLoaded {len(samples)} samples")
    
    # Show sample statistics
    phi_counts = [len(s.get_phi_entities()) for s in samples]
    print(f"PHI entities found: {sum(phi_counts)} total")
    print(f"Average PHI per sample: {sum(phi_counts)/len(samples):.2f}")
    
    # Build deidentification dataset
    print("\n" + "=" * 70)
    print("Building Deidentification Dataset")
    print("=" * 70)
    deid_dataset = adapter.build_evaluation_dataset(task="deidentification")
    adapter.save_dataset(deid_dataset, "ehr_deidentification")
    
    # Build summarization dataset
    print("\n" + "=" * 70)
    print("Building Summarization Dataset")
    print("=" * 70)
    summ_dataset = adapter.build_evaluation_dataset(task="summarization")
    adapter.save_dataset(summ_dataset, "ehr_summarization")
    
    # Export summary
    adapter.export_summary_csv("ehr_test_dataset")
    
    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)
