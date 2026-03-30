"""
Dataset Builder for Golden Test Cases
Creates evaluation datasets from clinical summaries
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
import pandas as pd

from config import get_config, CLINICAL_FEATURES

config = get_config()


@dataclass
class ClinicalGolden:
    """Clinical summary golden test case"""
    patient_id: str
    mrn: str
    input_query: str
    expected_output: str
    retrieval_context: List[str]
    features: Dict[str, str]
    metadata: Dict[str, Any]
    
    def to_golden(self) -> Golden:
        """Convert to DeepEval Golden"""
        return Golden(
            input=self.input_query,
            expected_output=self.expected_output,
            context=self.retrieval_context,
            additional_metadata={
                "patient_id": self.patient_id,
                "mrn": self.mrn,
                "features": self.features,
                **self.metadata
            }
        )


class DatasetBuilder:
    """Build evaluation datasets from clinical data"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or config.data.processed_dir
        self.goldens_dir = config.data.goldens_dir
        self.datasets: Dict[str, EvaluationDataset] = {}
    
    def load_parsed_summaries(self) -> List[Dict[str, Any]]:
        """Load parsed v2 summaries"""
        summaries = []
        
        for summary_file in self.data_dir.glob("*_parsed.json"):
            with open(summary_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                summaries.append(data)
        
        return summaries
    
    def load_features_csv(self) -> pd.DataFrame:
        """Load extracted features CSV"""
        csv_path = self.data_dir / "v2_summaries_features_extracted.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path)
        return pd.DataFrame()
    
    def create_golden_from_summary(self, summary_data: Dict[str, Any]) -> ClinicalGolden:
        """Create golden test case from summary"""
        patient_id = summary_data.get("patient_initials", "")
        mrn = str(summary_data.get("mrn", ""))
        
        lesions = summary_data.get("lesions", [])
        if not lesions:
            lesions = [{}]
        
        lesion = lesions[0]
        
        features = {}
        for feature in CLINICAL_FEATURES:
            feature_data = lesion.get(feature, {})
            if isinstance(feature_data, dict):
                features[feature] = feature_data.get("value", "")
            else:
                features[feature] = str(feature_data)
        
        input_query = f"Extract clinical features for patient {patient_id} (MRN: {mrn})"
        
        expected_output = json.dumps({
            "patient_id": patient_id,
            "mrn": mrn,
            "features": features
        }, indent=2)
        
        retrieval_context = []
        for feature, value in features.items():
            if value and value != "Not reported":
                retrieval_context.append(f"{feature}: {value}")
        
        return ClinicalGolden(
            patient_id=patient_id,
            mrn=mrn,
            input_query=input_query,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
            features=features,
            metadata={
                "source": "v2_prompt_test",
                "lesion_count": len(lesions)
            }
        )
    
    def build_dataset(self, name: str = "clinical_summarization") -> EvaluationDataset:
        """Build evaluation dataset from all summaries"""
        summaries = self.load_parsed_summaries()
        
        goldens = []
        for summary in summaries:
            try:
                clinical_golden = self.create_golden_from_summary(summary)
                goldens.append(clinical_golden.to_golden())
            except Exception as e:
                print(f"Error creating golden for summary: {e}")
                continue
        
        dataset = EvaluationDataset(goldens=goldens)
        self.datasets[name] = dataset
        
        return dataset
    
    def save_dataset(self, name: str, alias: Optional[str] = None):
        """Save dataset to file and optionally push to DeepEval"""
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found. Build it first.")
        
        dataset = self.datasets[name]
        
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
        
        if alias:
            try:
                dataset.push(alias=alias)
                print(f"Dataset pushed to DeepEval with alias: {alias}")
            except Exception as e:
                print(f"Failed to push dataset: {e}")
    
    def load_dataset(self, name: str) -> EvaluationDataset:
        """Load dataset from file"""
        input_path = self.goldens_dir / f"{name}_goldens.json"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {input_path}")
        
        with open(input_path, 'r', encoding='utf-8') as f:
            goldens_data = json.load(f)
        
        goldens = [
            Golden(
                input=g["input"],
                expected_output=g["expected_output"],
                context=g.get("context", []),
                additional_metadata=g.get("metadata", {})
            )
            for g in goldens_data
        ]
        
        dataset = EvaluationDataset(goldens=goldens)
        self.datasets[name] = dataset
        
        return dataset
    
    def create_test_cases(self, dataset_name: str, llm_app_fn: callable) -> List[LLMTestCase]:
        """Create test cases by running LLM app on goldens"""
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        dataset = self.datasets[dataset_name]
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
                dataset.add_test_case(test_case)
                
            except Exception as e:
                print(f"Error creating test case: {e}")
                continue
        
        return test_cases
    
    def export_goldens_csv(self, dataset_name: str) -> Path:
        """Export goldens to CSV for manual review"""
        if dataset_name not in self.datasets:
            self.load_dataset(dataset_name)
        
        dataset = self.datasets[dataset_name]
        
        rows = []
        for golden in dataset.goldens:
            metadata = golden.additional_metadata or {}
            rows.append({
                "patient_id": metadata.get("patient_id", ""),
                "mrn": metadata.get("mrn", ""),
                "input": golden.input,
                "expected_output": golden.expected_output[:200],
                "context_count": len(golden.context) if golden.context else 0,
                "source": metadata.get("source", "")
            })
        
        df = pd.DataFrame(rows)
        output_path = self.goldens_dir / f"{dataset_name}_goldens.csv"
        df.to_csv(output_path, index=False)
        
        print(f"Goldens exported to CSV: {output_path}")
        return output_path


def create_fabrication_detection_dataset() -> EvaluationDataset:
    """Create specialized dataset for fabrication detection"""
    goldens = [
        Golden(
            input="Validate this summary for fabrications",
            expected_output="No fabrications detected",
            context=["Patient has 2.5 cm mass in left breast"],
            additional_metadata={
                "task": "fabrication_detection",
                "expected_fabrications": 0
            }
        ),
        Golden(
            input="Validate this summary for fabrications",
            expected_output="Fabrication detected: lymph node involvement not mentioned in source",
            context=["Patient has 2.5 cm mass in left breast"],
            additional_metadata={
                "task": "fabrication_detection",
                "expected_fabrications": 1,
                "fabricated_claim": "lymph node involvement"
            }
        )
    ]
    
    return EvaluationDataset(goldens=goldens)


if __name__ == "__main__":
    builder = DatasetBuilder()
    
    print("Building clinical summarization dataset...")
    dataset = builder.build_dataset("clinical_summarization")
    
    print(f"\nDataset created with {len(dataset.goldens)} goldens")
    
    builder.save_dataset("clinical_summarization", alias="Clinical-Summarization-V2")
    
    csv_path = builder.export_goldens_csv("clinical_summarization")
    print(f"\nCSV exported for manual review: {csv_path}")
    
    fab_dataset = create_fabrication_detection_dataset()
    print(f"\nFabrication detection dataset created with {len(fab_dataset.goldens)} goldens")
