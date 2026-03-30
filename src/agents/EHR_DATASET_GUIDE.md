# Using EHR Deidentification Test Dataset

Complete guide for using the EHR deidentification test.jsonl dataset with the hierarchical multi-agent evaluation system.

## 📁 Dataset Location

```
C:\Users\jamesr4\OneDrive - Memorial Sloan Kettering Cancer Center\Documents\GitHub\llm_summarization_br_ca\ehr_deidentification\data\ner_datasets\test.jsonl
```

## 📊 Dataset Structure

The test.jsonl file contains **75 samples** of clinical text with NER (Named Entity Recognition) labels for PHI detection.

### Sample Format

```json
{
  "tokens": ["Physician", "Discharge", "Summary", ...],
  "labels": ["O", "O", "O", ...],
  "current_sent_info": [
    {"text": "Physician", "start": 0, "end": 9, "label": "O"},
    ...
  ],
  "note_sent_info": {"start": 0, "end": 35, "note_id": "note_1"}
}
```

### Label Types

- **`O`**: Non-PHI (safe text)
- **`NA`**: Not applicable
- **PHI labels**: Various PHI entity types (names, dates, locations, etc.)

## 🚀 Quick Start

### 1. Explore the Dataset

```bash
cd src/agents
python run_ehr_evaluation.py --task explore
```

**Output:**
- Total samples count
- PHI entity statistics
- Note distribution
- Sample examples
- CSV summary export

### 2. Run Deidentification Evaluation

```bash
python run_ehr_evaluation.py --task deidentify
```

**What it does:**
- Loads 75 test samples
- Creates golden test cases for deidentification
- Runs hierarchical agent system on each case
- Evaluates with text-based and safety metrics
- Reports pass/fail rates

### 3. Run Summarization Evaluation

```bash
python run_ehr_evaluation.py --task summarize
```

**What it does:**
- Groups samples by note_id
- Creates clinical summary goldens
- Runs summarization agent
- Evaluates with summarization and text metrics

### 4. Run Complete Evaluation

```bash
python run_ehr_evaluation.py --task all
```

Runs all three tasks sequentially.

## 📝 Python API Usage

### Basic Dataset Loading

```python
from ehr_dataset_adapter import EHRDatasetAdapter

# Initialize adapter
adapter = EHRDatasetAdapter()

# Load samples
samples = adapter.load_samples()
print(f"Loaded {len(samples)} samples")

# Analyze a sample
sample = samples[0]
print(f"Text: {sample.get_text()}")
print(f"PHI entities: {sample.get_phi_entities()}")
print(f"Note ID: {sample.get_note_id()}")
```

### Create Deidentification Dataset

```python
from ehr_dataset_adapter import EHRDatasetAdapter

adapter = EHRDatasetAdapter()

# Build deidentification dataset
dataset = adapter.build_evaluation_dataset(task="deidentification")

# Save to JSON
adapter.save_dataset(dataset, "ehr_deidentification")

# Export summary CSV
adapter.export_summary_csv("ehr_test_dataset")
```

### Create Summarization Dataset

```python
adapter = EHRDatasetAdapter()

# Build summarization dataset
dataset = adapter.build_evaluation_dataset(task="summarization")

# Save
adapter.save_dataset(dataset, "ehr_summarization")
```

### Run Evaluation with Custom Agent

```python
from ehr_dataset_adapter import create_test_cases_from_ehr_data
from hierarchical_agents import HierarchicalAgentSystem
from evaluation_metrics import MetricEvaluator
import asyncio

# Initialize your agent
agent_system = HierarchicalAgentSystem()

# Define wrapper function
def my_deidentify_fn(input_text: str) -> str:
    result = agent_system.invoke(input_text)
    return result["messages"][-1].content

# Create test cases
test_cases = create_test_cases_from_ehr_data(
    llm_app_fn=my_deidentify_fn,
    task="deidentification"
)

# Evaluate
evaluator = MetricEvaluator()

for test_case in test_cases[:5]:  # First 5 cases
    results = asyncio.run(
        evaluator.evaluate_async(
            test_case=test_case,
            metric_categories=["text_based", "safety"]
        )
    )
    
    for result in results:
        print(f"{result.metric_name}: {result.score:.3f}")
```

## 📈 Evaluation Metrics

### Deidentification Task

**Metrics used:**
- **Text-based**: ROUGE, BLEU, BERTScore, Exact Match
- **Safety**: Toxicity detection, Bias detection

**Success criteria:**
- PHI entities correctly identified and removed
- Text remains clinically meaningful
- No fabricated information added

### Summarization Task

**Metrics used:**
- **Summarization**: Correctness, Precision, Recall
- **Text-based**: ROUGE, BLEU, BERTScore

**Success criteria:**
- Key clinical information preserved
- Concise and accurate summary
- No hallucinations or fabrications

## 🔧 Customization

### Custom PHI Detection

```python
from ehr_dataset_adapter import NERSample

sample = NERSample(
    tokens=["Patient", "John", "Doe"],
    labels=["O", "PHI", "PHI"],
    current_sent_info=[...],
    note_sent_info={...}
)

# Extract PHI
phi_entities = sample.get_phi_entities()
for entity in phi_entities:
    print(f"{entity['label']}: {entity['text']}")
```

### Custom Evaluation Metrics

```python
from evaluation_metrics import MetricEvaluator
from deepeval.test_case import LLMTestCase

evaluator = MetricEvaluator()

test_case = LLMTestCase(
    input="Deidentify: Patient John Doe, DOB 1/1/1980",
    actual_output="Patient [NAME], DOB [DATE]",
    expected_output="Patient [NAME], DOB [DATE]"
)

# Custom metric categories
results = asyncio.run(
    evaluator.evaluate_async(
        test_case=test_case,
        metric_categories=["text_based", "safety", "summarization"]
    )
)
```

## 📊 Expected Results

### Dataset Statistics

- **Total samples**: 75
- **Unique notes**: ~5-10 (grouped by note_id)
- **PHI entities**: Varies per sample
- **Average tokens per sample**: ~50-100

### Deidentification Goldens

- **Created**: ~60-70 goldens (excludes NA-only samples)
- **PHI types**: Names, dates, locations, phone numbers, etc.
- **Task**: Remove PHI while preserving clinical meaning

### Summarization Goldens

- **Created**: ~5-10 goldens (one per note)
- **Input**: Full clinical notes
- **Expected**: Key clinical information summary

## 🎯 Use Cases

### 1. PHI Detection Validation

Test your deidentification agent's ability to identify and remove PHI:

```python
adapter = EHRDatasetAdapter()
samples = adapter.load_samples()

for sample in samples:
    phi_entities = sample.get_phi_entities()
    if len(phi_entities) > 0:
        print(f"Found {len(phi_entities)} PHI entities")
        # Test your agent here
```

### 2. Clinical Summarization

Test summarization quality on real clinical notes:

```python
dataset = adapter.build_evaluation_dataset(task="summarization")

for golden in dataset.goldens:
    # Run your summarization agent
    summary = your_agent(golden.input)
    # Compare with expected output
    print(f"Expected: {golden.expected_output}")
    print(f"Actual: {summary}")
```

### 3. Multi-Agent Workflow Testing

Test the complete hierarchical agent system:

```python
from hierarchical_agents import HierarchicalAgentSystem

system = HierarchicalAgentSystem()

# Test on EHR data
result = system.invoke(
    "Extract clinical features and deidentify: Patient John Doe..."
)

print(result["messages"][-1].content)
```

## 📁 Output Files

After running evaluations, you'll find:

```
data_private/goldens/
├── ehr_deidentification_goldens.json    # Deidentification test cases
├── ehr_summarization_goldens.json       # Summarization test cases
├── ehr_test_dataset_summary.csv         # Dataset statistics
└── ehr_evaluation_results.json          # Evaluation results
```

## 🐛 Troubleshooting

### Issue: "File not found"

**Solution**: Update the path in `ehr_dataset_adapter.py`:

```python
self.test_path = Path(r"YOUR_ACTUAL_PATH\test.jsonl")
```

### Issue: "No PHI entities found"

**Solution**: Check label filtering in `get_phi_entities()`:

```python
# Labels that are NOT PHI
non_phi_labels = ["O", "NA"]
```

### Issue: "Too many test cases"

**Solution**: Limit the number of cases:

```python
test_cases = test_cases[:10]  # First 10 only
```

## 📚 Integration with Existing System

The EHR dataset adapter integrates seamlessly with:

- **`hierarchical_agents.py`**: Multi-agent orchestration
- **`evaluation_metrics.py`**: Comprehensive metrics
- **`dataset_builder.py`**: Clinical data goldens
- **`main.py`**: Production pipeline

### Combined Evaluation

```python
from main import ClinicalSummarizationPipeline
from ehr_dataset_adapter import EHRDatasetAdapter

# Your clinical pipeline
pipeline = ClinicalSummarizationPipeline()

# EHR test data
adapter = EHRDatasetAdapter()
dataset = adapter.build_evaluation_dataset(task="deidentification")

# Run evaluation
for golden in dataset.goldens[:5]:
    result = pipeline.agent_system.invoke(golden.input)
    print(f"Result: {result['messages'][-1].content[:100]}...")
```

## 🎓 Next Steps

1. **Explore the data**: Run `--task explore` to understand the dataset
2. **Test deidentification**: Run `--task deidentify` to validate PHI removal
3. **Test summarization**: Run `--task summarize` to validate clinical summaries
4. **Customize metrics**: Add your own evaluation criteria
5. **Deploy to production**: Use the validated agents in your pipeline

## 📞 Support

For issues or questions:
- Check the main `README.md` for system architecture
- Review `evaluation_metrics.py` for available metrics
- See `hierarchical_agents.py` for agent implementation

---

**Version**: 1.0.0  
**Dataset**: EHR Deidentification test.jsonl (75 samples)  
**Last Updated**: 2026-03-05
