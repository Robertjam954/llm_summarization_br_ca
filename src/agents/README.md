# Hierarchical Multi-Agent Clinical Summarization System

**Google Cloud Agent Garden - Production Ready**

A sophisticated multi-agent system for clinical document summarization with comprehensive evaluation metrics, built on LangGraph and optimized for Google Cloud deployment.

## 🏗️ Architecture

### Hierarchical Agent Teams

```
Top-Level Supervisor
├── Research Team
│   ├── Search Agent (Tavily)
│   └── Web Scraper Agent
└── Summarization Team
    ├── Feature Extractor Agent
    ├── Validator Agent (Fabrication Detection)
    └── Deidentifier Agent (PHI Removal)
```

### Key Features

- **Hierarchical Orchestration**: Multi-level supervision for complex task decomposition
- **Comprehensive Metrics**: RAG, summarization, text-based, agentic, and safety metrics
- **Production Monitoring**: Built-in tracing with DeepEval and LangSmith
- **Google Cloud Ready**: Optimized for Cloud Run and Vertex AI Agent Engine
- **A2A Compatible**: Supports Agent2Agent protocol for interoperability

## 📊 Evaluation Metrics

### RAG Metrics
- **Retriever**: Contextual Relevancy, Precision, Recall
- **Generator**: Answer Relevancy, Faithfulness

### Summarization Metrics
- Correctness
- Precision (fabrication detection)
- Recall (completeness)

### Text-Based Metrics
- ROUGE-L
- BLEU (1-4)
- BERTScore
- Exact Match

### Agentic Metrics
- Task Completion
- Tool Correctness
- Argument Correctness

### Safety Metrics
- Toxicity Detection
- Bias Detection
- PII Detection

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-key"
export LANGCHAIN_API_KEY="your-key"  # Optional for tracing

# Run the system
python main.py

# Run with evaluation
RUN_EVALUATION=true python main.py
```

### Build Dataset

```python
from dataset_builder import DatasetBuilder

builder = DatasetBuilder()
dataset = builder.build_dataset("clinical_summarization")
builder.save_dataset("clinical_summarization", alias="Clinical-V2")
```

### Run Evaluation

```python
from main import ClinicalSummarizationPipeline

pipeline = ClinicalSummarizationPipeline()
results = pipeline.run_end_to_end_evaluation("clinical_summarization")
```

## 🐳 Docker Deployment

### Build Container

```bash
docker build -t clinical-summarization-agent .
```

### Run Locally

```bash
docker run -p 8080:8080 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  clinical-summarization-agent
```

## ☁️ Google Cloud Deployment

### Deploy to Cloud Run

```bash
# Build and deploy in one command
gcloud run deploy clinical-summarization-agent \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY
```

### Deploy to Vertex AI Agent Engine

```python
from google.cloud import aiplatform

aiplatform.init(project="your-project-id", location="us-central1")

# Deploy containerized agent
endpoint = aiplatform.Endpoint.create(display_name="clinical-summarization")
endpoint.deploy(
    model=model,
    deployed_model_display_name="clinical-agent-v1",
    machine_type="n1-standard-4"
)
```

## 📁 Project Structure

```
src/agents/
├── config.py                    # Configuration management
├── evaluation_metrics.py        # All evaluation metrics
├── hierarchical_agents.py       # Multi-agent orchestration
├── dataset_builder.py           # Golden dataset creation
├── main.py                      # Main orchestration script
├── Dockerfile                   # Container definition
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🔧 Configuration

Edit `config.py` to customize:

- **Models**: Primary and evaluation LLM models
- **Thresholds**: Metric passing thresholds
- **RAG Settings**: Retrieval parameters
- **Deployment**: Cloud Run/Vertex AI settings
- **Data Paths**: Input/output directories

## 📈 Monitoring & Tracing

### LangSmith Integration

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your-key"
export LANGCHAIN_PROJECT="clinical-summarization"
```

### DeepEval Dashboard

```bash
deepeval login
deepeval test run test_agents.py
```

### Cloud Monitoring

- Request count and latency
- Error rates
- Custom metrics (fabrication rate, feature extraction accuracy)
- Distributed tracing with Cloud Trace

## 🧪 Testing

### Unit Tests

```bash
pytest tests/test_agents.py
pytest tests/test_metrics.py
```

### Integration Tests

```bash
pytest tests/test_integration.py --run-slow
```

### Evaluation Tests

```bash
deepeval test run tests/test_evaluation.py
```

## 📊 Metrics Dashboard

Access real-time metrics:

- **DeepEval**: https://app.confident-ai.com
- **LangSmith**: https://smith.langchain.com
- **Cloud Console**: https://console.cloud.google.com/run

## 🔐 Security

- **PHI Removal**: Automatic deidentification of clinical text
- **VPC Support**: Deploy within private networks
- **IAM Integration**: Google Cloud identity and access management
- **Secrets Management**: Environment-based configuration

## 🤝 A2A Protocol Support

This agent is compatible with the Agent2Agent (A2A) protocol:

```python
# Discover agent capabilities
agent_card = system.get_agent_card()

# Initiate cross-agent task
response = system.invoke_a2a(
    target_agent="risk-assessment-agent",
    task="Assess clinical risk for patient"
)
```

## 📝 Clinical Features Extracted

1. Lesion Size
2. Lesion Location
3. Calcifications/Asymmetry
4. Additional Enhancement (MRI)
5. Disease Extent
6. Clip Placement Accuracy
7. Workup Recommendations
8. Lymph Node Status
9. Chronology Preservation
10. Biopsy Method
11. Invasive Component Size
12. Histologic Diagnosis
13. Receptor Status (ER, PR, HER2)

## 🎯 Use Cases

- **Clinical Research**: Automated feature extraction from pathology reports
- **Quality Assurance**: Fabrication detection in AI-generated summaries
- **Data Standardization**: Structured extraction from unstructured documents
- **Multi-center Studies**: Consistent feature extraction across institutions

## 📚 References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [DeepEval Metrics](https://docs.confident-ai.com/)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [Vertex AI Agent Builder](https://cloud.google.com/vertex-ai/docs/agent-builder)
- [A2A Protocol](https://github.com/google/a2a-protocol)

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'deepeval'`
```bash
pip install deepeval
```

**Issue**: API rate limits
```python
# Adjust in config.py
config.agent.max_iterations = 5
config.agent.timeout_seconds = 60
```

**Issue**: Memory errors with large documents
```python
# Adjust RAG chunk size
config.rag.chunk_size = 500
config.rag.chunk_overlap = 100
```

## 📞 Support

For issues or questions:
- Create an issue in the repository
- Check the [LangGraph Discord](https://discord.gg/langchain)
- Review [DeepEval Documentation](https://docs.confident-ai.com/)

## 📄 License

This project is part of the clinical research framework at Memorial Sloan Kettering Cancer Center.

## 🙏 Acknowledgments

- LangChain team for LangGraph framework
- Confident AI for DeepEval metrics
- Google Cloud for Agent Garden platform
- Clinical research team for domain expertise

---

**Version**: 1.0.0  
**Last Updated**: 2026-03-05  
**Status**: Production Ready ✅
