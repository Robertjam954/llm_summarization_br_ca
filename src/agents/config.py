"""
Configuration file for Hierarchical Multi-Agent System
Google Cloud Agent Garden - Clinical Document Summarization
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    """LLM model configuration"""
    primary_model: str = "gpt-4o"
    evaluation_model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 8192
    
@dataclass
class AgentConfig:
    """Agent-specific configuration"""
    recursion_limit: int = 150
    max_iterations: int = 10
    timeout_seconds: int = 300
    
@dataclass
class MetricThresholds:
    """Evaluation metric thresholds"""
    answer_relevancy: float = 0.7
    faithfulness: float = 0.8
    contextual_precision: float = 0.7
    contextual_recall: float = 0.7
    correctness: float = 0.75
    rouge_l: float = 0.5
    bleu: float = 0.4
    bert_f1: float = 0.7
    task_completion: float = 0.8
    tool_correctness: float = 0.9
    
@dataclass
class RAGConfig:
    """RAG retrieval configuration"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    chunk_size: int = 1000
    chunk_overlap: int = 200
    embedding_model: str = "text-embedding-3-small"
    
@dataclass
class SafetyConfig:
    """Safety and bias detection configuration"""
    toxicity_threshold: float = 0.3
    bias_threshold: float = 0.5
    pii_detection_enabled: bool = True
    
@dataclass
class TracingConfig:
    """Tracing and monitoring configuration"""
    langsmith_enabled: bool = True
    deepeval_enabled: bool = True
    log_level: str = "INFO"
    trace_sampling_rate: float = 1.0
    
@dataclass
class DeploymentConfig:
    """Google Cloud deployment configuration"""
    project_id: str = os.getenv("GCP_PROJECT_ID", "")
    region: str = "us-central1"
    service_name: str = "clinical-summarization-agent"
    min_instances: int = 0
    max_instances: int = 10
    memory: str = "2Gi"
    cpu: str = "2"
    timeout: int = 300
    
@dataclass
class DataConfig:
    """Data paths and storage configuration"""
    data_dir: Path = Path(r"C:\Users\jamesr4\loc\data_private")
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "v2_prompt_test_parsed"
    deidentified_dir: Path = data_dir / "v2_prompt_test_deidentified"
    goldens_dir: Path = data_dir / "goldens"
    output_dir: Path = data_dir / "agent_outputs"
    
    def __post_init__(self):
        for dir_path in [self.goldens_dir, self.output_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

@dataclass
class SystemConfig:
    """Main system configuration"""
    model: ModelConfig = field(default_factory=ModelConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    metrics: MetricThresholds = field(default_factory=MetricThresholds)
    rag: RAGConfig = field(default_factory=RAGConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    tracing: TracingConfig = field(default_factory=TracingConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def validate(self) -> bool:
        """Validate configuration"""
        required_env_vars = ["OPENAI_API_KEY"]
        
        if self.tracing.langsmith_enabled:
            required_env_vars.append("LANGCHAIN_API_KEY")
            
        if self.deployment.project_id:
            required_env_vars.append("GCP_PROJECT_ID")
            
        missing_vars = [var for var in required_env_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
            
        return True

CLINICAL_FEATURES = [
    "feature_1_lesion_size",
    "feature_2_lesion_location",
    "feature_3_calcifications_asymmetry",
    "feature_4_additional_enhancement_mri",
    "feature_5_extent",
    "feature_6_accurate_clip_placement",
    "feature_7_workup_recommendation",
    "feature_8_lymph_node",
    "feature_9_chronology_preserved",
    "feature_10_biopsy_method",
    "feature_11_invasive_component_size_pathology",
    "feature_12_histologic_diagnosis",
    "feature_13_receptor_ER",
    "feature_13_receptor_PR",
    "feature_13_receptor_HER2_IHC",
    "feature_13_receptor_HER2_ISH",
]

AGENT_ROLES = {
    "research_team": {
        "search": "Web search for clinical guidelines and evidence",
        "web_scraper": "Extract detailed information from medical sources"
    },
    "writing_team": {
        "doc_writer": "Generate structured clinical summaries",
        "note_taker": "Create outlines and organize information",
        "chart_generator": "Generate visualizations and data charts"
    },
    "validation_team": {
        "fabrication_detector": "Identify unsupported claims in summaries",
        "deidentifier": "Remove PHI from clinical text",
        "quality_checker": "Validate summary completeness and accuracy"
    }
}

METRIC_CATEGORIES = {
    "rag_retriever": ["contextual_relevancy", "contextual_precision", "contextual_recall"],
    "rag_generator": ["answer_relevancy", "faithfulness"],
    "summarization": ["correctness", "precision", "recall"],
    "text_based": ["rouge", "bleu", "exact_match", "bert", "faithfulness"],
    "agentic": ["task_completion", "argument_correctness", "tool_correctness"],
    "safety": ["toxicity", "bias", "pii_detection"]
}

def get_config() -> SystemConfig:
    """Get system configuration with environment variable overrides"""
    config = SystemConfig()
    
    if os.getenv("LANGSMITH_TRACING"):
        config.tracing.langsmith_enabled = os.getenv("LANGSMITH_TRACING").lower() == "true"
        
    if os.getenv("GCP_PROJECT_ID"):
        config.deployment.project_id = os.getenv("GCP_PROJECT_ID")
        
    if os.getenv("SERVICE_NAME"):
        config.deployment.service_name = os.getenv("SERVICE_NAME")
        
    config.validate()
    return config

if __name__ == "__main__":
    config = get_config()
    print("Configuration loaded successfully")
    print(f"Model: {config.model.primary_model}")
    print(f"Deployment: {config.deployment.service_name}")
    print(f"Data directory: {config.data.data_dir}")
