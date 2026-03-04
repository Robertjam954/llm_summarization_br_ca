# DeepEval Metrics Catalog
**Memorial Sloan Kettering | Goel Lab**  
**Date:** March 2026  
**Source:** Local DeepEval repository (`C:\Users\jamesr4\local git repos\deepeval`)

---

## Overview

This document catalogs all available DeepEval metrics from the local repository. Metrics are organized by category for easy selection and implementation in our LLM validation pipeline.

---

## Metric Categories

### 1. Core Evaluation Metrics

#### **GEval**
- **Module:** `deepeval.metrics.g_eval.g_eval`
- **Class:** `GEval`
- **Description:** General-purpose LLM-as-a-judge metric for custom evaluation criteria
- **Use Case:** Custom evaluation criteria (e.g., clinical accuracy, completeness)
- **Required Params:** Custom evaluation criteria, input, actual_output
- **LLM-based:** Yes

#### **DAGMetric** ⭐
- **Module:** `deepeval.metrics.dag.dag`
- **Class:** `DAGMetric`, `DeepAcyclicGraph`
- **Description:** Deep Acyclic Graph metric for complex multi-step evaluation with decision trees
- **Use Case:** Multi-step validation workflows, conditional evaluation logic
- **Required Params:** DAG structure with nodes, input, actual_output
- **LLM-based:** Yes (configurable per node)
- **Special Features:**
  - Supports decision tree logic
  - Exportable graph structure
  - Async execution
  - Custom node types (BinaryJudgementNode, NonBinaryJudgementNode)

#### **ArenaGEval**
- **Module:** `deepeval.metrics.arena_g_eval.arena_g_eval`
- **Class:** `ArenaGEval`
- **Description:** Arena-style comparison between multiple model outputs
- **Use Case:** A/B testing different LLM models or prompts
- **Required Params:** Multiple outputs to compare
- **LLM-based:** Yes

---

### 2. RAG (Retrieval-Augmented Generation) Metrics

#### **AnswerRelevancyMetric**
- **Module:** `deepeval.metrics.answer_relevancy.answer_relevancy`
- **Class:** `AnswerRelevancyMetric`
- **Description:** Measures how relevant the answer is to the input question
- **Use Case:** Validate LLM answers are on-topic
- **Required Params:** input, actual_output
- **LLM-based:** Yes

#### **FaithfulnessMetric** ⭐
- **Module:** `deepeval.metrics.faithfulness.faithfulness`
- **Class:** `FaithfulnessMetric`
- **Description:** Measures if the answer is faithful to the retrieval context (no hallucination)
- **Use Case:** Detect fabrications, ensure answers are grounded in source documents
- **Required Params:** actual_output, retrieval_context
- **LLM-based:** Yes

#### **ContextualRecallMetric** ⭐
- **Module:** `deepeval.metrics.contextual_recall.contextual_recall`
- **Class:** `ContextualRecallMetric`
- **Description:** Measures if all relevant information from expected output is present in retrieval context
- **Use Case:** Ensure retrieval is comprehensive
- **Required Params:** expected_output, retrieval_context
- **LLM-based:** Yes

#### **ContextualPrecisionMetric**
- **Module:** `deepeval.metrics.contextual_precision.contextual_precision`
- **Class:** `ContextualPrecisionMetric`
- **Description:** Measures if relevant chunks are ranked higher than irrelevant chunks
- **Use Case:** Evaluate retrieval ranking quality
- **Required Params:** input, actual_output, retrieval_context, expected_output
- **LLM-based:** Yes

#### **ContextualRelevancyMetric**
- **Module:** `deepeval.metrics.contextual_relevancy.contextual_relevancy`
- **Class:** `ContextualRelevancyMetric`
- **Description:** Measures relevance of retrieval context to the input
- **Use Case:** Ensure retrieved chunks are relevant to query
- **Required Params:** input, retrieval_context
- **LLM-based:** Yes

---

### 3. Safety & Compliance Metrics ⭐

#### **PIILeakageMetric** ⭐⭐
- **Module:** `deepeval.metrics.pii_leakage.pii_leakage`
- **Class:** `PIILeakageMetric`
- **Description:** Detects PII (Personally Identifiable Information) leakage in outputs
- **Use Case:** **HCAT Safety - Validate deidentification, detect PHI in LLM outputs**
- **Required Params:** actual_output
- **LLM-based:** Yes
- **HIPAA Relevant:** Yes

#### **ToxicityMetric** ⭐
- **Module:** `deepeval.metrics.toxicity.toxicity`
- **Class:** `ToxicityMetric`
- **Description:** Detects toxic, harmful, or offensive content
- **Use Case:** **HCAT Safety - Ensure LLM outputs are safe for clinical use**
- **Required Params:** actual_output
- **LLM-based:** No (uses local NLP model)

#### **NonAdviceMetric** ⭐
- **Module:** `deepeval.metrics.non_advice.non_advice`
- **Class:** `NonAdviceMetric`
- **Description:** Ensures LLM does not provide medical advice when it shouldn't
- **Use Case:** **HCAT Safety - Prevent unauthorized medical advice**
- **Required Params:** actual_output
- **LLM-based:** Yes

#### **MisuseMetric** ⭐
- **Module:** `deepeval.metrics.misuse.misuse`
- **Class:** `MisuseMetric`
- **Description:** Detects potential misuse of the system
- **Use Case:** **HCAT Safety - Identify inappropriate system usage**
- **Required Params:** actual_output
- **LLM-based:** Yes

#### **RoleViolationMetric** ⭐
- **Module:** `deepeval.metrics.role_violation.role_violation`
- **Class:** `RoleViolationMetric`
- **Description:** Detects when LLM violates its assigned role
- **Use Case:** **HCAT Safety - Ensure LLM stays within clinical validation role**
- **Required Params:** actual_output, expected_role
- **LLM-based:** Yes

#### **RoleAdherenceMetric** ⭐
- **Module:** `deepeval.metrics.role_adherence.role_adherence`
- **Class:** `RoleAdherenceMetric`
- **Description:** Measures how well LLM adheres to its assigned role
- **Use Case:** **HCAT Safety - Validate LLM follows clinical validation guidelines**
- **Required Params:** actual_output, expected_role
- **LLM-based:** Yes

---

### 4. Content Quality Metrics

#### **HallucinationMetric** ⭐
- **Module:** `deepeval.metrics.hallucination.hallucination`
- **Class:** `HallucinationMetric`
- **Description:** Detects hallucinations (fabricated information) in outputs
- **Use Case:** **Primary metric for fabrication detection**
- **Required Params:** actual_output, context
- **LLM-based:** Yes

#### **BiasMetric**
- **Module:** `deepeval.metrics.bias.bias`
- **Class:** `BiasMetric`
- **Description:** Detects bias in LLM outputs (gender, race, etc.)
- **Use Case:** Ensure fairness (excluded per user request)
- **Required Params:** actual_output
- **LLM-based:** Yes
- **Status:** **EXCLUDED - User requested no bias metrics**

#### **SummarizationMetric**
- **Module:** `deepeval.metrics.summarization.summarization`
- **Class:** `SummarizationMetric`
- **Description:** Evaluates quality of summarization
- **Use Case:** If we summarize clinical reports
- **Required Params:** input, actual_output
- **LLM-based:** Yes

---

### 5. Task-Specific Metrics

#### **TaskCompletionMetric** ⭐
- **Module:** `deepeval.metrics.task_completion.task_completion`
- **Class:** `TaskCompletionMetric`
- **Description:** Measures if the task was completed successfully
- **Use Case:** **Validate extraction tasks completed correctly**
- **Required Params:** input, actual_output
- **LLM-based:** Yes

#### **ToolCorrectnessMetric**
- **Module:** `deepeval.metrics.tool_correctness.tool_correctness`
- **Class:** `ToolCorrectnessMetric`
- **Description:** Evaluates correctness of tool usage in agentic workflows
- **Use Case:** If using tools in LangGraph agent
- **Required Params:** input, actual_output, tools_used
- **LLM-based:** Yes

#### **JsonCorrectnessMetric**
- **Module:** `deepeval.metrics.json_correctness.json_correctness`
- **Class:** `JsonCorrectnessMetric`
- **Description:** Validates JSON output correctness
- **Use Case:** If LLM outputs structured JSON
- **Required Params:** actual_output, expected_schema
- **LLM-based:** No

#### **PromptAlignmentMetric**
- **Module:** `deepeval.metrics.prompt_alignment.prompt_alignment`
- **Class:** `PromptAlignmentMetric`
- **Description:** Measures alignment between prompt and output
- **Use Case:** Validate LLM follows prompt instructions
- **Required Params:** input, actual_output
- **LLM-based:** Yes

#### **ArgumentCorrectnessMetric**
- **Module:** `deepeval.metrics.argument_correctness.argument_correctness`
- **Class:** `ArgumentCorrectnessMetric`
- **Description:** Evaluates correctness of arguments in reasoning
- **Use Case:** Validate logical reasoning in clinical context
- **Required Params:** actual_output, expected_arguments
- **LLM-based:** Yes

#### **KnowledgeRetentionMetric**
- **Module:** `deepeval.metrics.knowledge_retention.knowledge_retention`
- **Class:** `KnowledgeRetentionMetric`
- **Description:** Measures knowledge retention across conversation turns
- **Use Case:** Multi-turn validation conversations
- **Required Params:** Conversational test case
- **LLM-based:** Yes

---

### 6. Agentic Workflow Metrics

#### **TopicAdherenceMetric**
- **Module:** `deepeval.metrics.topic_adherence.topic_adherence`
- **Class:** `TopicAdherenceMetric`
- **Description:** Measures if agent stays on topic
- **Use Case:** Ensure agent focuses on clinical validation
- **Required Params:** input, actual_output, expected_topic
- **LLM-based:** Yes

#### **StepEfficiencyMetric**
- **Module:** `deepeval.metrics.step_efficiency.step_efficiency`
- **Class:** `StepEfficiencyMetric`
- **Description:** Measures efficiency of agent steps
- **Use Case:** Optimize agent workflow
- **Required Params:** steps_taken, optimal_steps
- **LLM-based:** Yes

#### **PlanAdherenceMetric**
- **Module:** `deepeval.metrics.plan_adherence.plan_adherence`
- **Class:** `PlanAdherenceMetric`
- **Description:** Measures if agent follows its plan
- **Use Case:** Validate agent execution
- **Required Params:** plan, actual_steps
- **LLM-based:** Yes

#### **PlanQualityMetric**
- **Module:** `deepeval.metrics.plan_quality.plan_quality`
- **Class:** `PlanQualityMetric`
- **Description:** Evaluates quality of agent's plan
- **Use Case:** Assess agent planning capability
- **Required Params:** plan, goal
- **LLM-based:** Yes

#### **ToolUseMetric**
- **Module:** `deepeval.metrics.tool_use.tool_use`
- **Class:** `ToolUseMetric`
- **Description:** Evaluates appropriateness of tool usage
- **Use Case:** Validate agent tool selection
- **Required Params:** tools_used, available_tools
- **LLM-based:** Yes

#### **GoalAccuracyMetric**
- **Module:** `deepeval.metrics.goal_accuracy.goal_accuracy`
- **Class:** `GoalAccuracyMetric`
- **Description:** Measures if agent achieves its goal
- **Use Case:** High-level agent success metric
- **Required Params:** goal, actual_output
- **LLM-based:** Yes

---

### 7. Conversational Metrics

#### **TurnRelevancyMetric**
- **Module:** `deepeval.metrics.turn_relevancy.turn_relevancy`
- **Class:** `TurnRelevancyMetric`
- **Description:** Measures relevancy of each conversation turn
- **Use Case:** Multi-turn validation conversations
- **Required Params:** Conversational test case
- **LLM-based:** Yes

#### **TurnFaithfulnessMetric**
- **Module:** `deepeval.metrics.turn_faithfulness.turn_faithfulness`
- **Class:** `TurnFaithfulnessMetric`
- **Description:** Measures faithfulness per conversation turn
- **Use Case:** Multi-turn fabrication detection
- **Required Params:** Conversational test case
- **LLM-based:** Yes

#### **TurnContextualPrecisionMetric**
- **Module:** `deepeval.metrics.turn_contextual_precision.turn_contextual_precision`
- **Class:** `TurnContextualPrecisionMetric`
- **Description:** Contextual precision per turn
- **Use Case:** Multi-turn retrieval quality
- **Required Params:** Conversational test case
- **LLM-based:** Yes

#### **TurnContextualRecallMetric**
- **Module:** `deepeval.metrics.turn_contextual_recall.turn_contextual_recall`
- **Class:** `TurnContextualRecallMetric`
- **Description:** Contextual recall per turn
- **Use Case:** Multi-turn retrieval completeness
- **Required Params:** Conversational test case
- **LLM-based:** Yes

#### **TurnContextualRelevancyMetric**
- **Module:** `deepeval.metrics.turn_contextual_relevancy.turn_contextual_relevancy`
- **Class:** `TurnContextualRelevancyMetric`
- **Description:** Contextual relevancy per turn
- **Use Case:** Multi-turn retrieval relevance
- **Required Params:** Conversational test case
- **LLM-based:** Yes

#### **ConversationCompletenessMetric**
- **Module:** `deepeval.metrics.conversation_completeness.conversation_completeness`
- **Class:** `ConversationCompletenessMetric`
- **Description:** Measures if conversation covers all necessary topics
- **Use Case:** Ensure comprehensive validation
- **Required Params:** Conversational test case
- **LLM-based:** Yes

#### **ConversationalGEval**
- **Module:** `deepeval.metrics.conversational_g_eval.conversational_g_eval`
- **Class:** `ConversationalGEval`
- **Description:** G-Eval for conversational contexts
- **Use Case:** Custom conversational evaluation
- **Required Params:** Conversational test case, custom criteria
- **LLM-based:** Yes

#### **ConversationalDAGMetric**
- **Module:** `deepeval.metrics.conversational_dag.conversational_dag`
- **Class:** `ConversationalDAGMetric`
- **Description:** DAG metric for conversational contexts
- **Use Case:** Complex multi-turn decision trees
- **Required Params:** Conversational test case, DAG structure
- **LLM-based:** Yes

---

### 8. MCP (Model Context Protocol) Metrics

#### **MCPTaskCompletionMetric**
- **Module:** `deepeval.metrics.mcp.mcp_task_completion`
- **Class:** `MCPTaskCompletionMetric`
- **Description:** Task completion for MCP-based systems
- **Use Case:** If using MCP protocol
- **Required Params:** MCP-specific params
- **LLM-based:** Yes

#### **MultiTurnMCPUseMetric**
- **Module:** `deepeval.metrics.mcp.multi_turn_mcp_use_metric`
- **Class:** `MultiTurnMCPUseMetric`
- **Description:** Multi-turn MCP usage evaluation
- **Use Case:** If using MCP protocol
- **Required Params:** MCP-specific params
- **LLM-based:** Yes

#### **MCPUseMetric**
- **Module:** `deepeval.metrics.mcp_use_metric.mcp_use_metric`
- **Class:** `MCPUseMetric`
- **Description:** MCP usage evaluation
- **Use Case:** If using MCP protocol
- **Required Params:** MCP-specific params
- **LLM-based:** Yes

---

### 9. Multimodal Metrics

#### **TextToImageMetric**
- **Module:** `deepeval.metrics.multimodal_metrics`
- **Class:** `TextToImageMetric`
- **Description:** Evaluates text-to-image generation
- **Use Case:** Not applicable for our project
- **Required Params:** text, image
- **LLM-based:** Yes

#### **ImageEditingMetric**
- **Module:** `deepeval.metrics.multimodal_metrics`
- **Class:** `ImageEditingMetric`
- **Description:** Evaluates image editing quality
- **Use Case:** Not applicable for our project
- **Required Params:** original_image, edited_image
- **LLM-based:** Yes

#### **ImageCoherenceMetric**
- **Module:** `deepeval.metrics.multimodal_metrics`
- **Class:** `ImageCoherenceMetric`
- **Description:** Evaluates image coherence
- **Use Case:** Not applicable for our project
- **Required Params:** image
- **LLM-based:** Yes

#### **ImageHelpfulnessMetric**
- **Module:** `deepeval.metrics.multimodal_metrics`
- **Class:** `ImageHelpfulnessMetric`
- **Description:** Evaluates image helpfulness
- **Use Case:** Not applicable for our project
- **Required Params:** image, context
- **LLM-based:** Yes

#### **ImageReferenceMetric**
- **Module:** `deepeval.metrics.multimodal_metrics`
- **Class:** `ImageReferenceMetric`
- **Description:** Evaluates image against reference
- **Use Case:** Not applicable for our project
- **Required Params:** image, reference_image
- **LLM-based:** Yes

---

### 10. Non-LLM Metrics

#### **ExactMatchMetric**
- **Module:** `deepeval.metrics.exact_match.exact_match`
- **Class:** `ExactMatchMetric`
- **Description:** Exact string matching
- **Use Case:** Simple validation (e.g., case_id matching)
- **Required Params:** actual_output, expected_output
- **LLM-based:** No

#### **PatternMatchMetric**
- **Module:** `deepeval.metrics.pattern_match.pattern_match`
- **Class:** `PatternMatchMetric`
- **Description:** Regex pattern matching
- **Use Case:** Validate output format (e.g., date formats, numeric ranges)
- **Required Params:** actual_output, pattern
- **LLM-based:** No

---

## Recommended Metrics for Our Project

### **Primary Metrics (Must Implement)**

1. **DAGMetric** ⭐⭐⭐
   - Use for complex validation decision tree
   - Export DAG structure for visualization
   - Multi-step conditional evaluation

2. **FaithfulnessMetric** ⭐⭐⭐
   - Core fabrication detection
   - Validates LLM outputs against source documents

3. **HallucinationMetric** ⭐⭐⭐
   - Secondary fabrication detection
   - Complements FaithfulnessMetric

4. **PIILeakageMetric** ⭐⭐⭐
   - HCAT Safety: Validate deidentification
   - Detect residual PHI in outputs

5. **ContextualRecallMetric** ⭐⭐
   - Ensure retrieval is comprehensive
   - Validate all relevant evidence is retrieved

6. **TaskCompletionMetric** ⭐⭐
   - Validate extraction tasks completed
   - High-level success metric

### **Secondary Metrics (Should Implement)**

7. **ToxicityMetric** ⭐
   - HCAT Safety: Ensure safe outputs

8. **NonAdviceMetric** ⭐
   - HCAT Safety: Prevent unauthorized medical advice

9. **RoleAdherenceMetric** ⭐
   - HCAT Safety: Validate LLM stays in role

10. **AnswerRelevancyMetric** ⭐
    - Ensure answers are on-topic

11. **ContextualRelevancyMetric** ⭐
    - Validate retrieval relevance

### **Optional Metrics (Nice to Have)**

12. **PromptAlignmentMetric**
    - Validate prompt following

13. **ArgumentCorrectnessMetric**
    - Validate logical reasoning

14. **ExactMatchMetric**
    - Simple validation checks

15. **PatternMatchMetric**
    - Format validation

---

## Metrics to Exclude

- **BiasMetric** — Excluded per user request
- **Multimodal Metrics** — Not applicable (text-only project)
- **MCP Metrics** — Not using MCP protocol
- **Conversational Metrics** — Not using multi-turn conversations (yet)

---

## Next Steps

**Please select which metrics to implement:**

```
Primary (recommend all 6):
- [ ] DAGMetric
- [ ] FaithfulnessMetric
- [ ] HallucinationMetric
- [ ] PIILeakageMetric
- [ ] ContextualRecallMetric
- [ ] TaskCompletionMetric

Secondary (select 2-4):
- [ ] ToxicityMetric
- [ ] NonAdviceMetric
- [ ] RoleAdherenceMetric
- [ ] AnswerRelevancyMetric
- [ ] ContextualRelevancyMetric

Optional (select 0-3):
- [ ] PromptAlignmentMetric
- [ ] ArgumentCorrectnessMetric
- [ ] ExactMatchMetric
- [ ] PatternMatchMetric
```

Once you confirm, I will:
1. Implement DeepEval integration in NB12
2. Create decision tree DAG metric for fabrication detection
3. Export DAG visualization after running
4. Update README.md with selected metrics summary
