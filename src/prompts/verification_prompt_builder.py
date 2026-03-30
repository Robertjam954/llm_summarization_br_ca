"""
verification_prompt_builder.py
Builds the retrieval-augmented verification prompt.
Asks the model to find direct source support for an extracted claim.
"""

from typing import List

from langchain_core.messages import HumanMessage, SystemMessage

from src.workflows.extraction_state import Chunk

VERIFIER_SYSTEM = """You are a clinical evidence verification system.
Your only job is to determine whether a specific claim is directly
supported by the retrieved source chunks.
You do NOT extract new information. You do NOT infer."""

VERIFICATION_SCHEMA = """{
  "supported": true,
  "exact_support_quote": "",
  "page_ref": null,
  "support_strength": "direct",
  "verification_confidence": 1.0
}

OR if not supported:

{
  "supported": false,
  "reason": "",
  "verification_confidence": 0.0
}"""


def build_verification_prompt(
    feature_name: str,
    claimed_value: str,
    retrieved_chunks: List[Chunk],
) -> List:
    from src.rag.retrievers import format_chunks_for_prompt

    chunks_text = format_chunks_for_prompt(retrieved_chunks)

    human_content = f"""I claimed:
Feature: {feature_name}
Value: {claimed_value}

Using ONLY the retrieved chunks below, determine whether this claim
is DIRECTLY supported by explicit text.

Rules:
- SUPPORTED = exact or near-exact phrase present in source text.
- NOT_VERIFIABLE = no direct source statement found.
- Do not infer. Do not use general knowledge.

If SUPPORTED:
- return supported: true
- quote the exact supporting sentence
- provide page reference
- set verification_confidence = 1.0

If NOT_VERIFIABLE:
- return supported: false
- explain why
- set verification_confidence = 0.0

RETRIEVED CHUNKS:
{chunks_text}

OUTPUT SCHEMA (strict JSON only):
{VERIFICATION_SCHEMA}"""

    return [
        SystemMessage(content=VERIFIER_SYSTEM),
        HumanMessage(content=human_content),
    ]


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/prompts/verification_prompt_builder.py
# Purpose: Builds retrieval-augmented verification prompts that instruct the
#          LLM to find direct verbatim support for an extracted claim.
#          Core anti-fabrication mechanism of the pipeline.
#
# Functions:
#   build_verification_prompt(feature_name, claimed_value, chunks_text,
#                             prompt_id) -> List[BaseMessage]
#     Assembles a [SystemMessage, HumanMessage] list for the verifier LLM.
#     Loads template from prompts/library/verification_prompts.yaml.
#     Instructs: find exact quote, return supported=true/false,
#     verification_confidence 0.0-1.0, exact_support_quote, page_ref.
#
# Outputs:
#   List[BaseMessage] for LLM invocation.
#   LLM response should parse to VerificationResult schema.
#
# Consumed by:
#   src/agents/verify_agent.py
# =============================================================================
