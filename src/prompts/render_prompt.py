"""
render_prompt.py
Utility to render prompt templates from YAML library files.
Supports Jinja2-style variable substitution for dynamic prompts.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROMPT_LIBRARY_DIR = (
    Path(__file__).parent.parent.parent / "prompts" / "library"
)


def load_prompt_yaml(filename: str) -> Dict[str, Any]:
    path = PROMPT_LIBRARY_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def render_template(template: str, variables: Dict[str, Any]) -> str:
    for key, value in variables.items():
        template = template.replace("{{" + key + "}}", str(value))
    return template


def get_extraction_prompt_template(
    prompt_id: str,
    variables: Optional[Dict[str, Any]] = None,
) -> str:
    data = load_prompt_yaml("extraction_prompts.yaml")
    template = data.get(prompt_id, data.get("default", ""))
    if variables:
        return render_template(template, variables)
    return template


def get_verification_prompt_template(
    prompt_id: str = "default",
    variables: Optional[Dict[str, Any]] = None,
) -> str:
    data = load_prompt_yaml("verification_prompts.yaml")
    template = data.get(prompt_id, data.get("default", ""))
    if variables:
        return render_template(template, variables)
    return template


def get_rewrite_prompt_template(
    prompt_id: str = "default",
    variables: Optional[Dict[str, Any]] = None,
) -> str:
    data = load_prompt_yaml("rewrite_prompts.yaml")
    template = data.get(prompt_id, data.get("default", ""))
    if variables:
        return render_template(template, variables)
    return template


# =============================================================================
# MODULE SUMMARY
# =============================================================================
# File   : src/prompts/render_prompt.py
# Purpose: Loads YAML prompt library files and renders templates with
#          {{variable}} placeholder substitution.
#
# Functions:
#   load_prompt_yaml(filename) -> dict
#     Loads a YAML file from prompts/library/ and returns prompt_id -> str.
#
#   render_template(template, variables) -> str
#     Replaces {{key}} placeholders with values from the variables dict.
#
#   get_extraction_prompt_template(prompt_id, variables) -> str
#     Loads extraction_prompts.yaml and renders the named template.
#
#   get_verification_prompt_template(prompt_id, variables) -> str
#     Loads verification_prompts.yaml and renders the named template.
#
#   get_rewrite_prompt_template(prompt_id, variables) -> str
#     Loads rewrite_prompts.yaml and renders the named template.
#
# Consumed by:
#   src/prompts/extraction_prompt_builder.py
#   src/prompts/verification_prompt_builder.py
#   src/agents/rewrite_agent.py
# =============================================================================
