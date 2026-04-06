"""
llm_factory.py
Unified LLM factory for extraction agents.
Routes to ChatAnthropic (claude-*) or ChatOpenAI (gpt-*, o1*) based on model_id.
"""
from __future__ import annotations

import os
from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


def get_llm(
    model_id: str = "claude-3-5-sonnet-20241022",
    temperature: float = 0.0,
    max_tokens: int = 2048,
) -> Union[ChatAnthropic, ChatOpenAI]:
    """Return the appropriate LangChain chat model for the given model_id."""
    if model_id.startswith("claude"):
        return ChatAnthropic(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
    # OpenAI: gpt-*, o1*, o3*
    return ChatOpenAI(
        model=model_id,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
