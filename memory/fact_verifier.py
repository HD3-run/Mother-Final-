"""
Fact Verification System for MOTHER

This module provides LLM-powered fact verification to ensure only relevant,
high-quality facts are stored in the knowledge graph. It verifies facts
before storage and reframes complex sentences into simple S-V-O format.
"""

import json
import logging
from typing import Optional

from processing.llm_handler import get_response

logger = logging.getLogger(__name__)


def verify_and_reframe_fact(
    original_topic: str,
    raw_sentence: str,
    config: dict | None = None,
) -> Optional[str]:
    """
    Uses the LLM to verify if a raw sentence is relevant to a topic and,
    if so, reframes it into a simple, atomic S-V-O sentence for learning.
    
    Args:
        original_topic: The topic the fact should be about (e.g., "pets", "photosynthesis").
        raw_sentence: The raw sentence to verify and potentially reframe.
        config: Configuration dictionary for LLM (optional).
        
    Returns:
        A simplified S-V-O sentence if relevant, None if irrelevant.
    """
    logger.info(f"[Fact Verifier]: Verifying fact for '{original_topic}'...")
    
    if config is None:
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except Exception:
            config = {"temperature": 0.0, "max_tokens": 64}
    
    system_prompt = (
        "You are a precise fact verification and reframing engine. Your task is to analyze a 'Raw Sentence' "
        "to see if it is a direct, useful fact about the 'Original Topic'. If it is, you MUST rephrase it into a "
        "single, simple, declarative sentence (Subject-Verb-Object). If it is not relevant, you MUST output ONLY the word 'None'."
    )
    
    examples_prompt = (
        "Here are some examples:\n"
        "Original Topic: fabric\n"
        "Raw Sentence: A textile is a flexible material made by creating an interlocking network of yarns or threads.\n"
        "Output: A fabric is a flexible material.\n\n"
        "Original Topic: history of bitcoin\n"
        "Raw Sentence: Bitcoin is a cryptocurrency, a digital asset that uses cryptography to control its creation and management.\n"
        "Output: None\n\n"
        "Original Topic: bees\n"
        "Raw Sentence: Bees are flying insects closely related to wasps and ants, known for their role in pollination.\n"
        "Output: Bees are flying insects."
    )
    
    full_prompt = (
        f"{system_prompt}\n\n{examples_prompt}\n\n"
        f"Original Topic: {original_topic}\nRaw Sentence: {raw_sentence}\n"
        f"Output:"
    )
    
    try:
        # Use MOTHER's LLM handler
        llm_config = config.copy()
        llm_config["temperature"] = 0.0
        llm_config["max_tokens"] = 64
        
        response = get_response(
            prompt=full_prompt,
            config=llm_config,
            system_context=system_prompt,
        )
        
        if not response:
            logger.warning("[Fact Verifier]: No response from LLM.")
            return None
        
        rephrased_fact = response.strip()
        
        if "none" in rephrased_fact.lower():
            logger.info("    - LLM rejected the fact as irrelevant.")
            return None
        
        logger.info(f"    - LLM verified and reframed: '{rephrased_fact}'")
        return rephrased_fact
        
    except Exception as e:
        logger.error(f"[Fact Verifier Error]: Could not process fact with LLM. Error: {e}")
        return None

