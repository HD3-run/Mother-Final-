"""
Fact Decomposition System for MOTHER

This module breaks down complex sentences into atomic facts that can be
stored as relationships in the knowledge graph. It decomposes definitions
and complex statements into simple S-V-O triples.
"""

import json
import logging
import re
from typing import List, Dict, Optional, Tuple

from processing.llm_handler import get_response

logger = logging.getLogger(__name__)


def decompose_sentence_to_relations(
    sentence: str,
    config: dict | None = None,
) -> List[Dict[str, str]]:
    """
    Decomposes a sentence into a list of atomic S-V-O relations.
    
    Args:
        sentence: The sentence to decompose.
        config: Configuration dictionary for LLM (optional).
        
    Returns:
        A list of dictionaries, each with 'subject', 'verb', 'object' keys.
    """
    logger.info(f"[Fact Decomposer]: Decomposing sentence: '{sentence}'")
    
    if config is None:
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
        except Exception:
            config = {"temperature": 0.0, "max_tokens": 200}
    
    system_prompt = (
        "You are a fact decomposition engine. Break down the given sentence into "
        "atomic Subject-Verb-Object relations. Output ONLY a JSON array of objects, "
        "each with 'subject', 'verb', 'object' keys. If the sentence cannot be decomposed, return an empty array []."
    )
    
    examples_prompt = (
        "Examples:\n"
        "Sentence: 'Bees are flying insects that pollinate flowers.'\n"
        "Output: [{\"subject\": \"bees\", \"verb\": \"are\", \"object\": \"flying insects\"}, "
        "{\"subject\": \"bees\", \"verb\": \"pollinate\", \"object\": \"flowers\"}]\n\n"
        "Sentence: 'The user has two rabbits named Choco and Bhuto.'\n"
        "Output: [{\"subject\": \"user\", \"verb\": \"has\", \"object\": \"rabbits\"}, "
        "{\"subject\": \"rabbit\", \"verb\": \"named\", \"object\": \"choco\"}, "
        "{\"subject\": \"rabbit\", \"verb\": \"named\", \"object\": \"bhuto\"}]\n"
    )
    
    full_prompt = f"{system_prompt}\n\n{examples_prompt}\n\nSentence: {sentence}\nOutput:"
    
    try:
        llm_config = config.copy()
        llm_config["temperature"] = 0.0
        llm_config["max_tokens"] = 200
        
        response = get_response(
            prompt=full_prompt,
            config=llm_config,
            system_context=system_prompt,
        )
        
        if not response:
            logger.warning("[Fact Decomposer]: No response from LLM.")
            return []
        
        # Extract JSON from response
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            relations = json.loads(json_match.group())
            logger.info(f"    - Decomposed into {len(relations)} relations")
            return relations
        else:
            logger.warning("[Fact Decomposer]: No JSON array found in response.")
            return []
            
    except json.JSONDecodeError as e:
        logger.error(f"[Fact Decomposer Error]: Failed to parse JSON: {e}")
        return []
    except Exception as e:
        logger.error(f"[Fact Decomposer Error]: Could not decompose sentence: {e}")
        return []


def break_down_definition(
    word: str,
    definition: str,
    config: dict | None = None,
) -> List[Dict[str, str]]:
    """
    Breaks down a dictionary-style definition into atomic relations.
    
    Args:
        word: The word being defined.
        definition: The definition text.
        config: Configuration dictionary for LLM (optional).
        
    Returns:
        A list of dictionaries with 'subject', 'verb', 'object' keys.
    """
    logger.info(f"[Fact Decomposer]: Breaking down definition for '{word}'")
    
    sentence = f"{word} is {definition}"
    return decompose_sentence_to_relations(sentence, config)


def extract_relations_from_text(
    text: str,
    config: dict | None = None,
) -> List[Dict[str, str]]:
    """
    Extracts all relations from a longer text by decomposing each sentence.
    
    Args:
        text: The text to extract relations from.
        config: Configuration dictionary for LLM (optional).
        
    Returns:
        A list of all extracted relations.
    """
    # Split into sentences (simple approach)
    sentences = re.split(r'[.!?]+', text)
    all_relations = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) > 10:  # Only process substantial sentences
            relations = decompose_sentence_to_relations(sentence, config)
            all_relations.extend(relations)
    
    logger.info(f"[Fact Decomposer]: Extracted {len(all_relations)} relations from text")
    return all_relations

