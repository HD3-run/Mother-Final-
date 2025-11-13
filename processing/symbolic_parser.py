"""
Symbolic Parser for MOTHER

This module provides deterministic parsing of user input to extract
structured information (facts, relationships, queries) without relying
solely on LLMs.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class SymbolicParser:
    """Deterministic parser for extracting structured information"""
    
    def __init__(self):
        """Initialize the symbolic parser"""
        self.patterns = self._initialize_patterns()
    
    def _initialize_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Initialize regex patterns for different types of information"""
        return {
            "name": [
                re.compile(r"my name is (\w+)", re.IGNORECASE),
                re.compile(r"i am (\w+)", re.IGNORECASE),
                re.compile(r"i'm (\w+)", re.IGNORECASE),
                re.compile(r"call me (\w+)", re.IGNORECASE),
            ],
            "location": [
                re.compile(r"i live in ([^,.!?]+)", re.IGNORECASE),
                re.compile(r"i am from ([^,.!?]+)", re.IGNORECASE),
                re.compile(r"i'm from ([^,.!?]+)", re.IGNORECASE),
                re.compile(r"located in ([^,.!?]+)", re.IGNORECASE),
            ],
            "age": [
                re.compile(r"i am (\d+) years? old", re.IGNORECASE),
                re.compile(r"i'm (\d+) years? old", re.IGNORECASE),
                re.compile(r"my age is (\d+)", re.IGNORECASE),
                re.compile(r"i am (\d+)", re.IGNORECASE),
            ],
            "pet": [
                re.compile(r"i have (?:a |an )?(\w+)(?: named (\w+))?", re.IGNORECASE),
                re.compile(r"my (\w+)'s name is (\w+)", re.IGNORECASE),
                re.compile(r"my (\w+) is named (\w+)", re.IGNORECASE),
                re.compile(r"i own (?:a |an )?(\w+)(?: named (\w+))?", re.IGNORECASE),
            ],
            "job": [
                re.compile(r"i work as (?:a |an )?([^,.!?]+)", re.IGNORECASE),
                re.compile(r"i am (?:a |an )?([^,.!?]+)", re.IGNORECASE),
                re.compile(r"my job is ([^,.!?]+)", re.IGNORECASE),
            ],
            "relationship": [
                re.compile(r"(\w+) is (?:a |an )?(\w+)", re.IGNORECASE),
                re.compile(r"(\w+) has (?:a |an )?(\w+)", re.IGNORECASE),
                re.compile(r"(\w+) (?:lives|lived) in ([^,.!?]+)", re.IGNORECASE),
            ],
        }
    
    def parse_fact(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parses a fact from text using pattern matching.
        
        Args:
            text: The text to parse.
            
        Returns:
            A dictionary with 'type', 'subject', 'value', 'object' keys, or None.
        """
        text = text.strip()
        
        # Try each pattern type
        for fact_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    groups = match.groups()
                    
                    if fact_type == "name":
                        return {
                            "type": "name",
                            "subject": "user",
                            "relation": "has_name",
                            "object": groups[0],
                        }
                    
                    elif fact_type == "location":
                        return {
                            "type": "location",
                            "subject": "user",
                            "relation": "lives_in",
                            "object": groups[0].strip(),
                        }
                    
                    elif fact_type == "age":
                        return {
                            "type": "age",
                            "subject": "user",
                            "relation": "has_age",
                            "object": groups[0],
                        }
                    
                    elif fact_type == "pet":
                        pet_type = groups[0]
                        pet_name = groups[1] if len(groups) > 1 else None
                        result = {
                            "type": "pet",
                            "subject": "user",
                            "relation": "has_pet",
                            "object": pet_type,
                        }
                        if pet_name:
                            result["pet_name"] = pet_name
                        return result
                    
                    elif fact_type == "job":
                        return {
                            "type": "job",
                            "subject": "user",
                            "relation": "works_as",
                            "object": groups[0].strip(),
                        }
                    
                    elif fact_type == "relationship":
                        return {
                            "type": "relationship",
                            "subject": groups[0],
                            "relation": "is_a",
                            "object": groups[1],
                        }
        
        return None
    
    def parse_query(self, text: str) -> Optional[Dict[str, str]]:
        """
        Parses a query from text to determine what information is being requested.
        
        Args:
            text: The query text.
            
        Returns:
            A dictionary with 'type', 'subject', 'relation' keys, or None.
        """
        text_lower = text.lower().strip()
        
        # Question patterns
        question_patterns = {
            "name": [
                (r"what is (?:my |your )?name", "user", "has_name"),
                (r"what's (?:my |your )?name", "user", "has_name"),
                (r"who am i", "user", "has_name"),
            ],
            "location": [
                (r"where (?:do i |does (?:the )?user )?live", "user", "lives_in"),
                (r"where (?:am i |is (?:the )?user )?from", "user", "is_from"),
            ],
            "age": [
                (r"how old (?:am i |is (?:the )?user)", "user", "has_age"),
                (r"what is (?:my |(?:the )?user'?s )?age", "user", "has_age"),
            ],
            "pet": [
                (r"what (?:are |is )?(?:my |(?:the )?user'?s )?pets?", "user", "has_pet"),
                (r"do i have (?:any )?pets?", "user", "has_pet"),
                (r"what (?:are |is )?(?:my |(?:the )?user'?s )?(?:pet )?names?", "user", "has_pet_name"),
            ],
            "general": [
                (r"what (?:is|are) (\w+)", None, "is_a"),
                (r"who (?:is|are) (\w+)", None, "is_a"),
            ],
        }
        
        for query_type, patterns in question_patterns.items():
            for pattern, subject, relation in patterns:
                match = re.search(pattern, text_lower)
                if match:
                    result = {
                        "type": query_type,
                        "relation": relation,
                    }
                    if subject:
                        result["subject"] = subject
                    elif match.groups():
                        result["subject"] = match.group(1)
                    else:
                        result["subject"] = "unknown"
                    
                    return result
        
        return None
    
    def extract_entities(self, text: str) -> List[str]:
        """
        Extracts potential entity names from text (simple heuristic).
        
        Args:
            text: The text to extract entities from.
            
        Returns:
            A list of potential entity names.
        """
        # Capitalized words (potential proper nouns)
        entities = re.findall(r'\b[A-Z][a-z]+\b', text)
        return list(set(entities))  # Remove duplicates
    
    def is_statement(self, text: str) -> bool:
        """
        Determines if text is a statement (providing information) vs a question.
        
        Args:
            text: The text to analyze.
            
        Returns:
            True if it appears to be a statement, False if a question.
        """
        text_lower = text.lower().strip()
        
        # Question indicators
        question_words = ["what", "who", "where", "when", "why", "how", "which"]
        if any(text_lower.startswith(qw) for qw in question_words):
            return False
        
        if text_lower.endswith("?"):
            return False
        
        # Statement indicators
        statement_patterns = [
            r"^my \w+",
            r"^i (?:am|have|like|enjoy|work|live)",
            r"^i'm",
        ]
        
        if any(re.match(pattern, text_lower) for pattern in statement_patterns):
            return True
        
        # Default: assume statement if no question markers
        return not text_lower.endswith("?")

