"""
Lexicon Manager for MOTHER

This module tracks word knowledge and definitions, storing them in the
knowledge graph. It helps MOTHER understand and remember word meanings.
"""

import json
import logging
from typing import Dict, Optional, List

from memory.knowledge_graph import KnowledgeGraph, ConceptNode, RelationshipEdge

# Try to import dictionary utils for WordNet integration
try:
    from utils.dictionary_utils import get_word_info_from_wordnet, get_pos_tag_simple, lemmatize_word
    WORDNET_AVAILABLE = True
except ImportError:
    WORDNET_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[Lexicon Manager] WordNet utilities not available. Install NLTK for automatic word learning.")

logger = logging.getLogger(__name__)


class LexiconManager:
    """Manages word knowledge and definitions in the knowledge graph"""
    
    def __init__(self, graph: KnowledgeGraph):
        """Initialize the lexicon manager.
        
        Args:
            graph: The KnowledgeGraph instance to store word knowledge in.
        """
        self.graph = graph
    
    def add_word_definition(
        self,
        word: str,
        definition: str,
        word_type: str = "noun",
        confidence: float = 0.8,
        provenance: str = "dictionary",
    ) -> bool:
        """
        Adds a word definition to the knowledge graph.
        
        Args:
            word: The word being defined.
            definition: The definition text.
            word_type: The grammatical type (noun, verb, adjective, etc.).
            confidence: Confidence in the definition.
            provenance: Source of the definition.
            
        Returns:
            True if successfully added, False otherwise.
        """
        logger.info(f"[Lexicon Manager]: Adding definition for '{word}'")
        
        try:
            # Create or get word node
            word_node = ConceptNode(word, node_type="word")
            word_node = self.graph.add_node(word_node)
            
            # Create definition node
            definition_node = ConceptNode(
                f"{word}_definition",
                node_type="definition",
                properties={"text": definition, "word_type": word_type},
            )
            definition_node = self.graph.add_node(definition_node)
            
            # Create relationship: word --[has_definition]--> definition
            edge = self.graph.add_edge(
                word_node,
                definition_node,
                "has_definition",
                weight=confidence,
                properties={"provenance": provenance},
            )
            
            if edge:
                logger.info(f"    - Added definition for '{word}'")
                return True
            else:
                logger.warning(f"    - Failed to add definition edge")
                return False
                
        except Exception as e:
            logger.error(f"[Lexicon Manager Error]: Failed to add definition: {e}")
            return False
    
    def get_word_definition(self, word: str) -> Optional[str]:
        """
        Retrieves the definition of a word from the knowledge graph.
        
        Args:
            word: The word to look up.
            
        Returns:
            The definition text if found, None otherwise.
        """
        word_node = self.graph.get_node_by_name(word)
        if not word_node:
            return None
        
        # Find definition edges
        edges = self.graph.get_edges_from_node(word_node.id)
        for edge in edges:
            if edge.type == "has_definition":
                definition_node = self.graph.get_node_by_id(edge.target)
                if definition_node:
                    return definition_node.properties.get("text")
        
        return None
    
    def word_is_known(self, word: str) -> bool:
        """
        Checks if a word is known (exists in the knowledge graph as a word node
        or has a definition, or is connected to a part of speech).
        
        Args:
            word: The word to check.
            
        Returns:
            True if the word is known, False otherwise.
        """
        if not word:
            return False
        
        word_lower = word.lower().strip()
        
        # Check if word exists as a node
        word_node = self.graph.get_node_by_name(word_lower)
        if word_node:
            # Check if it's connected to a part of speech (is_a relationship)
            edges = self.graph.get_edges_from_node(word_node.id)
            for edge in edges:
                if edge.type == "is_a":
                    target_node = self.graph.get_node_by_id(edge.target)
                    if target_node and target_node.name in ["noun", "verb", "adjective", "adverb", 
                                                             "pronoun", "preposition", "conjunction", 
                                                             "determiner", "article", "concept"]:
                        return True
        
        # Check if word has a definition
        if self.get_word_definition(word_lower) is not None:
            return True
        
        return False
    
    def get_all_known_words(self) -> List[str]:
        """
        Returns a list of all words that have definitions.
        
        Returns:
            A list of word strings.
        """
        known_words = []
        for node_id in self.graph.graph.nodes():
            node_data = self.graph.graph.nodes[node_id]
            if node_data.get("type") == "word":
                known_words.append(node_data.get("name", ""))
        return known_words
    
    def add_word_relationship(
        self,
        word1: str,
        word2: str,
        relation_type: str,
        confidence: float = 0.7,
    ) -> bool:
        """
        Adds a relationship between two words (e.g., synonym, antonym, related_to).
        
        Args:
            word1: First word.
            word2: Second word.
            relation_type: Type of relationship (synonym, antonym, related_to, etc.).
            confidence: Confidence in the relationship.
            
        Returns:
            True if successfully added, False otherwise.
        """
        logger.info(f"[Lexicon Manager]: Adding relationship '{word1}' --[{relation_type}]--> '{word2}'")
        
        try:
            word1_node = ConceptNode(word1, node_type="word")
            word1_node = self.graph.add_node(word1_node)
            
            word2_node = ConceptNode(word2, node_type="word")
            word2_node = self.graph.add_node(word2_node)
            
            edge = self.graph.add_edge(
                word1_node,
                word2_node,
                relation_type,
                weight=confidence,
            )
            
            if edge:
                logger.info(f"    - Added relationship")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"[Lexicon Manager Error]: Failed to add relationship: {e}")
            return False
    
    def get_related_words(self, word: str, relation_type: str = "related_to") -> List[str]:
        """
        Gets words related to a given word.
        
        Args:
            word: The word to find relations for.
            relation_type: The type of relationship to search for.
            
        Returns:
            A list of related word names.
        """
        word_node = self.graph.get_node_by_name(word)
        if not word_node:
            return []
        
        related = []
        edges = self.graph.get_edges_from_node(word_node.id)
        for edge in edges:
            if edge.type == relation_type:
                target_node = self.graph.get_node_by_id(edge.target)
                if target_node and target_node.node_type == "word":
                    related.append(target_node.name)
        
        return related
    
    def add_word(self, word: str, word_type: str = "concept") -> bool:
        """
        Add a word to the lexicon without a definition (simple word node).
        
        Args:
            word: The word to add.
            word_type: The type of word (noun, verb, concept, etc.).
            
        Returns:
            True if successfully added, False otherwise.
        """
        try:
            word_lower = word.lower().strip()
            if self.word_is_known(word_lower):
                return True  # Already known
            
            # Add word node
            word_node = ConceptNode(word_lower, node_type="word")
            word_node = self.graph.add_node(word_node)
            
            # Link to part of speech if provided
            if word_type and word_type != "concept":
                pos_node = self.graph.get_node_by_name(word_type)
                if not pos_node:
                    pos_node = ConceptNode(word_type, node_type="concept")
                    pos_node = self.graph.add_node(pos_node)
                
                if word_node and pos_node:
                    self.graph.add_edge(
                        word_node,
                        pos_node,
                        "is_a",
                        weight=0.9,
                        properties={"provenance": "manual", "confidence": 0.9},
                    )
            
            logger.debug(f"[Lexicon Manager]: Added word '{word_lower}' (type: {word_type})")
            return True
            
        except Exception as e:
            logger.error(f"[Lexicon Manager Error]: Failed to add word '{word}': {e}")
            return False
    
    def learn_word_from_wordnet(self, word: str) -> bool:
        """
        Automatically learn a word from WordNet when MOTHER encounters an unknown word.
        
        This method:
        1. Queries WordNet for the word's definition, part of speech, and related words
        2. Stores the definition in the knowledge graph
        3. Links the word to its part of speech
        4. Stores hypernyms (more general concepts) and related words
        
        Args:
            word: The word to learn.
            
        Returns:
            True if successfully learned, False otherwise.
        """
        if not WORDNET_AVAILABLE:
            logger.debug(f"[Lexicon Manager] WordNet not available, cannot learn '{word}'")
            return False
        
        if self.word_is_known(word):
            logger.debug(f"[Lexicon Manager] Word '{word}' already known, skipping WordNet lookup")
            return True
        
        logger.info(f"[Lexicon Manager] Learning word '{word}' from WordNet...")
        
        try:
            # Get word information from WordNet
            word_info = get_word_info_from_wordnet(word)
            
            if not word_info.get("definitions"):
                logger.debug(f"[Lexicon Manager] No WordNet definition found for '{word}'")
                return False
            
            # Add word node with its part of speech
            word_type = word_info.get("type", "concept")
            word_node = ConceptNode(word.lower(), node_type="word")
            word_node = self.graph.add_node(word_node)
            
            # Link to part of speech
            pos_node = self.graph.get_node_by_name(word_type)
            if not pos_node:
                pos_node = ConceptNode(word_type, node_type="concept")
                pos_node = self.graph.add_node(pos_node)
            
            if word_node and pos_node:
                self.graph.add_edge(
                    word_node,
                    pos_node,
                    "is_a",
                    weight=0.95,
                    properties={"provenance": "wordnet", "confidence": 0.95},
                )
            
            # Add definition(s)
            primary_definition = word_info["definitions"][0] if word_info["definitions"] else None
            if primary_definition:
                self.add_word_definition(
                    word.lower(),
                    primary_definition,
                    word_type=word_type,
                    confidence=0.9,
                    provenance="wordnet",
                )
            
            # Add hypernyms (more general concepts) as relationships
            for hypernym in word_info.get("hypernyms_raw", [])[:5]:  # Limit to 5
                if hypernym and len(hypernym.split()) == 1:  # Only single-word hypernyms
                    hypernym_node = ConceptNode(hypernym.lower(), node_type="concept")
                    hypernym_node = self.graph.add_node(hypernym_node)
                    if word_node and hypernym_node:
                        self.graph.add_edge(
                            word_node,
                            hypernym_node,
                            "is_a",
                            weight=0.8,
                            properties={"provenance": "wordnet", "confidence": 0.8},
                        )
            
            # Add synonyms
            for synonym in word_info.get("synonyms", [])[:5]:  # Limit to 5
                if synonym and synonym != word.lower():
                    synonym_node = ConceptNode(synonym.lower(), node_type="word")
                    synonym_node = self.graph.add_node(synonym_node)
                    if word_node and synonym_node:
                        self.add_word_relationship(
                            word.lower(),
                            synonym.lower(),
                            "synonym",
                            confidence=0.85,
                        )
            
            logger.info(f"[Lexicon Manager] Successfully learned '{word}' from WordNet (type: {word_type}, definition: {primary_definition[:50] if primary_definition else 'N/A'}...)")
            return True
            
        except Exception as e:
            logger.error(f"[Lexicon Manager Error]: Failed to learn word '{word}' from WordNet: {e}")
            return False

