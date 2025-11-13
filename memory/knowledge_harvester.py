"""
Knowledge Harvester for MOTHER

This module enables autonomous learning by harvesting knowledge from
user conversations, definitions, and external sources. It uses fact
verification and decomposition to build the knowledge graph.

Also implements autonomous learning cycles:
- Discovery Cycle: Finds new topics to learn
- Study Cycle: Researches and learns concepts
- Refinement Cycle: Breaks down complex facts into atomic ones
"""

import json
import logging
import random
import time
from datetime import datetime
from typing import List, Dict, Optional, TYPE_CHECKING

from memory.knowledge_graph import KnowledgeGraph, ConceptNode, RelationshipEdge
from memory.fact_verifier import verify_and_reframe_fact
from memory.fact_decomposer import decompose_sentence_to_relations, break_down_definition
from memory.conflict_resolver import handle_conflict_with_user

# Import web search utilities
try:
    from utils.web_search import (
        get_fact_from_wikipedia,
        get_fact_from_duckduckgo,
        find_new_topic,
    )
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    WEB_SEARCH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[Knowledge Harvester] Web search utilities not available")

if TYPE_CHECKING:
    from processing.universal_interpreter import UniversalInterpreter
    from memory.lexicon_manager import LexiconManager

logger = logging.getLogger(__name__)


class KnowledgeHarvester:
    """Harvests and stores knowledge from various sources"""
    
    def __init__(
        self,
        graph: KnowledgeGraph,
        config: dict | None = None,
        interpreter: Optional["UniversalInterpreter"] = None,
        lexicon: Optional["LexiconManager"] = None,
    ):
        """Initialize the knowledge harvester.
        
        Args:
            graph: The KnowledgeGraph instance to store knowledge in.
            config: Configuration dictionary (optional).
            interpreter: UniversalInterpreter for fact verification (optional).
            lexicon: LexiconManager for word knowledge (optional).
        """
        self.graph = graph
        self.config = config or {}
        self.interpreter = interpreter
        self.lexicon = lexicon
        self.enable_verification = self.config.get("enable_fact_verification", True)
        self.enable_decomposition = self.config.get("enable_fact_decomposition", True)
        self.rejected_topics: set = set()
        self.researched_topics: set = set()
    
    def harvest_from_sentence(
        self,
        sentence: str,
        topic: Optional[str] = None,
        provenance: str = "user",
        confidence: float = 0.8,
    ) -> Dict[str, any]:
        """
        Harvests knowledge from a single sentence.
        
        Args:
            sentence: The sentence to extract knowledge from.
            topic: Optional topic context for verification.
            provenance: Source of the knowledge.
            confidence: Initial confidence in the knowledge.
            
        Returns:
            A dictionary with harvest results:
            - "success": bool
            - "relations_added": int
            - "conflicts": List
            - "needs_clarification": bool
            - "clarification_question": str (if needed)
        """
        logger.info(f"[Knowledge Harvester]: Harvesting from sentence: '{sentence}'")
        
        # Step 1: Verify and reframe (if enabled)
        if self.enable_verification and topic:
            verified = verify_and_reframe_fact(topic, sentence, self.config)
            if not verified:
                logger.info("    - Fact verification rejected the sentence")
                return {
                    "success": False,
                    "relations_added": 0,
                    "conflicts": [],
                    "needs_clarification": False,
                }
            sentence = verified
        
        # Step 2: Decompose into relations (if enabled)
        if self.enable_decomposition:
            relations = decompose_sentence_to_relations(sentence, self.config)
        else:
            # Simple fallback: try to extract S-V-O manually
            relations = self._simple_extract_relations(sentence)
        
        if not relations:
            logger.warning("    - No relations extracted from sentence")
            return {
                "success": False,
                "relations_added": 0,
                "conflicts": [],
                "needs_clarification": False,
            }
        
        # Step 3: Store each relation in the graph
        relations_added = 0
        conflicts = []
        clarification_needed = False
        clarification_question = None
        
        for relation in relations:
            subject = relation.get("subject", "").strip()
            verb = relation.get("verb", "").strip()
            obj = relation.get("object", "").strip()
            
            if not all([subject, verb, obj]):
                continue
            
            # Normalize relation type from verb
            relation_type = self._verb_to_relation_type(verb)
            
            # Create or get nodes
            subject_node = ConceptNode(subject, node_type="concept")
            subject_node = self.graph.add_node(subject_node)
            
            obj_node = ConceptNode(obj, node_type="concept")
            obj_node = self.graph.add_node(obj_node)
            
            # Check for conflicts
            conflict_result = handle_conflict_with_user(
                self.graph,
                subject_node.name,
                relation_type,
                obj_node.name,
                confidence,
                provenance,
            )
            
            if conflict_result["has_conflict"]:
                conflicts.append(conflict_result)
                if conflict_result["needs_clarification"]:
                    clarification_needed = True
                    clarification_question = conflict_result.get("clarification_question")
                    # Don't add the relation if clarification is needed
                    continue
            
            # Add edge if no conflict or conflict was resolved
            if not conflict_result["has_conflict"] or conflict_result["resolved"]:
                edge = self.graph.add_edge(
                    subject_node,
                    obj_node,
                    relation_type,
                    weight=confidence,
                    properties={"provenance": provenance},
                )
                if edge:
                    relations_added += 1
                    logger.info(f"    - Added relation: {subject} --[{relation_type}]--> {obj}")
        
        return {
            "success": relations_added > 0,
            "relations_added": relations_added,
            "conflicts": conflicts,
            "needs_clarification": clarification_needed,
            "clarification_question": clarification_question,
        }
    
    def harvest_from_definition(
        self,
        word: str,
        definition: str,
        provenance: str = "dictionary",
        confidence: float = 0.9,
    ) -> Dict[str, any]:
        """
        Harvests knowledge from a dictionary-style definition.
        
        Args:
            word: The word being defined.
            definition: The definition text.
            provenance: Source of the definition.
            confidence: Confidence in the definition.
            
        Returns:
            A dictionary with harvest results.
        """
        logger.info(f"[Knowledge Harvester]: Harvesting definition for '{word}'")
        
        # Break down definition into relations
        relations = break_down_definition(word, definition, self.config)
        
        if not relations:
            return {
                "success": False,
                "relations_added": 0,
                "conflicts": [],
                "needs_clarification": False,
            }
        
        # Store relations
        return self.harvest_from_sentence(
            f"{word} is {definition}",
            topic=word,
            provenance=provenance,
            confidence=confidence,
        )
    
    def _verb_to_relation_type(self, verb: str) -> str:
        """
        Converts a verb to a standardized relation type.
        
        Args:
            verb: The verb from the relation.
            
        Returns:
            A standardized relation type string.
        """
        verb_lower = verb.lower()
        
        # Common verb mappings
        mappings = {
            "is": "is_a",
            "are": "is_a",
            "has": "has",
            "have": "has",
            "lives": "lives_in",
            "lived": "lives_in",
            "works": "works_at",
            "worked": "works_at",
            "located": "is_located_in",
            "from": "is_from",
            "named": "has_name",
            "called": "has_name",
            "likes": "likes",
            "enjoys": "enjoys",
            "owns": "owns",
            "has_pet": "has_pet",
        }
        
        return mappings.get(verb_lower, verb_lower.replace(" ", "_"))
    
    def _simple_extract_relations(self, sentence: str) -> List[Dict[str, str]]:
        """
        Simple fallback relation extraction (basic pattern matching).
        
        Args:
            sentence: The sentence to extract from.
            
        Returns:
            A list of relation dictionaries.
        """
        # Very basic extraction - just for fallback
        # In practice, this would be more sophisticated
        relations = []
        
        # Pattern: "X is Y"
        import re
        is_pattern = r"(\w+)\s+is\s+(.+)"
        match = re.search(is_pattern, sentence, re.IGNORECASE)
        if match:
            relations.append({
                "subject": match.group(1),
                "verb": "is",
                "object": match.group(2).strip(".,!?"),
            })
        
        # Pattern: "X has Y"
        has_pattern = r"(\w+)\s+has\s+(.+)"
        match = re.search(has_pattern, sentence, re.IGNORECASE)
        if match:
            relations.append({
                "subject": match.group(1),
                "verb": "has",
                "object": match.group(2).strip(".,!?"),
            })
        
        return relations
    
    def harvest_from_conversation(
        self,
        conversation_text: str,
        provenance: str = "user",
    ) -> Dict[str, any]:
        """
        Harvests knowledge from a longer conversation text.
        
        Args:
            conversation_text: The conversation text to extract from.
            provenance: Source of the knowledge.
            
        Returns:
            A dictionary with harvest results.
        """
        logger.info("[Knowledge Harvester]: Harvesting from conversation text")
        
        # Split into sentences
        import re
        sentences = re.split(r'[.!?]+', conversation_text)
        
        total_relations = 0
        all_conflicts = []
        clarification_questions = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Only process substantial sentences
                result = self.harvest_from_sentence(
                    sentence,
                    provenance=provenance,
                    confidence=0.7,  # Lower confidence for conversation
                )
                total_relations += result.get("relations_added", 0)
                all_conflicts.extend(result.get("conflicts", []))
                if result.get("needs_clarification"):
                    clarification_questions.append(result.get("clarification_question"))
        
        return {
            "success": total_relations > 0,
            "relations_added": total_relations,
            "conflicts": all_conflicts,
            "needs_clarification": len(clarification_questions) > 0,
            "clarification_questions": clarification_questions,
        }
    
    # ==================== AUTONOMOUS LEARNING CYCLES ====================
    
    def discover_cycle(self) -> None:
        """Run one discovery cycle to find a new topic to learn.
        
        This cycle explores new subjects and finds interesting topics
        that MOTHER doesn't know about yet.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n--- [Discovery Cycle Started at {timestamp}] ---")
        
        if not WEB_SEARCH_AVAILABLE:
            logger.warning("[Discovery Cycle]: Web search not available, skipping")
            return
        
        # Find a new topic
        new_topic = find_new_topic(
            rejected_topics=self.rejected_topics,
            max_attempts=3,
        )
        
        if new_topic:
            logger.info(f"[Discovery Cycle]: Found new topic: '{new_topic}'")
            
            # Check if MOTHER already knows about this topic
            if self.lexicon and self.lexicon.word_is_known(new_topic):
                logger.info(f"[Discovery Cycle]: Topic '{new_topic}' already known, skipping")
                self.rejected_topics.add(new_topic)
                return
            
            # Try to learn about this topic
            result = get_fact_from_wikipedia(new_topic) or get_fact_from_duckduckgo(new_topic)
            
            if result:
                topic, fact_sentence = result
                logger.info(f"[Discovery Cycle]: Found fact: '{fact_sentence[:100]}...'")
                
                # Verify and reframe if interpreter is available
                if self.interpreter:
                    try:
                        verified = self.interpreter.verify_and_reframe_fact(
                            original_topic=topic,
                            raw_sentence=fact_sentence,
                        )
                        if verified:
                            fact_sentence = verified
                    except Exception as e:
                        logger.debug(f"[Discovery Cycle]: Verification failed: {e}")
                
                # Harvest the fact
                harvest_result = self.harvest_from_sentence(
                    fact_sentence,
                    topic=topic,
                    provenance="autonomous_discovery",
                    confidence=0.7,  # Lower confidence for autonomous discovery
                )
                
                if harvest_result.get("success"):
                    logger.info(
                        f"[Discovery Cycle]: Successfully learned about '{topic}' "
                        f"({harvest_result.get('relations_added', 0)} relations added)"
                    )
                    self.researched_topics.add(new_topic)
                    # Save knowledge graph to disk
                    try:
                        from memory.structured_store import save_knowledge_graph
                        save_knowledge_graph()
                        logger.debug("[Discovery Cycle]: Knowledge graph saved to disk")
                    except Exception as e:
                        logger.warning(f"[Discovery Cycle]: Failed to save knowledge graph: {e}")
                else:
                    logger.warning(f"[Discovery Cycle]: Failed to harvest fact for '{topic}'")
                    self.rejected_topics.add(new_topic)
            else:
                logger.warning(f"[Discovery Cycle]: Could not find information about '{new_topic}'")
                self.rejected_topics.add(new_topic)
        else:
            logger.info("[Discovery Cycle]: No suitable new topic found")
        
        logger.info("--- [Discovery Cycle Finished] ---")
    
    def study_cycle(self, learning_goals: Optional[List[str]] = None) -> None:
        """Run one study cycle to learn about a concept.
        
        This cycle focuses on learning specific concepts, either from
        learning goals or by deepening knowledge of existing concepts.
        
        Args:
            learning_goals: Optional list of concepts to learn about.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n--- [Study Cycle Started at {timestamp}] ---")
        
        if not WEB_SEARCH_AVAILABLE:
            logger.warning("[Study Cycle]: Web search not available, skipping")
            return
        
        # Priority 1: Learn from learning goals
        if learning_goals and len(learning_goals) > 0:
            term_to_learn = learning_goals[0]
            logger.info(f"[Study Cycle]: Learning goal: '{term_to_learn}'")
            
            # Try dictionary API first (if single word and lexicon available)
            is_single_word = len(term_to_learn.split()) == 1
            if is_single_word and self.lexicon:
                # Check if already known
                if self.lexicon.word_is_known(term_to_learn):
                    logger.info(f"[Study Cycle]: '{term_to_learn}' already known")
                    return
            
            # Try web search
            queries = [
                f"what is {term_to_learn}",
                f"define {term_to_learn}",
                term_to_learn,
            ]
            
            web_fact = None
            source_topic = None
            for query in queries:
                result = get_fact_from_wikipedia(query) or get_fact_from_duckduckgo(query)
                if result:
                    source_topic, web_fact = result
                    break
                time.sleep(0.5)  # Small delay between queries
            
            if web_fact:
                logger.info(f"[Study Cycle]: Found fact: '{web_fact[:100]}...'")
                
                # Verify and reframe if interpreter is available
                if self.interpreter:
                    try:
                        verified = self.interpreter.verify_and_reframe_fact(
                            original_topic=source_topic or term_to_learn,
                            raw_sentence=web_fact,
                        )
                        if verified:
                            web_fact = verified
                    except Exception as e:
                        logger.debug(f"[Study Cycle]: Verification failed: {e}")
                
                # Harvest the fact
                harvest_result = self.harvest_from_sentence(
                    web_fact,
                    topic=source_topic or term_to_learn,
                    provenance="autonomous_study",
                    confidence=0.8,
                )
                
                if harvest_result.get("success"):
                    logger.info(
                        f"[Study Cycle]: Successfully learned '{term_to_learn}' "
                        f"({harvest_result.get('relations_added', 0)} relations added)"
                    )
                    # Save knowledge graph to disk
                    try:
                        from memory.structured_store import save_knowledge_graph
                        save_knowledge_graph()
                        logger.debug("[Study Cycle]: Knowledge graph saved to disk")
                    except Exception as e:
                        logger.warning(f"[Study Cycle]: Failed to save knowledge graph: {e}")
                    return
                else:
                    logger.warning(f"[Study Cycle]: Failed to harvest fact for '{term_to_learn}'")
            else:
                logger.warning(f"[Study Cycle]: Could not find information about '{term_to_learn}'")
        
        # Priority 2: Deepen knowledge of random existing concept
        else:
            logger.info("[Study Cycle]: No learning goals, deepening existing knowledge")
            self._deepen_knowledge_of_random_concept()
        
        logger.info("--- [Study Cycle Finished] ---")
    
    def refinement_cycle(self) -> None:
        """Run one refinement cycle to break down complex facts.
        
        This cycle finds "chunky" facts (long, definitional concepts)
        and breaks them down into smaller, more precise atomic facts.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"\n--- [Refinement Cycle Started at {timestamp}] ---")
        
        # Find a chunky fact (one with a long target node name or definition)
        chunky_fact = self._find_chunky_fact()
        
        if chunky_fact:
            source_node, target_node, edge = chunky_fact
            logger.info(
                f"[Refinement Cycle]: Found chunky fact: '{source_node.name}' "
                f"--[{edge.type}]--> '{target_node.name}'"
            )
            
            # Try to break down the target (if it's a definition)
            if self.interpreter:
                try:
                    # Create a sentence from the fact
                    fact_sentence = f"{source_node.name} {edge.type.replace('_', ' ')} {target_node.name}"
                    
                    # Break down into atomic facts
                    atomic_sentences = self.interpreter.break_down_definition(
                        word=source_node.name,
                        definition=target_node.name,
                    )
                    
                    if atomic_sentences and len(atomic_sentences) > 1:
                        logger.info(
                            f"[Refinement Cycle]: Breaking down into {len(atomic_sentences)} atomic facts"
                        )
                        
                        # Remove the original chunky edge
                        self.graph.remove_edge(edge)
                        
                        # Add atomic facts
                        relations_added = 0
                        for atomic_sentence in atomic_sentences:
                            result = self.harvest_from_sentence(
                                atomic_sentence,
                                provenance="autonomous_refinement",
                                confidence=0.85,
                            )
                            relations_added += result.get("relations_added", 0)
                        
                        logger.info(
                            f"[Refinement Cycle]: Refined fact into {relations_added} atomic relations"
                        )
                        
                        # Save knowledge graph to disk
                        try:
                            from memory.structured_store import save_knowledge_graph
                            save_knowledge_graph()
                            logger.debug("[Refinement Cycle]: Knowledge graph saved to disk")
                        except Exception as e:
                            logger.warning(f"[Refinement Cycle]: Failed to save knowledge graph: {e}")
                    else:
                        logger.info("[Refinement Cycle]: Could not break down fact further")
                except Exception as e:
                    logger.debug(f"[Refinement Cycle]: Refinement failed: {e}")
            else:
                logger.warning("[Refinement Cycle]: Interpreter not available for refinement")
        else:
            logger.info("[Refinement Cycle]: No chunky facts found to refine")
        
        logger.info("--- [Refinement Cycle Finished] ---")
    
    def _find_chunky_fact(self) -> Optional[tuple]:
        """Find a fact with a long/complex target that could be refined.
        
        Returns:
            A tuple of (source_node, target_node, edge) or None.
        """
        all_edges = self.graph.get_all_edges()
        
        # Look for edges with long target node names (likely definitions)
        chunky_candidates = []
        for edge in all_edges:
            target_node = self.graph.get_node_by_id(edge.target)
            if target_node and len(target_node.name) > 50:  # Long definition
                chunky_candidates.append(edge)
        
        if chunky_candidates:
            # Pick a random one
            edge = random.choice(chunky_candidates)
            source_node = self.graph.get_node_by_id(edge.source)
            target_node = self.graph.get_node_by_id(edge.target)
            if source_node and target_node:
                return (source_node, target_node, edge)
        
        return None
    
    def _deepen_knowledge_of_random_concept(self) -> None:
        """Deepen knowledge by learning more about a random existing concept."""
        all_nodes = self.graph.get_all_node_names()
        
        if not all_nodes:
            logger.info("[Study Cycle]: No concepts in knowledge graph to deepen")
            return
        
        # Pick a random concept
        random_concept = random.choice(all_nodes)
        logger.info(f"[Study Cycle]: Deepening knowledge of '{random_concept}'")
        
        # Try to find more information about it
        queries = [
            f"what is {random_concept}",
            f"more about {random_concept}",
            random_concept,
        ]
        
        for query in queries:
            result = get_fact_from_wikipedia(query) or get_fact_from_duckduckgo(query)
            if result:
                topic, fact_sentence = result
                
                # Verify and reframe
                if self.interpreter:
                    try:
                        verified = self.interpreter.verify_and_reframe_fact(
                            original_topic=topic,
                            raw_sentence=fact_sentence,
                        )
                        if verified:
                            fact_sentence = verified
                    except Exception:
                        pass
                
                # Harvest
                harvest_result = self.harvest_from_sentence(
                    fact_sentence,
                    topic=topic,
                    provenance="autonomous_deepening",
                    confidence=0.75,
                )
                
                if harvest_result.get("success"):
                    logger.info(
                        f"[Study Cycle]: Deepened knowledge of '{random_concept}' "
                        f"({harvest_result.get('relations_added', 0)} relations added)"
                    )
                    # Save knowledge graph to disk
                    try:
                        from memory.structured_store import save_knowledge_graph
                        save_knowledge_graph()
                        logger.debug("[Study Cycle]: Knowledge graph saved to disk")
                    except Exception as e:
                        logger.warning(f"[Study Cycle]: Failed to save knowledge graph: {e}")
                    return
                
                time.sleep(0.5)
        
        logger.info(f"[Study Cycle]: Could not find additional information about '{random_concept}'")

