"""
Cognitive Agent for MOTHER

This module provides a unified orchestrator for MOTHER's cognitive processes.
It coordinates interpretation, learning, reasoning, and response generation
in a symbolic-first architecture similar to Axiom, but adapted for MOTHER's
emotional intelligence and identity formation capabilities.
"""

from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, cast

try:
    from thefuzz import process
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False
    logger.warning("[Cognitive Agent] thefuzz not available. Entity correction will be disabled.")

from processing.universal_interpreter import (
    UniversalInterpreter,
    InterpretData,
    RelationData,
    Entity,
)
from processing.symbolic_parser import SymbolicParser
from memory.knowledge_graph import KnowledgeGraph, ConceptNode, RelationshipEdge
from memory.structured_store import get_knowledge_graph, save_knowledge_graph
from memory.knowledge_harvester import KnowledgeHarvester
from memory.conflict_resolver import handle_conflict_with_user
from memory.learning_manager import LearningGoalManager
from memory.lexicon_manager import LexiconManager
from memory.lexicon_seeder import seed_core_vocabulary, seed_common_tech_words
from memory.unified_query import think
from personality.loader import load_config

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Relation type mappings
RELATION_TYPE_MAP: Dict[str, str] = {
    "be": "is_a",
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

# Exclusive relations (only one can exist per subject)
_EXCLUSIVE_RELATIONS = frozenset([
    "has_name",
    "is_located_in",
    "has_age",
    "is_from",
])

# Provenance ranking (higher = more trusted)
PROVENANCE_RANK: Dict[str, int] = {
    "system": 4,
    "dictionary": 3,
    "llm_verified": 2,
    "user": 1,
    "llm": 1,
}


class CognitiveAgent:
    """Unified orchestrator for MOTHER's cognitive processes.
    
    This class coordinates the entire cognitive loop:
    1. Symbolic parsing (deterministic)
    2. LLM interpretation (fallback for complex input)
    3. Intent processing
    4. Learning and knowledge storage
    5. Reasoning and query answering
    6. Response synthesis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Cognitive Agent.
        
        Args:
            config: Configuration dictionary (optional).
        """
        self.config = config or load_config()
        logger.info("[Cognitive Agent] Initializing...")
        
        # Core components
        self.interpreter = UniversalInterpreter(self.config)
        self.parser = SymbolicParser()
        self.graph = get_knowledge_graph()
        self.harvester = KnowledgeHarvester(self.graph, self.config)
        self.learning_manager = LearningGoalManager(self.graph, self.config)
        self.lexicon = LexiconManager(self.graph)
        
        # Seed lexicon with core vocabulary if not already seeded
        if self.config.get("seed_lexicon", True):
            try:
                # Check if lexicon is already seeded (quick check: see if common words exist)
                if not self.lexicon.word_is_known("the") or not self.lexicon.word_is_known("is"):
                    logger.info("[Cognitive Agent] Seeding lexicon with core vocabulary...")
                    core_count = seed_core_vocabulary(self.lexicon, self.graph)
                    tech_count = seed_common_tech_words(self.lexicon, self.graph)
                    save_knowledge_graph()
                    logger.info(f"[Cognitive Agent] Lexicon seeded: {core_count + tech_count} words added.")
                else:
                    logger.debug("[Cognitive Agent] Lexicon already seeded, skipping.")
            except Exception as e:
                logger.warning(f"[Cognitive Agent] Failed to seed lexicon: {e}")
        
        # State
        self.structured_history: List[Tuple[str, List[InterpretData]]] = []
        self.is_awaiting_clarification = False
        self.clarification_context: Dict = {}
        self.recently_researched: Dict[str, float] = {}  # Word -> timestamp
        self._has_reentered_chat = False  # Safety flag for re-entry
        
        logger.info("[Cognitive Agent] Initialized successfully")
    
    def chat(self, user_input: str, skip_cooldown: bool = False) -> str:
        """Process a single user input and return MOTHER's response.
        
        This orchestrates the entire cognitive process:
        1. Symbolic parse (try deterministic parsing first)
        2. LLM fallback (if parsing fails)
        3. Intent processing
        4. Learning (if statement of fact)
        5. Reasoning (if question)
        6. Response synthesis
        
        Args:
            user_input: The raw text message from the user.
            skip_cooldown: If True, skip the research cooldown check (used for re-entry after learning).
            
        Returns:
            A natural language string representing MOTHER's response.
        """
        logger.info(f"\n[Cognitive Agent] User: {user_input}")
        
        # Decay activations in knowledge graph (if method exists)
        if hasattr(self.graph, 'decay_activations'):
            self.graph.decay_activations(decay_rate=0.1)
        
        # Handle clarification if awaiting
        if self.is_awaiting_clarification:
            return self._handle_clarification(user_input)
        
        # Check research cooldown (skip if this is a re-entry after learning)
        if not skip_cooldown:
            for word, timestamp in list(self.recently_researched.items()):
                if time.time() - timestamp < 600:  # 10 minutes cooldown
                    if word in user_input.lower():
                        logger.info(f"[Cognitive Reflex]: Skipping '{word}' (cooldown active).")
                        return f"I'm still processing what I learned about '{word}'. Let's move on for now."
                else:
                    del self.recently_researched[word]
        
        # Preprocess input
        sanitized_input = self._sanitize_sentence_for_learning(user_input)
        expanded_input = self._expand_contractions(sanitized_input)
        contextual_input = self._resolve_references(expanded_input)
        normalized_input = self._preprocess_self_reference(contextual_input)
        
        # Try symbolic parsing first
        interpretations: Optional[List[InterpretData]] = self.parser.parse_fact(normalized_input)
        
        # Convert to InterpretData format if needed
        if interpretations:
            # Symbolic parser returns a dict, convert to InterpretData
            if isinstance(interpretations, dict):
                interpretations = [cast("InterpretData", {
                    "intent": "statement_of_fact" if interpretations.get("type") else "unknown",
                    "entities": [],
                    "relation": {
                        "subject": interpretations.get("subject", ""),
                        "verb": interpretations.get("relation", ""),
                        "object": interpretations.get("object", ""),
                    } if interpretations.get("subject") else None,
                    "key_topics": [interpretations.get("subject", "")],
                    "full_text_rephrased": user_input,
                    "provenance": "symbolic",
                    "confidence": 0.9,
                })]
            else:
                interpretations = [interpretations]
        
        # Check if parse was successful
        # IMPORTANT: If symbolic parser returned something, check if it's actually a question
        # The symbolic parser might incorrectly parse questions as facts
        if interpretations and len(interpretations) > 0:
            first_interpretation = interpretations[0]
            # Check if user input is clearly a question
            user_input_lower = user_input.lower().strip()
            question_indicators = ["what", "who", "where", "when", "why", "how", "which"]
            is_question = (
                any(user_input_lower.startswith(qw) for qw in question_indicators) or
                user_input_lower.endswith("?")
            )
            
            # If it's a question but symbolic parser said it's a statement, override it
            if is_question and first_interpretation.get("intent") == "statement_of_fact":
                logger.info(f"[Cognitive Agent] Detected question but symbolic parser said statement. Overriding intent.")
                # Force LLM interpretation for questions
                interpretations = None
        is_bad_parse = False
        if interpretations:
            entities_list = interpretations[0].get("entities", [])
            if entities_list:
                primary_entity = entities_list[0] if entities_list else {}
                entity_name = primary_entity.get("name", "")
                if len(entity_name.split()) > 5:
                    is_bad_parse = True
                    logger.warning(
                        "[Cognitive Flow]: Symbolic parse produced nonsensical entity. Forcing fallback to LLM."
                    )
        
        # Fallback to LLM if symbolic parsing failed
        if not interpretations or is_bad_parse:
            # IMPORTANT: The "brain" systems (lexicon, knowledge graph) should ENHANCE understanding,
            # not interrupt conversation. Check for unknown words in background, but don't block.
            # Only learn words for statements, not questions (questions need immediate answers)
            user_input_lower = normalized_input.lower().strip()
            question_indicators = ["what", "who", "where", "when", "why", "how", "which", "tell me", "explain", "describe"]
            is_question = (
                any(user_input_lower.startswith(qw) for qw in question_indicators) or
                user_input_lower.endswith("?")
            )
            
            # For questions: Answer immediately, learn words in background (don't interrupt)
            # For statements: Learn words first to better understand what user is telling us
            if not is_question:
                # Check for unknown words (now that lexicon is seeded)
                words = normalized_input.lower().split()
                unknown_words = []
                for w in words:
                    if not w or len(w.strip()) == 0:
                        continue
                    cleaned_word = re.sub(r"[^\w\s-]", "", w).strip()
                    if cleaned_word and len(cleaned_word) > 0 and not self.lexicon.word_is_known(cleaned_word):
                        unknown_words.append(cleaned_word)
                
                if unknown_words:
                    # Cognitive reflex: learn ALL unknown words first, then re-enter once
                    # Filter out very common words that might have been missed
                    common_word_patterns = ["etc", "u", "ur", "pls", "plz", "thx", "ty"]
                    truly_unknown = [w for w in unknown_words if w not in common_word_patterns]
                    
                    if truly_unknown:
                        # Learn ALL unknown words before re-entering (prevents multiple re-entries)
                        learned_words = []
                        failed_words = []
                        
                        for word_to_learn in sorted(set(truly_unknown)):
                            if word_to_learn and len(word_to_learn) > 0:
                                logger.info(f"[Cognitive Reflex]: Attempting to learn unknown word '{word_to_learn}' from WordNet...")
                                # Try to learn from WordNet first (automatic dictionary lookup)
                                learned = self.lexicon.learn_word_from_wordnet(word_to_learn)
                                if learned:
                                    logger.info(f"[Cognitive Reflex]: Successfully learned '{word_to_learn}' from WordNet.")
                                    learned_words.append(word_to_learn)
                                    # Mark as recently researched
                                    self.recently_researched[word_to_learn] = time.time()
                                else:
                                    # If WordNet doesn't have it, add to learning goals for LLM-based learning
                                    logger.info(f"[Cognitive Reflex]: '{word_to_learn}' not in WordNet, added to learning goals.")
                                    self.learning_manager.add_learning_goal(word_to_learn)
                                    failed_words.append(word_to_learn)
                        
                        # If we learned any words, save the knowledge graph and re-enter once
                        if learned_words:
                            from memory.structured_store import save_knowledge_graph
                            save_knowledge_graph()
                            logger.info(f"[Cognitive Reflex]: Learned {len(learned_words)} word(s): {', '.join(learned_words)}. Re-evaluating input...")
                            # Re-evaluate input once after learning all words
                            return self._chat_reentry_once(user_input)
                        # If no words were learned but we have failed words, proceed with LLM interpretation
            else:
                # For questions: Check for unknown words in background, but don't block
                # Add to learning goals for later, but answer the question NOW
                words = normalized_input.lower().split()
                unknown_words = []
                for w in words:
                    if not w or len(w.strip()) == 0:
                        continue
                    cleaned_word = re.sub(r"[^\w\s-]", "", w).strip()
                    if cleaned_word and len(cleaned_word) > 0 and not self.lexicon.word_is_known(cleaned_word):
                        unknown_words.append(cleaned_word)
                
                # Add unknown words to learning goals (background learning, don't interrupt)
                if unknown_words:
                    common_word_patterns = ["etc", "u", "ur", "pls", "plz", "thx", "ty"]
                    truly_unknown = [w for w in unknown_words if w not in common_word_patterns]
                    for word in truly_unknown:
                        logger.info(f"[Background Learning]: Adding '{word}' to learning goals for later.")
                        self.learning_manager.add_learning_goal(word)
            
            # Use LLM interpretation (can handle both known and unknown words)
            logger.info(f"[Cognitive Flow]: Symbolic parsing failed or not applicable. Falling back to LLM interpreter for: '{normalized_input}'")
            llm_interpretation = self.interpreter.interpret(normalized_input)
            if llm_interpretation:
                logger.info(f"[Cognitive Flow]: LLM interpretation successful. Intent: {llm_interpretation.get('intent', 'N/A')}")
                interpretations = [llm_interpretation]
            else:
                logger.warning("[Cognitive Flow]: LLM interpretation returned None or empty result.")
        
        if not interpretations:
            return "I'm sorry, I was unable to understand that."
        
        # Store in history
        self.structured_history.append(("user", interpretations))
        if len(self.structured_history) > 10:
            self.structured_history = self.structured_history[-10:]
        
        # Process primary interpretation
        primary_interpretation = interpretations[0]
        logger.info(
            "[Interpreter Output]: Intent='%s', Entities=%s, Relation=%s",
            primary_interpretation.get("intent", "N/A"),
            [e.get("name") for e in primary_interpretation.get("entities", [])],
            primary_interpretation.get("relation"),
        )
        
        intent = primary_interpretation.get("intent", "unknown")
        entities: List[Entity] = primary_interpretation.get("entities", [])
        relation: Optional[RelationData] = primary_interpretation.get("relation")
        
        # Process intent
        logger.info(f"[Intent Processing]: Processing intent '{intent}' with {len(entities)} entities")
        structured_response = self._process_intent(
            intent,
            entities,
            relation,
            user_input,
        )
        logger.info(f"[Intent Processing]: Intent processing complete. Response type: {type(structured_response).__name__}")
        
        # Process additional interpretations (if multiple)
        if len(interpretations) > 1:
            for extra in interpretations[1:]:
                if extra.get("intent") == "statement_of_fact":
                    rel = extra.get("relation")
                    if rel:
                        self._process_statement_for_learning(rel)
        
        # Synthesize final response
        final_response, synthesizer_was_used = self._synthesize_response(
            structured_response,
            user_input,
        )
        
        # Introspection: analyze synthesized response for new knowledge
        if synthesizer_was_used and intent.startswith("question"):
            introspection_context = None
            if relation and relation.get("subject"):
                introspection_context = relation["subject"]
            elif entities:
                introspection_context = entities[0]["name"]
            
            if introspection_context:
                logger.info("[Introspection]: Analyzing synthesized response for new knowledge...")
                new_interpretations = self.parser.parse_fact(final_response)
                if new_interpretations and isinstance(new_interpretations, dict):
                    if new_interpretations.get("type") == "relationship":
                        self._process_statement_for_learning(cast("RelationData", {
                            "subject": new_interpretations.get("subject", ""),
                            "verb": new_interpretations.get("relation", ""),
                            "object": new_interpretations.get("object", ""),
                        }))
        
        return final_response
    
    def _resolve_references(self, text: str) -> str:
        """Resolve simple pronouns using stored interpretations from history."""
        pronouns_to_resolve = {"it", "they", "its", "their", "them", "this", "that"}
        if not any(
            re.search(rf"\b{pronoun}\b", text, re.IGNORECASE)
            for pronoun in pronouns_to_resolve
        ):
            return text
        
        # Find antecedent from history
        antecedent = None
        for speaker, interpretations in reversed(self.structured_history):
            if speaker == "user" and interpretations:
                primary_interpretation = interpretations[0]
                relation = primary_interpretation.get("relation")
                entities = primary_interpretation.get("entities", [])
                
                if relation and relation.get("subject"):
                    antecedent = relation["subject"]
                    break
                if entities:
                    antecedent = entities[0]["name"]
                    break
        
        if antecedent:
            clean_antecedent = self._clean_phrase(antecedent)
            if clean_antecedent:
                modified_text = re.sub(
                    r"\b(it|they|them|this|that)\b",
                    clean_antecedent,
                    text,
                    flags=re.IGNORECASE,
                )
                modified_text = re.sub(
                    r"\b(its|their)\b",
                    f"{clean_antecedent}'s",
                    modified_text,
                    flags=re.IGNORECASE,
                )
                
                if modified_text != text:
                    logger.info(
                        "[Coreference]: Resolved pronouns, transforming '%s' to '%s'",
                        text,
                        modified_text,
                    )
                    return modified_text
        
        return text
    
    def _handle_clarification(self, user_input: str) -> str:
        """Handle the user's response after a contradiction was detected."""
        logger.info("[Curiosity]: Processing user's clarification...")
        interpretation = self.interpreter.interpret(user_input)
        entities = interpretation.get("entities", [])
        
        if not entities:
            logger.warning("[Curiosity]: No entities found in clarification.")
            return "I'm sorry, I couldn't understand your clarification."
        
        correct_answer_name = self._clean_phrase(entities[0]["name"])
        subject_name = self.clarification_context.get("subject")
        relation_type = self.clarification_context.get("conflicting_relation")
        
        if not subject_name or not relation_type:
            return "I'm sorry, I cannot process this clarification."
        
        # Update graph with correct answer
        subject_node = self.graph.get_node_by_name(subject_name)
        if not subject_node:
            subject_node = ConceptNode(name=subject_name)
            self.graph.add_node(subject_node)
        
        correct_node = self.graph.get_node_by_name(correct_answer_name)
        if not correct_node:
            correct_node = ConceptNode(name=correct_answer_name)
            self.graph.add_node(correct_node)
        
        # Reinforce correct, punish incorrect
        edges = self.graph.get_edges_from_node(subject_node.id)
        for edge in edges:
            if edge.type == relation_type:
                target_node = self.graph.get_node_by_id(edge.target)
                if target_node:
                    if self._clean_phrase(target_node.name) == correct_answer_name:
                        # Reinforce
                        self.graph.update_edge_weight(edge, 1.0, "user")
                        logger.info(f"    - REINFORCED: {subject_name} --[{relation_type}]--> {correct_answer_name}")
                    else:
                        # Punish
                        self.graph.update_edge_weight(edge, 0.1, "user")
                        logger.info(f"    - PUNISHED: {subject_name} --[{relation_type}]--> {target_node.name}")
        
        # Add missing edge if needed
        has_correct_edge = any(
            self._clean_phrase(self.graph.get_node_by_id(edge.target).name) == correct_answer_name
            and edge.type == relation_type
            for edge in self.graph.get_edges_from_node(subject_node.id)
        )
        
        if not has_correct_edge:
            self.graph.add_edge(
                subject_node,
                correct_node,
                relation_type,
                weight=1.0,
                properties={"provenance": "user"},
            )
        
        save_knowledge_graph()
        self.is_awaiting_clarification = False
        self.clarification_context = {}
        return "Thank you for the clarification. I have updated my knowledge."
    
    def _process_intent(
        self,
        intent: str,
        entities: List[Entity],
        relation: Optional[RelationData],
        user_input: str,
    ) -> str:
        """Route the interpreted user input to the appropriate cognitive function."""
        if intent == "greeting":
            return "Hello! I'm MOTHER. How can I help you today?"
        if intent == "farewell":
            return "Goodbye! Take care."
        if intent in ("gratitude", "acknowledgment"):
            return "You're welcome!"
        if intent == "positive_affirmation":
            return "I'm glad you think so!"
        
        if intent == "statement_of_fact":
            # If relation is None but we have entities, try to reconstruct the relation from the input
            if not relation and entities:
                # Try to extract relation from user input directly
                # Pattern: "i play valorant , counterstrike , callofduty"
                relation = self._reconstruct_relation_from_entities(user_input, entities)
            
            if not relation:
                return "I understood this as a factual statement, but couldn't extract the relationship. Could you rephrase?"
            
            subject = relation.get("subject")
            predicate = (
                relation.get("predicate")
                or relation.get("verb")
                or relation.get("relation")
            )
            obj = relation.get("object")
            
            if not subject or not predicate or not obj:
                return "I understood this as a factual statement, but some elements were missing."
            
            # Enhanced: Check if entities exist that aren't in the object
            # This handles cases like "i play videogames , like valorant , callofduty"
            # where the interpreter only extracted "videogames" as the object
            if entities and len(entities) > 0:
                entity_names = [e.get("name", "").lower() for e in entities if e.get("name")]
                obj_lower = obj.lower()
                
                # Check if any entities are missing from the object
                missing_entities = [e for e in entity_names if e not in obj_lower]
                
                if missing_entities:
                    # Reconstruct the full object list from the original input
                    # Pattern: "i play videogames , like valorant , callofduty"
                    input_lower = user_input.lower()
                    subject_lower = subject.lower()
                    predicate_lower = predicate.lower()
                    
                    # Try to find the full list after the verb
                    # Match pattern: subject verb [list with commas]
                    pattern = rf"{re.escape(subject_lower)}\s+{re.escape(predicate_lower)}\s+(.+?)(?:\.|$)"
                    match = re.search(pattern, input_lower, re.IGNORECASE)
                    
                    if match:
                        full_list = match.group(1).strip()
                        # Use the full list from the input instead of just the extracted object
                        obj = full_list
                        logger.info(f"[Enhanced Extraction]: Reconstructed full object list: '{obj}'")
            
            # Check for comma-separated lists in the object
            # Example: "i play video games , counter strike , valorant , call of duty"
            # Should become multiple facts: "i play video games", "i play counter strike", etc.
            comma_separated_facts = self._extract_comma_separated_facts(user_input, subject, predicate, obj)
            
            if len(comma_separated_facts) > 1:
                # Process multiple facts
                logger.info(f"[Knowledge Acquisition]: Detected {len(comma_separated_facts)} comma-separated facts")
                learned_count = 0
                failed_count = 0
                
                for fact_relation in comma_separated_facts:
                    was_learned, learn_msg = self._process_statement_for_learning(fact_relation)
                    if was_learned:
                        learned_count += 1
                    else:
                        failed_count += 1
                        logger.warning(f"[Knowledge Acquisition]: Failed to learn fact: {learn_msg}")
                
                if learned_count > 0:
                    save_knowledge_graph()
                    if learned_count == len(comma_separated_facts):
                        return f"I understand. I have noted all {learned_count} of those."
                    else:
                        return f"I understand. I noted {learned_count} out of {len(comma_separated_facts)} facts."
                else:
                    return f"I tried to record those facts but something went wrong."
            
            # Special handling for pet information with names
            # Pattern: "i have two rabbits, ones name is choco other is bhuto"
            # Check if this is a pet-related statement (has_pet verb, or has + number/animal, or contains pet names)
            is_pet_statement = (
                predicate.lower() in ["has_pet", "has"] or
                (predicate.lower() == "has" and (obj.lower().isdigit() or any(animal in user_input.lower() for animal in ["rabbit", "cat", "dog", "bird", "hamster", "fish", "monkey", "pet", "pets"])))
            )
            
            # Always try to extract pet names if the input contains pet name patterns
            # This handles cases where the interpreter extracted "two" but the input has "ones name is X"
            pet_names = self._extract_pet_names_from_input(user_input)
            pet_type = self._extract_pet_type_from_input(user_input)
            
            if pet_names:
                logger.info(f"[Pet Names]: Extracted pet names: {pet_names}, type: {pet_type}")
                # Store each pet name as a separate fact
                for pet_name in pet_names:
                    pet_relation = cast("RelationData", {
                        "subject": subject,
                        "verb": "has_pet",
                        "object": pet_name,
                        "properties": {"type": pet_type if pet_type else "pet"}
                    })
                    was_learned, learn_msg = self._process_statement_for_learning(pet_relation)
                    if was_learned:
                        logger.info(f"[Pet Names]: Stored pet name: {pet_name}")
                save_knowledge_graph()
                return f"I understand. I've noted that you have pets named {', '.join(pet_names)}."
            
            # If it's a pet statement but no names extracted, still process the main fact
            if is_pet_statement:
                logger.info(f"[Pet Statement]: Processing pet statement without names. Predicate: {predicate}, Object: {obj}")
            
            # Single fact processing (original logic)
            was_learned, learn_msg = self._process_statement_for_learning(relation)
            logger.info(f"[Knowledge Acquisition]: Learned = '{was_learned}', msg = '{learn_msg}'")
            
            if was_learned:
                return "I understand. I have noted that."
            if learn_msg == "exclusive_conflict":
                # Generate clarification question
                conflicting_nodes = self.graph.get_conflicting_facts(
                    subject,
                    predicate,
                )
                clarification_question = self.interpreter.synthesize(
                    structured_facts=[node.name for node in conflicting_nodes],
                    mode="clarification_question",
                )
                
                self.is_awaiting_clarification = True
                self.clarification_context = {
                    "subject": subject,
                    "conflicting_relation": predicate,
                    "conflicting_nodes": [node.name for node in conflicting_nodes],
                }
                return clarification_question
            
            return f"I tried to record that fact but something went wrong: {learn_msg}"
        
        if intent == "question_yes_no" and relation:
            return self._answer_yes_no_question(relation)
        
        if intent == "question_by_relation" and relation:
            subject = relation.get("subject")
            verb = relation.get("verb")
            
            if subject and verb:
                corrected_subject = self._get_corrected_entity(subject)
                subject_node = self.graph.get_node_by_name(corrected_subject)
                
                if subject_node:
                    for edge in self.graph.get_edges_from_node(subject_node.id):
                        if edge.type == verb:
                            target_node = self.graph.get_node_by_id(edge.target)
                            if target_node:
                                property_name = verb.replace("has_", "").replace("_", " ")
                                return f"The {property_name} of {subject.capitalize()} is {target_node.name.capitalize()}."
            
            return f"I don't have information about the {relation.get('verb', 'property').replace('has_', '')} of {subject}."
        
        if intent in ("question_about_entity", "question_about_concept") and relation:
            start_concept = relation.get("subject")
            end_concept = relation.get("object")
            if isinstance(start_concept, str) and isinstance(end_concept, str):
                start_concept = self._get_corrected_entity(start_concept)
                end_concept = self._get_corrected_entity(end_concept)
                start_node = self.graph.get_node_by_name(start_concept)
                end_node = self.graph.get_node_by_name(end_concept)
                if start_node and end_node:
                    logger.debug(
                        "[Multi-Hop]: Querying for path between '%s' and '%s'.",
                        start_node.name,
                        end_node.name,
                    )
                    path = self._perform_multi_hop_query(start_node, end_node)
                    if path:
                        explanation = self._format_path_as_sentence(path)
                        return f"Based on what I know: {explanation}"
                    return f"I don't know of a direct relationship between {start_concept} and {end_concept}."
        
        if intent in ("question_about_entity", "question_about_concept"):
            # Check if this is asking for "everything" or "all" in knowledge graph
            user_input_lower = user_input.lower()
            is_everything_query = any(phrase in user_input_lower for phrase in [
                "everything you have", "everything in your", "all you have", "all in your",
                "tell me about everything", "what do you have", "what's in your",
                "list everything", "show everything", "what's stored", "what have you stored"
            ]) and "knowledge graph" in user_input_lower
            
            if is_everything_query:
                logger.info(f"[Question Processing]: Detected 'everything' query about knowledge graph, enumerating all stored data")
                return self._get_all_knowledge_graph_contents()
            
            # Use unified thinking system (queries all memory layers)
            logger.info(f"[Question Processing]: Using unified thinking system for question: '{user_input}'")
            thinking_result = think(user_input)
            logger.info(f"[Question Processing]: Thinking result - Confidence: {thinking_result.get('confidence', 0):.2f}, Has answer: {bool(thinking_result.get('answer'))}")
            
            if thinking_result.get("confidence", 0) >= 0.7 and thinking_result.get("answer"):
                logger.info(f"[Question Processing]: Returning answer from thinking system (confidence >= 0.7)")
                return thinking_result["answer"]
            else:
                logger.info(f"[Question Processing]: Thinking system confidence too low ({thinking_result.get('confidence', 0):.2f}) or no answer. Using LLM directly with system context...")
            
            # IMPORTANT: The knowledge graph and lexicon are MOTHER's "brain" - they should ENHANCE understanding
            # If the graph doesn't have good answers, use LLM with full context (including graph insights)
            # The LLM can use the graph's partial knowledge + its own understanding to answer
            from processing.context_builder import ContextBuilder
            from processing.llm_handler import get_response
            
            context_builder = ContextBuilder()
            system_context = context_builder._build_system_context()
            
            # Build context that includes what we DO know from the graph (even if incomplete)
            graph_insights = ""
            if entities:
                entity_name = entities[0]["name"] if entities else ""
                corrected_entity = self._get_corrected_entity(entity_name)
                graph_facts = self._answer_question_about(corrected_entity, user_input)
                if graph_facts:
                    graph_insights = f"\n\nRelevant knowledge from my memory: {graph_facts}"
            
            # Build a prompt that uses both graph knowledge and LLM understanding
            # Check if full thinking should be shown
            show_thinking = self.config.get("show_full_thinking", False)
            if show_thinking:
                prompt = f"User question: {user_input}{graph_insights}\n\nPlease provide a comprehensive answer that includes your reasoning process, step-by-step analysis, and how you arrived at your conclusion. Show your thinking process clearly. Use the knowledge from my memory if relevant, but feel free to draw from your own understanding to provide a complete answer."
            else:
                prompt = f"User question: {user_input}{graph_insights}\n\nPlease provide a direct, informative answer. Use the knowledge from my memory if relevant, but feel free to draw from your own understanding to provide a complete answer."
            
            try:
                # Increase max_tokens if full thinking is enabled
                llm_config = self.config.copy()
                if show_thinking:
                    llm_config["max_tokens"] = 2048  # Allow much longer responses with full reasoning
                
                llm_response = get_response(prompt, llm_config, system_context=system_context)
                if llm_response and len(llm_response.strip()) > 10:
                    logger.info(f"[Question Processing]: LLM provided answer using graph insights + understanding (length: {len(llm_response)})")
                    return llm_response
            except Exception as e:
                logger.warning(f"[Question Processing]: LLM fallback failed: {e}")
            
            # Fallback: Check structured memory directly (for pet queries, etc.)
            query_lower = user_input.lower()
            if "pet" in query_lower:
                from memory.structured_store import get_fact
                pet_data = get_fact("pets")
                if pet_data and isinstance(pet_data, dict) and pet_data.get("has_pets"):
                    names = pet_data.get("names", [])
                    count = pet_data.get("count", len(names) if names else 0)
                    if names:
                        if len(names) == 1:
                            return f"You have {count} pet. Its name is {names[0]}."
                        elif len(names) == 2:
                            return f"You have {count} pets. Their names are {names[0]} and {names[1]}."
                        else:
                            names_str = ', '.join(names[:-1]) + f', and {names[-1]}'
                            return f"You have {count} pets. Their names are {names_str}."
                    else:
                        return f"You have {count} pet{'s' if count > 1 else ''}."
            
            # Fallback to graph query with entity correction and multi-hop reasoning
            entity_name = entities[0]["name"] if entities else user_input
            corrected_entity_name = self._get_corrected_entity(entity_name)
            response = self._answer_question_about(corrected_entity_name, user_input)
            return response if response else f"I don't have any specific information about '{corrected_entity_name}' right now."
        
        if intent == "command_show_all_facts":
            return self._get_all_facts_as_string()
        
        # Handle generic "command" intents - check if it's asking for user information
        if intent == "command":
            # Check if the command is asking for user information (pets, games, personal facts, etc.)
            user_input_lower = user_input.lower()
            is_user_info_query = any(keyword in user_input_lower for keyword in [
                "my pets", "my games", "what information", "what do you know about me",
                "list my", "tell me about me", "my personal", "stored about me"
            ])
            
            if is_user_info_query:
                logger.info(f"[Command Processing]: Detected user information query, using unified thinking system")
                # Use unified thinking system to retrieve user information
                thinking_result = think(user_input)
                logger.info(f"[Command Processing]: Thinking result - Confidence: {thinking_result.get('confidence', 0):.2f}, Has answer: {bool(thinking_result.get('answer'))}")
                
                if thinking_result.get("confidence", 0) >= 0.5 and thinking_result.get("answer"):
                    logger.info(f"[Command Processing]: Returning answer from thinking system")
                    return thinking_result["answer"]
                else:
                    # Fallback: Query knowledge graph directly for user information
                    logger.info(f"[Command Processing]: Thinking system didn't provide good answer, querying knowledge graph directly")
                    from memory.unified_query import _query_knowledge_graph
                    from memory.structured_store import get_knowledge_graph
                    
                    graph = get_knowledge_graph()
                    # Try multiple possible user node names - check ALL of them and merge results
                    user_nodes = []
                    for node_name in ["i", "user", "you"]:
                        node = graph.get_node_by_name(node_name)
                        if node:
                            user_nodes.append(node)
                    
                    if user_nodes:
                        logger.info(f"[Command Processing]: Found {len(user_nodes)} user node(s): {[n.name for n in user_nodes]}")
                        # Get all relationships from ALL user nodes, organized by type
                        pets = []
                        games = []
                        other_facts = []
                        
                        all_edges = []
                        for user_node in user_nodes:
                            edges = graph.get_edges_from_node(user_node.id)
                            logger.info(f"[Command Processing]: Found {len(edges)} edges from '{user_node.name}' node")
                            all_edges.extend(edges)
                        
                        logger.info(f"[Command Processing]: Total edges from all user nodes: {len(all_edges)}")
                        
                        # Process all edges from all user nodes
                        seen_pets = set()  # Avoid duplicates
                        seen_games = set()  # Avoid duplicates
                        
                        for edge in all_edges:
                            target_node = graph.get_node_by_id(edge.target)
                            if target_node:
                                source_node_name = next((n.name for n in user_nodes if n.id == edge.source), "unknown")
                                logger.debug(f"[Command Processing]: Edge: {source_node_name} --[{edge.type}]--> {target_node.name}")
                                
                                if edge.type == "has_pet":
                                    # Avoid duplicates
                                    if target_node.name not in seen_pets:
                                        pet_type = edge.properties.get("type", "pet") if edge.properties else "pet"
                                        pets.append(f"{target_node.name} ({pet_type})")
                                        seen_pets.add(target_node.name)
                                        logger.info(f"[Command Processing]: Found pet: {target_node.name} ({pet_type})")
                                elif edge.type == "play":
                                    # Avoid duplicates
                                    if target_node.name not in seen_games:
                                        games.append(target_node.name)
                                        seen_games.add(target_node.name)
                                        logger.debug(f"[Command Processing]: Found game: {target_node.name}")
                                else:
                                    # Skip system/internal facts
                                    if edge.type not in ["is_a", "has_part_of_speech", "has_synonym", "has_hypernym"]:
                                        other_facts.append(f"{edge.type}: {target_node.name}")
                        
                        logger.info(f"[Command Processing]: Collected {len(pets)} pets, {len(games)} games, {len(other_facts)} other facts")
                        
                        # Build response with actual data
                        response_parts = []
                        
                        if pets:
                            response_parts.append(f"**Your Pets:**\n{', '.join(pets)}")
                        if games:
                            response_parts.append(f"**Games You Play:**\n{', '.join(games)}")
                        if other_facts:
                            response_parts.append(f"**Other Personal Facts:**\n" + "\n".join(other_facts[:10]))
                        
                        if response_parts:
                            facts_section = "\n\n".join(response_parts)
                            
                            # Add technical explanation if requested
                            if "technically" in user_input_lower or "how" in user_input_lower:
                                technical_explanation = (
                                    "\n\n**Technical Storage Details:**\n"
                                    "My knowledge graph stores this data using NetworkX (a Python graph library) and saves it as JSON to `data/structured_memory/knowledge_graph.json`. "
                                    "Each fact is stored as a relationship (edge) between nodes. For example, your pet names are stored as: "
                                    "`i --[has_pet]--> nia` where 'i' is the user node, 'has_pet' is the relationship type, and 'nia' is the pet name node. "
                                    "The graph is managed by `memory.knowledge_graph.py` using functions like `add_node()` and `add_edge()`. "
                                    "When you tell me something, the `CognitiveAgent` extracts facts and the `KnowledgeHarvester` stores them in the graph structure."
                                )
                                return facts_section + technical_explanation
                            
                            return facts_section + "\n\nWould you like me to explain how this data is stored technically?"
                    
                    # If no user node found, try to get any personal information
                    return "I don't have specific information stored about you yet. Could you share some details about yourself so I can remember them?"
            
            # For other commands, try to process as a question
            logger.info(f"[Command Processing]: Generic command, treating as question")
            return self._process_intent("question_about_concept", entities, relation, user_input)
        
        return "I'm not sure how to process that. Could you rephrase?"
    
    def _process_statement_for_learning(
        self,
        relation: RelationData,
    ) -> Tuple[bool, str]:
        """Process a structured fact, handling validation and belief revision."""
        subject_name = relation.get("subject")
        verb_raw = relation.get("verb") or relation.get("relation")
        object_name = relation.get("object")
        
        if not all([subject_name, verb_raw, object_name]):
            return (False, "Incomplete fact structure")
        
        # Clean and normalize
        subject_name = self._clean_phrase(subject_name)
        object_name = self._clean_phrase(object_name)
        verb_cleaned = verb_raw.lower().strip()
        
        if len(object_name.split()) > 5:
            return (False, "Deferred learning: Object appears to be a description, not a concept.")
        
        logger.info(f"[LEARNING]: Processing: {subject_name} -> {verb_cleaned} -> {object_name}")
        
        # Get relation type
        relation_type = self.get_relation_type(verb_cleaned, subject_name, object_name)
        
        # Normalize relation type
        if relation_type in RELATION_TYPE_MAP:
            relation_type = RELATION_TYPE_MAP[relation_type]
        
        # Create or get nodes
        subject_node = ConceptNode(subject_name, node_type="concept")
        subject_node = self.graph.add_node(subject_node)
        
        obj_node = ConceptNode(object_name, node_type="concept")
        obj_node = self.graph.add_node(obj_node)
        
        # Handle exclusive relations
        if relation_type in _EXCLUSIVE_RELATIONS:
            was_learned, message = self._resolve_exclusive_conflict(
                subject_node,
                obj_node,
                relation_type,
                relation,
            )
        else:
            was_learned, message = self._add_new_fact(
                subject_node,
                obj_node,
                relation_type,
                relation,
            )
        
        # Update lexicon with new concepts
        self.lexicon.add_word(subject_name, "concept")
        self.lexicon.add_word(object_name, "concept")
        
        if was_learned:
            save_knowledge_graph()
            # Clear multi-hop cache when new knowledge is added
            try:
                self._gather_facts_multihop.cache_clear()
            except Exception:
                pass
            logger.info("[Cache]: Cleared reasoning cache due to new knowledge.")
        
        return (was_learned, message)
    
    def _resolve_exclusive_conflict(
        self,
        sub_node: ConceptNode,
        obj_node: ConceptNode,
        relation_type: str,
        relation_data: RelationData,
    ) -> Tuple[bool, str]:
        """Handle belief revision for relationships that must be unique."""
        # Check for existing exclusive relation
        existing_edges = self.graph.get_edges_from_node(sub_node.id)
        conflict_edge = None
        for edge in existing_edges:
            if edge.type == relation_type:
                conflict_edge = edge
                break
        
        if not conflict_edge:
            return self._add_new_fact(sub_node, obj_node, relation_type, relation_data)
        
        rel_props = relation_data.get("properties") or {}
        candidate_confidence = float(rel_props.get("confidence", 0.95))
        candidate_provenance = rel_props.get("provenance", "user")
        candidate_rank = PROVENANCE_RANK.get(candidate_provenance, 0)
        
        existing_confidence = conflict_edge.weight
        existing_prov = conflict_edge.properties.get("provenance", "unknown")
        existing_rank = PROVENANCE_RANK.get(existing_prov, 0)
        
        if candidate_rank > existing_rank or (candidate_rank == existing_rank and candidate_confidence > existing_confidence):
            logger.warning("[Belief Revision]: New fact is stronger. Deprecating old fact.")
            # Update edge properties
            conflict_edge.properties.update({"superseded_by": obj_node.name})
            self.graph.update_edge_properties(conflict_edge, conflict_edge.properties)
            # Reduce weight of old edge
            self.graph.update_edge_weight(conflict_edge, 0.2)
            return self._add_new_fact(sub_node, obj_node, relation_type, relation_data)
        
        if candidate_rank < existing_rank:
            logger.warning("[Belief Revision]: Existing fact is stronger. Rejecting new fact.")
            return (False, "existing_fact_stronger")
        
        logger.warning("[Belief Revision]: Stalemate detected. Triggering clarification.")
        return (False, "exclusive_conflict")
    
    def _add_new_fact(
        self,
        sub_node: ConceptNode,
        obj_node: ConceptNode,
        relation_type: str,
        relation_data: RelationData,
    ) -> Tuple[bool, str]:
        """Add a new fact to the knowledge graph."""
        # Check for conflicts
        conflict_result = handle_conflict_with_user(
            self.graph,
            sub_node.name,
            relation_type,
            obj_node.name,
            new_confidence=relation_data.get("properties", {}).get("confidence", 0.8),
            new_provenance=relation_data.get("properties", {}).get("provenance", "user"),
        )
        
        if conflict_result.get("needs_clarification"):
            return (False, "needs_clarification")
        
        if not conflict_result.get("has_conflict") or conflict_result.get("resolved"):
            edge = self.graph.add_edge(
                sub_node,
                obj_node,
                relation_type,
                weight=relation_data.get("properties", {}).get("confidence", 0.8),
                properties=relation_data.get("properties", {}),
            )
            if edge:
                logger.info(
                    "Learned new fact: %s --[%s]--> %s",
                    sub_node.name,
                    relation_type,
                    obj_node.name,
                )
                return (True, "inserted")
        
        return (False, "conflict_not_resolved")
    
    def get_relation_type(self, verb: str, subject: str, object_: str) -> str:
        """Determine the semantic relationship type from a simple verb."""
        return RELATION_TYPE_MAP.get(verb, verb.replace(" ", "_"))
    
    def _answer_question_about(self, entity_name: str, user_input: str) -> Optional[str]:
        """Find and return known facts related to a specific entity."""
        clean_entity_name = self._clean_phrase(entity_name)
        subject_node = self.graph.get_node_by_name(clean_entity_name)
        
        if not subject_node:
            return None
        
        logger.info(
            "[CognitiveAgent]: Starting multi-hop reasoning for '%s'.",
            entity_name,
        )
        
        # Use cached multi-hop fact gathering
        facts_with_props = self._gather_facts_multihop(subject_node.id, max_hops=4)
        
        # Check for temporal queries
        is_temporal_query = any(
            keyword in user_input.lower()
            for keyword in ["now", "currently", "today", "this year"]
        )
        
        if is_temporal_query:
            facts = self._filter_facts_for_temporal_query(facts_with_props)
        else:
            facts = {fact_str for fact_str, _ in facts_with_props}
        
        if not facts:
            return None
        
        # Filter out poor/useless facts (e.g., "X is a noun", "X is a word")
        # These don't help answer questions and just confuse the synthesis
        useful_facts = []
        useless_patterns = [
            " is a noun",
            " is a verb",
            " is a word",
            " is a concept",
            " is a thing",
            " is an entity",
        ]
        
        for fact in sorted(facts):
            # Skip facts that are just part-of-speech definitions
            is_useless = any(pattern in fact.lower() for pattern in useless_patterns)
            # Also skip very short facts (likely incomplete)
            is_too_short = len(fact.strip()) < 15
            
            if not is_useless and not is_too_short:
                useful_facts.append(fact)
        
        if not useful_facts:
            logger.info(f"[CognitiveAgent]: All facts filtered out as useless for '{entity_name}'. Returning None.")
            return None
        
        # If we have useful facts, return them
        if len(useful_facts) == 1:
            return useful_facts[0] + "."
        else:
            return ". ".join(useful_facts) + "."
    
    def _chat_reentry_once(self, user_input: str) -> str:
        """Safely re-enter chat() once after learning, avoiding infinite recursion."""
        if self._has_reentered_chat:
            logger.warning("[Safety]: Prevented recursive chat() re-entry. Processing input normally instead.")
            # Instead of returning a generic message, process the input normally
            # This handles cases where multiple words were learned and we've already re-entered once
            # Reset the flag and process normally
            self._has_reentered_chat = False
            # Continue with normal processing (skip cooldown since we just learned)
            return self.chat(user_input, skip_cooldown=True)
        
        self._has_reentered_chat = True
        try:
            logger.info("[Cognitive Reflex]: Re-entering chat() after learning new words...")
            # Skip cooldown check on re-entry since we just learned the word
            result = self.chat(user_input, skip_cooldown=True)
            logger.info("[Cognitive Reflex]: Re-entry completed successfully.")
            return result
        finally:
            self._has_reentered_chat = False
    
    def _get_corrected_entity(self, entity_name: str) -> str:
        """Uses fuzzy matching to correct potential typos in an entity name."""
        if not FUZZY_MATCHING_AVAILABLE:
            return entity_name
        
        all_concepts = list(self.graph.get_all_node_names())
        if not all_concepts:
            return entity_name
        
        match_result = process.extractOne(entity_name, all_concepts)
        if not match_result:
            return entity_name
        
        best_match: str = match_result[0]
        score: int = match_result[1]
        
        if score > 85:
            if entity_name != best_match:
                logger.info(
                    "[Cognitive Reflex]: Corrected entity '%s' to '%s' (confidence: %d%%).",
                    entity_name,
                    best_match,
                    score,
                )
            return best_match
        
        return entity_name
    
    @lru_cache(maxsize=256)
    def _gather_facts_multihop(
        self,
        start_node_id: str,
        max_hops: int,
    ) -> Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...]:
        """Gather all facts related to a starting node via graph traversal."""
        logger.info(
            "[Cache]: MISS! Executing full multi-hop graph traversal for node ID: %s",
            start_node_id,
        )
        start_node = self.graph.get_node_by_id(start_node_id)
        if not start_node:
            return ()
        
        found_facts: Dict[str, RelationshipEdge] = {}
        queue: List[Tuple[str, int]] = [(start_node_id, 0)]
        visited: set = {start_node_id}
        
        while queue:
            current_node_id, current_hop = queue.pop(0)
            if current_hop >= max_hops:
                continue
            
            current_node = self.graph.get_node_by_id(current_node_id)
            if not current_node:
                continue
            
            # Traverse outgoing edges
            for edge in self.graph.get_edges_from_node(current_node_id):
                if edge.type == "might_relate":
                    continue
                target_node = self.graph.get_node_by_id(edge.target)
                if target_node:
                    fact_str = f"{current_node.name} {edge.type.replace('_', ' ')} {target_node.name}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.target not in visited:
                        visited.add(edge.target)
                        queue.append((edge.target, current_hop + 1))
            
            # Traverse incoming edges
            for edge in self.graph.get_edges_to_node(current_node_id):
                if edge.type == "might_relate":
                    continue
                source_node = self.graph.get_node_by_id(edge.source)
                if source_node:
                    fact_str = f"{source_node.name} {edge.type.replace('_', ' ')} {current_node.name}"
                    if fact_str not in found_facts:
                        found_facts[fact_str] = edge
                    if edge.source not in visited:
                        visited.add(edge.source)
                        queue.append((edge.source, current_hop + 1))
        
        all_facts_items = list(found_facts.items())
        
        # Relevance filtering
        if len(all_facts_items) > 10:
            original_subject = start_node.name
            relevance_filtered = [
                (f, e)
                for f, e in all_facts_items
                if f.lower().startswith(original_subject.lower())
            ]
            if relevance_filtered:
                all_facts_items = relevance_filtered
        
        # Limit to top 10 by weight
        if len(all_facts_items) > 10:
            all_facts_items.sort(key=lambda item: item[1].weight, reverse=True)
            all_facts_items = all_facts_items[:10]
        
        final_results = []
        for fact_str, edge in all_facts_items:
            stringified_items = [
                (str(key), str(value)) for key, value in (edge.properties or {}).items()
            ]
            sorted_items = sorted(stringified_items)
            final_results.append((fact_str, tuple(sorted_items)))
        
        return tuple(final_results)
    
    def _filter_facts_for_temporal_query(
        self,
        facts_with_props_tuple: Tuple[Tuple[str, Tuple[Tuple[str, str], ...]], ...],
    ) -> set:
        """Filter a set of facts to find the most current one."""
        logger.debug("[TemporalReasoning]: Filtering facts by date...")
        today = datetime.utcnow().date()
        best_fact: Optional[str] = None
        best_date: Optional[date] = None
        
        facts_list = [
            (fact_str, dict(props_tuple))
            for fact_str, props_tuple in facts_with_props_tuple
        ]
        
        for fact_str, props in facts_list:
            date_str = props.get("effective_date")
            if date_str:
                try:
                    fact_date = datetime.fromisoformat(date_str).date()
                    if fact_date <= today:
                        if best_date is None or fact_date > best_date:
                            best_date = fact_date
                            best_fact = fact_str
                except (ValueError, TypeError):
                    continue
        
        if best_fact:
            return {best_fact}
        
        return {
            fact_str
            for fact_str, props in facts_list
            if not props.get("effective_date")
        }
    
    def _perform_multi_hop_query(
        self,
        start_node: ConceptNode,
        end_node: ConceptNode,
        max_hops: int = 3,
    ) -> Optional[List[RelationshipEdge]]:
        """Find a path of relationships between a start and end node."""
        queue: List[Tuple[str, List[RelationshipEdge]]] = [(start_node.id, [])]
        visited: set = {start_node.id}
        
        while queue:
            current_node_id, path = queue.pop(0)
            
            if len(path) >= max_hops:
                continue
            
            for edge in self.graph.get_edges_from_node(current_node_id):
                neighbor_id = edge.target
                
                if neighbor_id == end_node.id:
                    return path + [edge]
                
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [edge]
                    queue.append((neighbor_id, new_path))
        
        logger.warning("[BFS Engine]: FAILED. Queue is empty, no path found.")
        return None
    
    def _format_path_as_sentence(self, path: List[RelationshipEdge]) -> str:
        """Convert a path of edges into a human-readable sentence."""
        if not path:
            return ""
        
        parts = []
        for i, edge in enumerate(path):
            source_node = self.graph.get_node_by_id(edge.source)
            target_node = self.graph.get_node_by_id(edge.target)
            
            if not source_node or not target_node:
                continue
            
            if i == 0:
                parts.append(
                    f"{source_node.name.capitalize()} {edge.type.replace('_', ' ')} {target_node.name}"
                )
            else:
                parts.append(
                    f"which in turn {edge.type.replace('_', ' ')} {target_node.name}"
                )
        
        return ", and ".join(parts) + "."
    
    def _answer_yes_no_question(self, relation: RelationData) -> str:
        """Answer a yes/no question by checking for facts and contradictions."""
        subject = relation.get("subject", "")
        object_ = relation.get("object", "")
        
        if not subject or not object_:
            return "I'm not sure. I don't have enough information to answer that."
        
        subject_node = self.graph.get_node_by_name(subject)
        if not subject_node:
            return f"I don't have any information about {subject}."
        
        for edge in self.graph.get_edges_from_node(subject_node.id):
            target_node = self.graph.get_node_by_id(edge.target)
            if not target_node:
                continue
            
            if target_node.name == object_:
                return f"Yes, based on what I know, {subject} is {object_}."
            
            if edge.type in ["is_a", "has_property"]:
                if target_node.name != object_:
                    return f"No, based on what I know, {subject} is {target_node.name}, not {object_}."
        
        return f"I'm not sure. I don't have any information about whether {subject} is {object_}."
    
    def _get_all_facts_as_string(self) -> str:
        """Retrieve and format all facts from the knowledge graph."""
        all_edges = self.graph.get_all_edges()
        
        if not all_edges:
            return "My knowledge base is currently empty."
        
        high_confidence_edges = sorted(
            [
                edge
                for edge in all_edges
                if edge.weight >= 0.8
            ],
            key=lambda edge: edge.weight,
            reverse=True,
        )
        
        fact_strings = []
        for edge in high_confidence_edges:
            source_node = self.graph.get_node_by_id(edge.source)
            target_node = self.graph.get_node_by_id(edge.target)
            
            if source_node and target_node:
                fact_string = (
                    f"- {source_node.name.capitalize()} "
                    f"--[{edge.type}]--> "
                    f"{target_node.name.capitalize()} "
                    f"(Weight: {edge.weight:.2f})"
                )
                fact_strings.append(fact_string)
        
        if fact_strings:
            return (
                "Here are all the high-confidence facts I know (strongest first):\n\n"
                + "\n".join(fact_strings)
            )
        
        return "I currently lack high-confidence facts to display."
    
    def _get_all_knowledge_graph_contents(self) -> str:
        """Get a comprehensive overview of everything stored in the knowledge graph."""
        try:
            all_nodes = self.graph.get_all_node_names()
            all_edges = self.graph.get_all_edges()
            
            if not all_nodes:
                return "My knowledge graph is currently empty. I haven't stored any information yet."
            
            logger.info(f"[Knowledge Graph Contents]: Found {len(all_nodes)} nodes and {len(all_edges)} edges")
            
            # Helper function to filter out invalid data
            def is_valid_item(item: str, relation_type: str) -> bool:
                """Filter out invalid entries like 'two' as pet, 'i play' as thing loved, etc."""
                item_lower = item.lower()
                # Remove pet type in parentheses for checking
                item_clean = item_lower.split(" (")[0] if " (" in item_lower else item_lower
                
                # Filter out numbers as pets
                if relation_type == "has_pet" and item_clean in ["two", "three", "four", "five", "1", "2", "3", "4", "5"]:
                    return False
                
                # Filter out verb phrases as things loved/liked
                if relation_type in ["love", "like"] and any(verb in item_clean for verb in ["i play", "i love", "i like", "to play"]):
                    return False
                
                # Filter out very short or generic items
                if len(item_clean) < 2:
                    return False
                
                return True
            
            # Organize by category
            user_nodes = ["i", "user", "you"]
            user_data = {}
            user_linguistic_data = {}  # For definitions, synonyms, etc.
            system_data = {}
            concept_data = {}
            linguistic_data = {}  # For word definitions, synonyms, hypernyms
            relationship_stats = {}  # Count relationship types
            
            # Count relationship types
            for edge in all_edges:
                rel_type = edge.type
                relationship_stats[rel_type] = relationship_stats.get(rel_type, 0) + 1
            
            # Get all user-related data
            for user_node_name in user_nodes:
                user_node = self.graph.get_node_by_name(user_node_name)
                if user_node:
                    edges = self.graph.get_edges_from_node(user_node.id)
                    for edge in edges:
                        target_node = self.graph.get_node_by_id(edge.target)
                        if target_node:
                            # Separate linguistic facts (definitions, synonyms, etc.)
                            if edge.type in ["is_a", "has_part_of_speech", "has_synonym", "has_hypernym", "has_definition"]:
                                if edge.type not in user_linguistic_data:
                                    user_linguistic_data[edge.type] = []
                                user_linguistic_data[edge.type].append(target_node.name)
                            else:
                                # Regular user facts
                                if edge.type not in user_data:
                                    user_data[edge.type] = []
                                pet_type = edge.properties.get("type", "") if edge.properties else ""
                                item_str = f"{target_node.name} ({pet_type})" if pet_type else target_node.name
                                
                                # Filter invalid items
                                if is_valid_item(item_str, edge.type):
                                    user_data[edge.type].append(item_str)
            
            # Get system-related data (MOTHER node) - NO LIMIT
            mother_node = self.graph.get_node_by_name("MOTHER")
            if mother_node:
                edges = self.graph.get_edges_from_node(mother_node.id)
                for edge in edges:  # Removed limit
                    target_node = self.graph.get_node_by_id(edge.target)
                    if target_node:
                        if edge.type not in system_data:
                            system_data[edge.type] = []
                        system_data[edge.type].append(target_node.name)
            
            # Get linguistic data (definitions, synonyms, hypernyms) from all nodes
            for node_name in all_nodes:
                node = self.graph.get_node_by_name(node_name)
                if node:
                    edges = self.graph.get_edges_from_node(node.id)
                    for edge in edges:
                        if edge.type in ["has_definition", "has_synonym", "has_hypernym", "has_part_of_speech"]:
                            target_node = self.graph.get_node_by_id(edge.target)
                            if target_node:
                                key = f"{node_name} --[{edge.type}]--> {target_node.name}"
                                if edge.type not in linguistic_data:
                                    linguistic_data[edge.type] = []
                                linguistic_data[edge.type].append(key)
            
            # Build response
            response_parts = []
            response_parts.append(f"**Knowledge Graph Overview:**\n")
            response_parts.append(f"Total Nodes: {len(all_nodes)}\n")
            response_parts.append(f"Total Relationships: {len(all_edges)}\n")
            
            # Relationship type statistics
            if relationship_stats:
                response_parts.append(f"\n**Relationship Types:**\n")
                for rel_type, count in sorted(relationship_stats.items(), key=lambda x: x[1], reverse=True):
                    rel_display = rel_type.replace("_", " ").title()
                    response_parts.append(f"  {rel_display}: {count}")
            
            # User data section
            if user_data:
                response_parts.append(f"\n**About You:**\n")
                for relation_type, items in sorted(user_data.items()):
                    relation_display = relation_type.replace("_", " ").title()
                    if relation_type == "has_pet":
                        # Deduplicate and filter
                        seen = set()
                        filtered_items = []
                        for item in items:
                            item_clean = item.split(" (")[0] if " (" in item else item
                            if item_clean.lower() not in seen and is_valid_item(item, relation_type):
                                seen.add(item_clean.lower())
                                filtered_items.append(item)
                        if filtered_items:
                            response_parts.append(f"  Pets: {', '.join(filtered_items)}")
                    elif relation_type == "play":
                        response_parts.append(f"  Games You Play: {', '.join(items)}")
                    elif relation_type == "like":
                        # Filter out invalid items
                        filtered = [item for item in items if is_valid_item(item, relation_type)]
                        if filtered:
                            response_parts.append(f"  Things You Like: {', '.join(filtered)}")
                    elif relation_type == "love":
                        # Filter out invalid items
                        filtered = [item for item in items if is_valid_item(item, relation_type)]
                        if filtered:
                            response_parts.append(f"  Things You Love: {', '.join(filtered)}")
                    else:
                        filtered = [item for item in items if is_valid_item(item, relation_type)]
                        if filtered:
                            response_parts.append(f"  {relation_display}: {', '.join(filtered)}")
            
            # System data section - show ALL, not limited
            if system_data:
                response_parts.append(f"\n**About My System:**\n")
                for relation_type, items in sorted(system_data.items()):
                    relation_display = relation_type.replace("_", " ").title()
                    response_parts.append(f"  {relation_display}: {', '.join(items)}")  # Removed limit
            
            # ALL other concepts (not user nodes, not MOTHER, not already listed)
            user_node_names = set()
            for user_node_name in user_nodes:
                user_node = self.graph.get_node_by_name(user_node_name)
                if user_node:
                    user_node_names.add(user_node.name.lower())
            
            # Get all target nodes from user data
            user_target_nodes = set()
            for items in user_data.values():
                for item in items:
                    node_name = item.split(" (")[0] if " (" in item else item
                    user_target_nodes.add(node_name.lower())
            
            # Get all target nodes from system data
            system_target_nodes = set()
            for items in system_data.values():
                for item in items:
                    system_target_nodes.add(item.lower())
            
            # Get MOTHER node name
            mother_node_name = mother_node.name.lower() if mother_node else "mother"
            
            # Find ALL other concepts
            other_concepts = []
            for node_name in all_nodes:
                node_lower = node_name.lower()
                if (node_lower not in ["i", "user", "you"] and 
                    node_lower != mother_node_name and
                    node_lower not in user_node_names and
                    node_lower not in user_target_nodes and
                    node_lower not in system_target_nodes):
                    other_concepts.append(node_name)
            
            if other_concepts:
                response_parts.append(f"\n**All Other Concepts Stored ({len(other_concepts)} total):**\n")
                # Show first 100, then indicate remaining
                if len(other_concepts) <= 100:
                    response_parts.append(f"  {', '.join(other_concepts)}")
                else:
                    response_parts.append(f"  {', '.join(other_concepts[:100])}")
                    response_parts.append(f"  ... and {len(other_concepts) - 100} more concepts")
            
            # Linguistic data (definitions, synonyms, hypernyms)
            if linguistic_data:
                response_parts.append(f"\n**Word Knowledge (Definitions, Synonyms, Hypernyms):**\n")
                for rel_type, items in sorted(linguistic_data.items()):
                    rel_display = rel_type.replace("_", " ").title()
                    response_parts.append(f"  {rel_display}: {len(items)} relationships")
                    # Show first 20 examples
                    if items:
                        response_parts.append(f"    Examples: {', '.join(items[:20])}")
                        if len(items) > 20:
                            response_parts.append(f"    ... and {len(items) - 20} more")
            
            response_parts.append(f"\n**Storage Location:**\n")
            response_parts.append(f"  File: `data/structured_memory/knowledge_graph.json`")
            response_parts.append(f"  Format: NetworkX graph serialized as JSON")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"[Knowledge Graph Contents]: Error enumerating contents: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return f"I encountered an error while retrieving my knowledge graph contents: {str(e)}"
    
    def _synthesize_response(
        self,
        structured_response: str,
        user_input: str,
    ) -> Tuple[str, bool]:
        """Convert a structured, internal response into natural language."""
        non_synthesize_triggers = [
            "Hello!",
            "Goodbye!",
            "I understand. I have noted that.",
            "I don't have any information about",
            "You're welcome!",
            "I'm glad you think so!",
            "Here are all the high-confidence facts",
            "Thank you for the clarification.",
            "**Your Pets:**",  # Command responses with formatted sections
            "**Games You Play:**",
            "**Other Personal Facts:**",
            "**Technical Storage Details:**",
            "**Knowledge Graph Overview:**",  # Knowledge graph contents enumeration
            "**About You:**",
            "**About My System:**",
            "**Other Concepts Stored:**",
            "**Storage Location:**",
        ]
        
        if any(trigger in structured_response for trigger in non_synthesize_triggers):
            return (structured_response, False)
        
        # Use interpreter's synthesis
        fluent_response = self.interpreter.synthesize(
            structured_response,
            original_question=user_input,
        )
        
        return (fluent_response, True)
    
    def _sanitize_sentence_for_learning(self, sentence: str) -> str:
        """Pre-processes a raw sentence to make it easier to parse."""
        sanitized = re.sub(r"\s*\(.*?\)\s*", " ", sentence).strip()
        sanitized = re.sub(r"^.*?\)\s*", "", sanitized)
        sanitized = re.sub(
            r"^(In|According to)\s+[\w\s]+,\s*",
            "",
            sanitized,
            flags=re.IGNORECASE,
        )
        sanitized = sanitized.split(";")[0]
        sanitized = re.sub(r"^\s*[:\-\–]+\s*", "", sanitized)
        return sanitized.strip()
    
    def _clean_phrase(self, phrase: str) -> str:
        """Clean and normalize a phrase for use as a concept in the graph."""
        clean_phrase = phrase.lower().strip()
        clean_phrase = re.sub(r"\s*\([^)]*\)\s*", "", clean_phrase).strip()
        clean_phrase = re.sub(r"[.,!?;']+$", "", clean_phrase)
        
        words = clean_phrase.split()
        if len(words) > 1 and words[0] in ("a", "an", "the"):
            return " ".join(words[1:]).strip()
        
        return clean_phrase
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common English contractions."""
        contractions = {
            "what's": "what is",
            "it's": "it is",
            "i'm": "i am",
            "you're": "you are",
            "can't": "can not",
            "don't": "do not",
        }
        
        words = text.lower().split()
        expanded_words = [contractions.get(word, word) for word in words]
        return " ".join(expanded_words)
    
    def _preprocess_self_reference(self, text: str) -> str:
        """Normalize user input to replace self-references with canonical name."""
        processed_text = re.sub(
            r"\byour name\b",
            "MOTHER's name",
            text,
            flags=re.IGNORECASE,
        )
        processed_text = re.sub(
            r"\bwho are you\b",
            "what is MOTHER",
            processed_text,
            flags=re.IGNORECASE,
        )
        return processed_text
    
    def _extract_comma_separated_facts(
        self,
        user_input: str,
        subject: str,
        predicate: str,
        obj: str,
    ) -> List[RelationData]:
        """
        Extract multiple facts from comma-separated lists.
        
        Example: "i play video games , counter strike , valorant , call of duty"
        Returns: [
            {"subject": "i", "verb": "play", "object": "video games"},
            {"subject": "i", "verb": "play", "object": "counter strike"},
            {"subject": "i", "verb": "play", "object": "valorant"},
            {"subject": "i", "verb": "play", "object": "call of duty"},
        ]
        
        Args:
            user_input: Original user input
            subject: Subject from parsed relation
            predicate: Predicate/verb from parsed relation
            obj: Object from parsed relation
            
        Returns:
            List of RelationData dictionaries, one per fact
        """
        # Check if the object contains commas (indicating a list)
        if ',' not in obj:
            # No comma, return single fact
            return [cast("RelationData", {
                "subject": subject,
                "verb": predicate,
                "object": obj,
            })]
        
        # Split by comma, but be smart about it
        # Pattern: "video games , counter strike , valorant , call of duty"
        # We want to split on " , " (comma with spaces) but preserve multi-word items
        
        # First, check if the original input has the pattern: subject verb obj1 , obj2 , obj3
        input_lower = user_input.lower()
        subject_lower = subject.lower()
        predicate_lower = predicate.lower()
        
        # Enhanced: Handle multiple verbs in the sentence
        # Pattern: "i play videogames , like valorant , callofduty"
        # Should extract: "i play videogames", "i like valorant", "i play callofduty"
        
        # Check for multiple verbs (like, play, enjoy, love, etc.)
        common_verbs = ["play", "like", "enjoy", "love", "hate", "prefer"]
        verb_pattern = "|".join([re.escape(v) for v in common_verbs])
        
        # Find all verb occurrences in the input
        # Pattern: subject verb [object]
        verb_matches = list(re.finditer(
            rf"{re.escape(subject_lower)}\s+({verb_pattern})\s+([^,]+?)(?:\s*,\s*|$)",
            input_lower,
            re.IGNORECASE
        ))
        
        if len(verb_matches) > 1:
            # Multiple verbs found - extract facts for each verb-object pair
            facts = []
            for match in verb_matches:
                found_verb = match.group(1).lower()
                found_obj = match.group(2).strip()
                
                # Clean up object
                found_obj = re.sub(r'^\s*and\s+', '', found_obj, flags=re.IGNORECASE).strip()
                if found_obj and found_obj.lower() != "and":
                    facts.append(cast("RelationData", {
                        "subject": subject,
                        "verb": found_verb,
                        "object": found_obj,
                    }))
            
            if len(facts) > 1:
                logger.info(f"[Multi-Verb Detection]: Extracted {len(facts)} facts with different verbs: {[(f.get('verb'), f.get('object')) for f in facts]}")
                return facts
        
        # Fallback: Single verb with comma-separated list
        # Try to find where the list starts in the original input
        # Look for pattern like "i play video games , counter strike , valorant"
        pattern = rf"{re.escape(subject_lower)}\s+{re.escape(predicate_lower)}\s+(.+?)(?:\.|$)"
        match = re.search(pattern, input_lower, re.IGNORECASE)
        
        if match:
            # Extract the list portion
            list_portion = match.group(1).strip()
            
            # Split by " , " (comma with spaces on both sides)
            # Also handle "and" before the last item: "valorant, pubg, and counterstrike"
            items = [item.strip() for item in re.split(r'\s*,\s*', list_portion) if item.strip()]
            
            # Detect if there's a verb change in the middle of the list
            # Pattern: "to play videogames , i play , valorant, pubg"
            # Should detect that after "i play", subsequent items use "play" verb
            current_verb = predicate_lower
            cleaned_items = []
            
            for i, item in enumerate(items):
                # Check if item is "i play" or "i [verb]" - this indicates verb change
                verb_change_match = re.match(rf"^{re.escape(subject_lower)}\s+({verb_pattern})(?:\s*,\s*|$)", item, re.IGNORECASE)
                if verb_change_match:
                    # Verb change detected: "i play" means subsequent items use "play"
                    current_verb = verb_change_match.group(1).lower()
                    logger.info(f"[Verb Change]: Detected verb change to '{current_verb}' at position {i}")
                    continue  # Skip this item, it's just a verb indicator
                
                # Check if item starts with "to [verb]" pattern (like "to play videogames")
                to_verb_match = re.match(rf"^to\s+({verb_pattern})\s+(.+)$", item, re.IGNORECASE)
                if to_verb_match:
                    # Item has "to [verb] [object]" pattern
                    item_verb = to_verb_match.group(1).lower()
                    item_obj = to_verb_match.group(2).strip()
                    cleaned_items.append((item_verb, item_obj))
                    current_verb = item_verb  # Update current verb for subsequent items
                    continue
                
                # Check if item starts with a verb directly (like "like valorant")
                verb_match = re.match(rf"^({verb_pattern})\s+(.+)$", item, re.IGNORECASE)
                if verb_match:
                    # Item has its own verb: "like valorant"
                    item_verb = verb_match.group(1).lower()
                    item_obj = verb_match.group(2).strip()
                    cleaned_items.append((item_verb, item_obj))
                    current_verb = item_verb  # Update current verb for subsequent items
                else:
                    # Item is just an object, use the current verb (may have been updated above)
                    item = re.sub(r'^\s*and\s+', '', item, flags=re.IGNORECASE).strip()
                    if item and item.lower() != "and" and item.lower() != subject_lower:
                        cleaned_items.append((current_verb, item))
            
            if len(cleaned_items) > 1:
                # Found multiple items, create facts for each
                facts = []
                for item_verb, item_obj in cleaned_items:
                    facts.append(cast("RelationData", {
                        "subject": subject,
                        "verb": item_verb,
                        "object": item_obj,
                    }))
                logger.info(f"[Comma Detection]: Split into {len(facts)} facts: {[(f.get('verb'), f.get('object')) for f in facts]}")
                return facts
        
        # Fallback: if pattern matching failed, try splitting the object directly
        items = [item.strip() for item in obj.split(',') if item.strip()]
        # Clean up items: remove "and" prefix, filter out empty or just "and"
        cleaned_items = []
        for item in items:
            # Remove leading "and" if present
            item = re.sub(r'^\s*and\s+', '', item, flags=re.IGNORECASE).strip()
            # Skip if empty or just "and"
            if item and item.lower() != "and":
                cleaned_items.append(item)
        
        if len(cleaned_items) > 1:
            facts = []
            for item in cleaned_items:
                facts.append(cast("RelationData", {
                    "subject": subject,
                    "verb": predicate,
                    "object": item,
                }))
            logger.info(f"[Comma Detection]: Split object into {len(facts)} facts: {[f.get('object') for f in facts]}")
            return facts
        
        # No comma-separated list detected, return single fact
        return [cast("RelationData", {
            "subject": subject,
            "verb": predicate,
            "object": obj,
        })]
    
    def _extract_pet_names_from_input(self, user_input: str) -> List[str]:
        """Extract pet names from user input using various patterns."""
        pet_names = []
        input_lower = user_input.lower()
        
        # Pattern 1: "ones name is X other is Y" or "one's name is X other is Y"
        match = re.search(r"(?:one'?s|ones) name is ([a-z]+)(?:\s+other is ([a-z]+))?", input_lower, re.IGNORECASE)
        if match:
            if match.group(1):
                pet_names.append(match.group(1).strip())
            if match.group(2):
                pet_names.append(match.group(2).strip())
            return pet_names
        
        # Pattern 2: "one is named X, the other is Y" or "one is X, the other is Y"
        match = re.search(r"one (?:is|named) ([a-z]+)(?:,?\s+the other (?:is|named) ([a-z]+))?", input_lower, re.IGNORECASE)
        if match:
            if match.group(1):
                pet_names.append(match.group(1).strip())
            if match.group(2):
                pet_names.append(match.group(2).strip())
            return pet_names
        
        # Pattern 3: "named X and Y" or "called X and Y"
        match = re.search(r"(?:named|called) ([a-z]+)(?:\s+and\s+([a-z]+))?", input_lower, re.IGNORECASE)
        if match:
            if match.group(1):
                pet_names.append(match.group(1).strip())
            if match.group(2):
                pet_names.append(match.group(2).strip())
            return pet_names
        
        return pet_names
    
    def _extract_pet_type_from_input(self, user_input: str) -> Optional[str]:
        """Extract pet type from user input."""
        input_lower = user_input.lower()
        pet_types = {
            "rabbit": r"\b(?:rabbit|rabbits)\b",
            "cat": r"\b(?:cat|cats)\b",
            "dog": r"\b(?:dog|dogs)\b",
            "bird": r"\b(?:bird|birds)\b",
            "hamster": r"\b(?:hamster|hamsters)\b",
            "fish": r"\b(?:fish|fishes)\b",
        }
        
        for pet_type, pattern in pet_types.items():
            if re.search(pattern, input_lower, re.IGNORECASE):
                return pet_type
        
        return None
    
    def _reconstruct_relation_from_entities(
        self,
        user_input: str,
        entities: List[Entity],
    ) -> Optional[RelationData]:
        """
        Reconstruct a relation from entities when the interpreter didn't extract one.
        
        Example: "i play valorant , counterstrike , callofduty"
        Entities: ['Valorant', 'Counter-Strike', 'Call of Duty']
        Returns: {"subject": "i", "verb": "play", "object": "valorant , counterstrike , callofduty"}
        
        Args:
            user_input: Original user input
            entities: List of entities extracted by interpreter
            
        Returns:
            RelationData dictionary or None if reconstruction fails
        """
        if not entities:
            return None
        
        input_lower = user_input.lower()
        
        # Common verb patterns for statements
        verb_patterns = [
            (r"\bi\s+(play|like|enjoy|love|hate|prefer)\s+", "i"),
            (r"\bmy\s+(favorite|favourite|preferred)\s+", "my"),
            (r"\bi\s+(am|was|is)\s+", "i"),
            (r"\bi\s+(have|had|own)\s+", "i"),
        ]
        
        # Try to find subject and verb
        subject = "i"  # Default subject
        verb = "play"  # Default verb for game-related statements
        
        for pattern, default_subject in verb_patterns:
            match = re.search(pattern, input_lower, re.IGNORECASE)
            if match:
                verb = match.group(1) if match.groups() else "play"
                subject = default_subject
                break
        
        # Extract the object part (everything after the verb)
        # Pattern: "i play valorant , counterstrike , callofduty"
        # But also handle: "i love to play videogames , i play , valorant, pubg, and counterstrike"
        # In this case, we want to extract just the part after the last verb match
        
        # Find all verb matches
        verb_matches = list(re.finditer(rf"{re.escape(subject)}\s+{re.escape(verb)}\s+(.+?)(?:\.|$)", input_lower, re.IGNORECASE))
        
        if verb_matches:
            # Use the last match (most recent verb occurrence)
            last_match = verb_matches[-1]
            obj = last_match.group(1).strip()
            
            # Clean up: remove any leading "to" or "to play" if present
            obj = re.sub(r'^\s*to\s+(?:play\s+)?', '', obj, flags=re.IGNORECASE).strip()
        else:
            # Fallback: combine all entity names
            obj = ", ".join([e.get("name", "") for e in entities if e.get("name")])
        
        if obj:
            return cast("RelationData", {
                "subject": subject,
                "verb": verb,
                "object": obj,
            })
        
        return None

