"""
Universal Interpreter for MOTHER

This module provides a unified interface for complex language tasks using
Groq API. It handles interpretation, context resolution, decomposition,
synthesis, and other language understanding tasks.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Dict, List, Optional, TypedDict, Literal, NotRequired, cast

from processing.llm_handler import get_response

if TYPE_CHECKING:
    from memory.knowledge_graph import ConceptNode

logger = logging.getLogger(__name__)

# Pronoun patterns for context resolution
PRONOUNS = ("it", "its", "they", "them", "their", "he", "she", "his", "her", "this", "that", "these", "those")

# Intent types
Intent = Literal[
    "greeting",
    "farewell",
    "question_about_entity",
    "question_about_concept",
    "statement_of_fact",
    "statement_of_correction",
    "gratitude",
    "acknowledgment",
    "positive_affirmation",
    "command",
    "unknown",
    "question_yes_no",
    "meta_question_self",
    "meta_question_purpose",
    "meta_question_abilities",
    "command_show_all_facts",
    "question_by_relation",
]


class PropertyData(TypedDict, total=False):
    """Optional metadata about a relationship"""
    effective_date: NotRequired[str]
    location: NotRequired[str]
    confidence: NotRequired[float]
    provenance: NotRequired[str]
    negated: NotRequired[bool]
    revision_status: NotRequired[str]
    superseded_by: NotRequired[str]
    last_modified: NotRequired[float]
    confidence_updated: NotRequired[bool]


class RelationData(TypedDict, total=False):
    """Defines a structured relationship extracted from a sentence"""
    subject: str
    verb: NotRequired[str]
    object: str
    predicate: NotRequired[str]
    relation: NotRequired[str]
    properties: NotRequired[PropertyData]


class Entity(TypedDict):
    """Represents a key concept or entity extracted from text"""
    name: str
    type: Literal["CONCEPT", "PERSON", "ROLE", "PROPERTY"]


class InterpretData(TypedDict):
    """Structured result from the interpretation step"""
    intent: Intent
    entities: List[Entity]
    relation: Optional[RelationData]
    key_topics: List[str]
    full_text_rephrased: str
    provenance: NotRequired[str]
    confidence: NotRequired[float]


class UniversalInterpreter:
    """Provides a unified interface for complex language tasks using Groq API.
    
    This class acts as MOTHER's language understanding system, handling:
    - Structured interpretation of user input
    - Context resolution (pronoun handling)
    - Sentence decomposition into atomic facts
    - Definition breakdown
    - Fact verification
    - Natural language synthesis
    - Curriculum generation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Universal Interpreter.
        
        Args:
            config: Configuration dictionary (optional).
        """
        self.config = config or {}
        self.interpretation_cache: Dict[str, InterpretData] = {}
        self.synthesis_cache: Dict[str, str] = {}
        logger.info("[Universal Interpreter] Initialized with Groq API")
    
    def _is_pronoun_present(self, text: str) -> bool:
        """Check if any pronoun exists as a whole word in the text."""
        for pronoun in PRONOUNS:
            if re.search(rf"\b{pronoun}\b", text, re.IGNORECASE):
                return True
        return False
    
    def _clean_reasoning_text(self, raw_text: str) -> str:
        """Remove reasoning text from LLM responses (for synthesis).
        
        If show_full_thinking is enabled in config, returns the original text unchanged.
        """
        if not raw_text:
            return ""
        
        # Check if full thinking should be shown
        if self.config.get("show_full_thinking", False):
            logger.debug("[Synthesizer]: Full thinking enabled, returning original response with reasoning.")
            return raw_text
        
        cleaned = raw_text
        
        # Remove reasoning tags
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', cleaned)
        filtered_sentences = []
        
        # Reasoning indicators
        reasoning_indicators = [
            'okay, let\'s', 'first,', 'let me', 'wait,', 'the user wants',
            'i need to', 'i should', 'looking at', 'based on', 'the facts',
            'so combining', 'however,', 'but the', 'maybe the', 'perhaps',
            'the user\'s instruction', 'the question is', 'the facts provided',
            'i need to make sure', 'the original question', 'the rephrased sentence',
            'so i need to', 'combining the information', 'the task is to',
            'maybe start with', 'so, how can i', 'that\'s simple', 'the example given',
            'the response should be', 'the rules are', 'the initial prompt',
            'how can i rephrase', 'rephrase this', 'the given facts', 'using only',
            'no extra info', 'just the sentence', 'start directly', 'do not explain'
        ]
        
        # Find the first sentence that doesn't look like reasoning
        found_answer = False
        for sentence in sentences:
            sentence_lower = sentence.strip().lower()
            
            # Skip reasoning sentences
            if any(indicator in sentence_lower for indicator in reasoning_indicators):
                continue
            
            # If sentence looks like an actual answer (not reasoning), keep it
            if sentence.strip():
                # Check if it's a valid answer (not too short, not a question about the task)
                if len(sentence.strip()) > 10 and not sentence_lower.startswith(('so ', 'but ', 'wait', 'maybe', 'perhaps')):
                    filtered_sentences.append(sentence.strip())
                    found_answer = True
                elif found_answer:
                    # Once we found an answer, keep subsequent sentences
                    filtered_sentences.append(sentence.strip())
        
        # If we found filtered sentences, join them
        if filtered_sentences:
            cleaned = '. '.join(filtered_sentences)
            # Ensure it ends with proper punctuation
            if not cleaned.rstrip().endswith(('.', '!', '?')):
                cleaned = cleaned.rstrip() + '.'
        else:
            # Fallback: try to find the last sentence that might be the answer
            # Look for sentences that don't start with reasoning patterns
            for sentence in reversed(sentences):
                sentence_lower = sentence.strip().lower()
                if sentence.strip() and len(sentence.strip()) > 10:
                    if not any(indicator in sentence_lower for indicator in reasoning_indicators):
                        cleaned = sentence.strip()
                        if not cleaned.endswith(('.', '!', '?')):
                            cleaned += '.'
                        break
            else:
                # Last resort: return original but cleaned
                cleaned = raw_text.strip()
        
        return cleaned
    
    def _clean_llm_json_output(self, raw_text: str) -> str:
        """Clean and extract a JSON object from the raw output of an LLM."""
        if not raw_text:
            return ""
        
        # Remove common reasoning prefixes (like <think>, <reasoning>, <think>, etc.)
        cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<think>.*?</think>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        cleaned = re.sub(r'<reasoning>.*?</reasoning>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
        # Remove lines that start with reasoning patterns
        cleaned = re.sub(r'^Okay,.*?\n', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r'^First,.*?\n', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        cleaned = re.sub(r'^Let me.*?\n', '', cleaned, flags=re.MULTILINE | re.IGNORECASE)
        
        # Try to find JSON object boundaries
        start_brace = cleaned.find("{")
        end_brace = cleaned.rfind("}")
        
        if start_brace == -1 or end_brace == -1:
            # Try to find JSON in code blocks
            code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', cleaned, re.DOTALL)
            if code_block_match:
                return code_block_match.group(1)
            # Try to find JSON after "Output:" or similar markers
            output_match = re.search(r'(?:Output|JSON|Response):\s*(\{.*?\})', cleaned, re.DOTALL | re.IGNORECASE)
            if output_match:
                json_str = output_match.group(1)
                # Clean up trailing commas
                json_str = re.sub(r",\s*(\}|\])", r"\1", json_str)
                return json_str
            # Last resort: try to find any JSON-like structure
            json_like = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', cleaned, re.DOTALL)
            if json_like:
                return json_like.group(0)
            logger.warning(f"[JSON Cleaner] No JSON object found in response. First 200 chars: {raw_text[:200]}")
            return ""
        
        json_str = cleaned[start_brace : end_brace + 1]
        # Clean up trailing commas and fix common JSON issues
        json_str = re.sub(r",\s*(\}|\])", r"\1", json_str)
        json_str = re.sub(r'"\s*\n\s*"', '", "', json_str)
        # Remove any markdown code block markers if present
        json_str = re.sub(r'```(?:json)?\s*', '', json_str)
        json_str = re.sub(r'\s*```', '', json_str)
        return json_str
    
    def interpret(self, user_input: str) -> InterpretData:
        """Analyze user input with the LLM and return a structured interpretation.
        
        Args:
            user_input: The raw user message to be interpreted.
            
        Returns:
            A InterpretData object representing the structured understanding.
        """
        cache_key = user_input
        if cache_key in self.interpretation_cache:
            logger.debug("[Interpreter Cache]: Hit!")
            return self.interpretation_cache[cache_key]
        
        logger.info("[Interpreter]: Running LLM for interpretation.")
        
        system_prompt = (
            "You are a JSON-only output engine. Your task is to analyze user input and return ONLY a valid JSON object. "
            "CRITICAL RULES:\n"
            "1. Your output MUST start with '{' and end with '}'\n"
            "2. Do NOT include any reasoning, explanations, or thinking process\n"
            "3. Do NOT include markdown code blocks (no ```json```)\n"
            "4. Do NOT include any text before or after the JSON\n"
            "5. Do NOT use <think> tags or any reasoning tags\n"
            "6. Return ONLY the raw JSON object, nothing else\n"
            "INTENT CLASSIFICATION RULES:\n"
            "- If the input starts with question words (what, who, where, when, why, how, which) OR ends with '?', "
            "it is ALWAYS a question (use 'question_about_entity' or 'question_about_concept'), NEVER 'statement_of_fact'.\n"
            "- If the input is a declarative sentence that presents a fact (e.g., 'I play games', 'The sky is blue'), "
            "classify intent as 'statement_of_fact'.\n"
            "- Questions starting with 'how' are asking about processes, mechanisms, or methods - use 'question_about_concept'.\n"
            "- Questions starting with 'what', 'who', 'where', 'when', 'which' are asking about entities or concepts - use 'question_about_entity' or 'question_about_concept'.\n"
            "Otherwise, classify the user's intent and extract relationships into the JSON structure."
        )
        
        json_structure_prompt = (
            "The JSON object must have the following fields:\n"
            "- 'intent': Classify the user's primary intent. Possible values are: 'greeting', 'farewell', "
            "'question_about_entity', 'question_about_concept', 'statement_of_fact', 'statement_of_correction', "
            "'gratitude', 'acknowledgment', 'positive_affirmation', 'command', 'unknown'.\n"
            "- 'relation': If 'statement_of_fact' or 'statement_of_correction', extract the core relationship. "
            "This object has fields: 'subject', 'verb', 'object', and optional 'predicate', 'relation', or 'properties'.\n"
            "- 'key_topics': A list of the main subjects or topics.\n"
            "- 'full_text_rephrased': A neutral, one-sentence rephrasing.\n"
            "- 'entities': A list of entities with 'name' and 'type' fields."
        )
        
        examples_list = [
            'Input: \'show all facts\'\nOutput: {"intent": "command", "entities": [], "relation": null, "key_topics": ["show all facts"], "full_text_rephrased": "User has issued a command to show all facts."}',
            'Input: \'what is a human\'\nOutput: {"intent": "question_about_concept", "entities": [{"name": "human", "type": "CONCEPT"}], "relation": null, "key_topics": ["human"], "full_text_rephrased": "User is asking for information about a human."}',
            'Input: \'how is your knowledge graph storing data\'\nOutput: {"intent": "question_about_concept", "entities": [{"name": "knowledge graph", "type": "CONCEPT"}], "relation": null, "key_topics": ["knowledge graph", "storing data"], "full_text_rephrased": "User is asking how the knowledge graph stores data."}',
            'Input: \'how does the system work\'\nOutput: {"intent": "question_about_concept", "entities": [{"name": "system", "type": "CONCEPT"}], "relation": null, "key_topics": ["system"], "full_text_rephrased": "User is asking how the system works."}',
            'Input: \'correction: the sky is blue\'\nOutput: {"intent": "statement_of_correction", "entities": [{"name": "sky", "type": "CONCEPT"}], "relation": {"subject": "the sky", "verb": "is", "object": "blue"}, "key_topics": ["sky", "blue"], "full_text_rephrased": "User is correcting the fact about the sky to state that it is blue."}',
            'Input: \'In 2023, Tim Cook was the CEO of Apple.\'\nOutput: {"intent": "statement_of_fact", "entities": [{"name": "Tim Cook", "type": "PERSON"}, {"name": "CEO of Apple", "type": "ROLE"}], "relation": {"subject": "Tim Cook", "verb": "was", "object": "the CEO of Apple", "properties": {"effective_date": "2023-01-01"}}, "key_topics": ["Tim Cook", "Apple", "CEO"], "full_text_rephrased": "User is stating that Tim Cook was the CEO of Apple in 2023."}',
            'Input: \'who is Donald Trump?\'\nOutput: {"intent": "question_about_entity", "entities": [{"name": "Donald Trump", "type": "PERSON"}], "relation": null, "key_topics": ["Donald Trump"], "full_text_rephrased": "User is asking for information about Donald Trump."}',
        ]
        examples_prompt = "Here are some examples:\n" + "\n\n".join(examples_list)
        
        sanitized_input = json.dumps(user_input)
        full_prompt = (
            f"{json_structure_prompt}\n\n{examples_prompt}\n\n"
            f"USER INPUT: {sanitized_input}\n\n"
            f"YOUR RESPONSE (JSON ONLY, NO REASONING, NO EXPLANATIONS):"
        )
        
        try:
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.0  # Zero temperature for deterministic JSON
            llm_config["max_tokens"] = 512
            
            # Try to use structured response for JSON (if available)
            try:
                from processing.llm_handler import get_structured_response
                structured_response = get_structured_response(
                    prompt=full_prompt,
                    config=llm_config,
                    response_format={"type": "json_object"}
                )
                if structured_response and not structured_response.get("format_error"):
                    # Successfully got structured JSON
                    interpretation = cast("InterpretData", structured_response)
                    if isinstance(interpretation, dict):
                        interpretation.setdefault("provenance", "llm")
                        interpretation.setdefault("confidence", 0.8)
                        interpretation.setdefault("key_topics", [])
                        interpretation.setdefault("full_text_rephrased", "")
                        interpretation.setdefault("entities", [])
                        self.interpretation_cache[cache_key] = interpretation
                        return interpretation
            except (ImportError, Exception) as e:
                logger.debug(f"[Interpreter] Structured response not available, using regular: {e}")
            
            # Fallback to regular response
            response = get_response(
                prompt=full_prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if not response:
                raise ValueError("No response from LLM")
            
            # Log raw response for debugging (first 500 chars)
            logger.debug(f"[Interpreter] Raw LLM response (first 500 chars): {response[:500]}")
            
            cleaned_json_str = self._clean_llm_json_output(response)
            if not cleaned_json_str:
                logger.error(f"[Interpreter] Failed to extract JSON. Full response: {response}")
                raise json.JSONDecodeError("No JSON object found", response, 0)
            
            raw_interpretation = json.loads(cleaned_json_str)
            interpretation = cast("InterpretData", raw_interpretation)
            
            if isinstance(interpretation, dict):
                interpretation.setdefault("provenance", "llm")
                interpretation.setdefault("confidence", 0.6)
                interpretation.setdefault("key_topics", [])
                interpretation.setdefault("full_text_rephrased", "")
                interpretation.setdefault("entities", [])
                
                rel = interpretation.get("relation")
                if isinstance(rel, dict):
                    props = rel.setdefault("properties", cast("PropertyData", {}))
                    props.setdefault("confidence", 0.6)
                    props.setdefault("provenance", "llm")
                    
                    raw_verb = rel.get("verb", "").lower()
                    raw_object_text = rel.get("object", "")
                    
                    # Handle negation
                    if re.search(r"\b(not|never|no|without)\b", raw_verb, re.IGNORECASE) or \
                       re.search(r"\b(not|never|no|without)\b", raw_object_text, re.IGNORECASE):
                        props["negated"] = True
                        rel["object"] = re.sub(
                            r"\b(not|never|no|without)\b",
                            "",
                            raw_object_text,
                            flags=re.IGNORECASE,
                        ).strip()
                        rel["verb"] = re.sub(
                            r"\b(not|never|no|without)\b",
                            "",
                            raw_verb,
                            flags=re.IGNORECASE,
                        ).strip()
                    
                    interpretation["confidence"] = props.get("confidence", 0.6)
            
            self.interpretation_cache[cache_key] = interpretation
            return interpretation
            
        except Exception as e:
            logger.error(f"[Interpreter Error]: Could not parse LLM output. Error: {e}")
            return cast(
                "InterpretData",
                {
                    "intent": "unknown",
                    "entities": [],
                    "relation": None,
                    "key_topics": user_input.split(),
                    "full_text_rephrased": f"Could not fully interpret: '{user_input}'",
                    "provenance": "llm",
                    "confidence": 0.0,
                },
            )
    
    def resolve_context(self, history: List[str], new_input: str) -> str:
        """Use the LLM to perform coreference resolution on the user's input.
        
        This method attempts to replace pronouns in the user's latest
        message with the specific nouns they refer to from the preceding
        conversation history.
        
        Args:
            history: A list of the previous turns in the conversation.
            new_input: The user's latest message, potentially containing pronouns.
            
        Returns:
            The rephrased input string with pronouns resolved, or the original input if no changes were needed.
        """
        logger.info("[Context Resolver]: Attempting to resolve pronouns...")
        
        formatted_history = "\n".join(history)
        system_prompt = (
            "You are a strict coreference resolution engine. Your one and only task is to rephrase the 'New Input' "
            "by replacing pronouns (like it, its, they, them, their) with the specific noun they refer to from the 'Conversation History'.\n"
            "RULES:\n"
            "1. You MUST replace the pronoun with the full noun phrase from the history.\n"
            "2. If the New Input contains NO pronouns, you MUST return it completely unchanged.\n"
            "3. Your output MUST be ONLY the rephrased sentence and nothing else."
        )
        
        examples_prompt = (
            "Here are some examples:\n"
            "Conversation History:\n"
            "User: what is an apple?\nAgent: An apple is a fruit.\n"
            "New Input: what color is it?\nOutput: what color is an apple?\n\n"
            "Conversation History:\n"
            "User: tell me about dogs\nAgent: Dogs are mammals.\n"
            "New Input: what do they eat?\nOutput: what do dogs eat?\n\n"
            "Conversation History:\n"
            "User: tell me about the solar system\nAgent: The solar system has eight planets.\n"
            "New Input: what is the largest planet?\nOutput: what is the largest planet?"
        )
        
        full_prompt = (
            f"{system_prompt}\n\n{examples_prompt}\n\n"
            f"Conversation History:\n{formatted_history}\n"
            f"New Input: {new_input}\nOutput:"
        )
        
        try:
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.0
            llm_config["max_tokens"] = 128
            
            response = get_response(
                prompt=full_prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if response and response.strip().lower() != new_input.lower():
                logger.info(f"    - Context resolved: '{new_input}' -> '{response.strip()}'")
                return response.strip()
            
            logger.debug("    - No context to resolve, using original input.")
            return new_input
            
        except Exception as e:
            logger.error(f"[Context Resolver Error]: Could not resolve context. Error: {e}")
            return new_input
    
    def interpret_with_context(
        self,
        user_input: str,
        history: List[str],
    ) -> InterpretData:
        """Interpret user input after first attempting to resolve context.
        
        Args:
            user_input: The raw message from the user.
            history: The preceding conversation history.
            
        Returns:
            A InterpretData object representing the structured understanding.
        """
        contextual_input = user_input
        
        if history and self._is_pronoun_present(user_input):
            contextual_input = self.resolve_context(history, user_input)
        
        return self.interpret(contextual_input)
    
    def decompose_sentence_to_relations(
        self, text: str, main_topic: Optional[str] = None
    ) -> List[RelationData]:
        """Uses the LLM to decompose a sentence into a list of atomic relations.
        
        Args:
            text: The sentence to decompose.
            main_topic: Optional topic context.
            
        Returns:
            A list of atomic RelationData objects.
        """
        logger.info("  [Interpreter]: Decomposing sentence into atomic facts...")
        
        topic_context = (
            f"The primary topic of this sentence is '{main_topic}'. "
            "Ensure the subject of at least one core relation is this topic or a direct synonym."
            if main_topic
            else ""
        )
        
        prompt = f"""
        **ROLE:** You are a knowledge engineering system. Your task is to extract and decompose knowledge from a sentence into a list of simple, atomic semantic relations.

        **TASK:** Analyze the user's sentence and break it down into multiple, simple, atomic relations.
        - Each relation MUST be a JSON object: {{"subject": "...", "verb": "...", "object": "..."}}
        - Subjects and objects should be simple concepts (e.g., "cats", "photosynthesis", "the sun").
        - Verbs should be concise, standardized predicates (e.g., "is_a", "has_property", "causes", "is_part_of"). Use snake_case.
        - The goal is DECOMPOSITION. One complex sentence should become several simple facts.

        **CONTEXT:** {topic_context}

        **EXAMPLE:**
        Topic: Photosynthesis
        Sentence: "Photosynthesis is a process used by plants to convert light energy into chemical energy."
        Output:
        [
          {{"subject": "photosynthesis", "verb": "is_a", "object": "process"}},
          {{"subject": "photosynthesis", "verb": "is_used_by", "object": "plants"}},
          {{"subject": "photosynthesis", "verb": "converts", "object": "light energy"}},
          {{"subject": "light energy", "verb": "is_converted_into", "object": "chemical energy"}}
        ]

        **SENTENCE TO ANALYZE:**
        "{text}"

        **RULES:**
        1.  Return ONLY a valid JSON list of relation objects.
        2.  Do NOT include markdown, explanations, or any other text outside the JSON list.
        3.  If the sentence contains no extractable facts, return an empty list `[]`.
        """
        
        try:
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.1
            llm_config["max_tokens"] = 1024
            
            response = get_response(
                prompt=prompt,
                config=llm_config,
                system_context="You are a knowledge engineering system. Extract atomic relations from sentences.",
            )
            
            if not response:
                return []
            
            start_bracket = response.find("[")
            end_bracket = response.rfind("]")
            if start_bracket == -1 or end_bracket == -1:
                return []
            
            json_str = response[start_bracket : end_bracket + 1]
            relations = json.loads(json_str)
            
            if isinstance(relations, list):
                logger.info(f"    - Decomposed sentence into {len(relations)} atomic relations.")
                return cast("List[RelationData]", relations)
            return []
            
        except Exception as e:
            logger.error(f"  [Interpreter Error]: Failed to decompose sentence. Error: {e}")
            return []
    
    def break_down_definition(self, subject: str, chunky_definition: str) -> List[str]:
        """Use the LLM to break a complex definition into simple, atomic facts.
        
        Args:
            subject: The subject of the definition (e.g., "Bacteria").
            chunky_definition: The complex phrase to break down.
            
        Returns:
            A list of simple, atomic fact sentences.
        """
        logger.info(f"  [Interpreter]: Breaking down chunky definition for '{subject}'...")
        
        system_prompt = (
            "You are a logical decomposition engine. Your task is to break down a "
            "complex 'Definition' about a 'Subject' into a list of simple, atomic, "
            "declarative sentences. Each sentence must be a standalone fact.\n"
            "RULES:\n"
            "1. Each output sentence MUST start with the original 'Subject'.\n"
            "2. The output MUST be a simple list, with each sentence on a new line, "
            "prefixed with a hyphen.\n"
            "3. DO NOT add any other text, explanation, or commentary."
        )
        
        examples_prompt = (
            "Here are some examples:\n"
            "Subject: Bacteria\n"
            "Definition: ubiquitous, mostly free-living organisms often consisting of one biological cell\n"
            "Output:\n"
            "- Bacteria are ubiquitous.\n"
            "- Bacteria are mostly free-living organisms.\n"
            "- Bacteria consist of one biological cell.\n\n"
            "Subject: SymbolicParser\n"
            "Definition: a deterministic, rule-based parser for understanding simple language\n"
            "Output:\n"
            "- SymbolicParser is a deterministic parser.\n"
            "- SymbolicParser is a rule-based parser.\n"
            "- SymbolicParser is for understanding simple language."
        )
        
        full_prompt = (
            f"{system_prompt}\n\n{examples_prompt}\n\n"
            f"Subject: {subject.capitalize()}\nDefinition: {chunky_definition}\n"
            f"Output:"
        )
        
        try:
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.2
            llm_config["max_tokens"] = 256
            
            response = get_response(
                prompt=full_prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if not response:
                return []
            
            atomic_sentences = [
                s.strip()
                for s in response.replace("-", "").split("\n")
                if s.strip()
            ]
            
            if atomic_sentences:
                logger.info(f"    - Decomposed into {len(atomic_sentences)} atomic facts.")
                return atomic_sentences
            return []
            
        except Exception as e:
            logger.error(f"  [Interpreter Error]: Could not break down definition. Error: {e}")
            return []
    
    def verify_and_reframe_fact(
        self,
        original_topic: str,
        raw_sentence: str,
    ) -> Optional[str]:
        """Uses the LLM to verify if a raw sentence is relevant to a topic and reframe it.
        
        Args:
            original_topic: The topic the fact should be about.
            raw_sentence: The raw sentence to verify and potentially reframe.
            
        Returns:
            A simplified S-V-O sentence if relevant, None if irrelevant.
        """
        logger.info(f"  [Fact Verifier]: Verifying fact for '{original_topic}'...")
        
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
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.0
            llm_config["max_tokens"] = 64
            
            response = get_response(
                prompt=full_prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if not response:
                return None
            
            rephrased_fact = response.strip()
            
            if "none" in rephrased_fact.lower():
                logger.info("    - LLM rejected the fact as irrelevant.")
                return None
            
            logger.info(f"    - LLM verified and reframed: '{rephrased_fact}'")
            return rephrased_fact
            
        except Exception as e:
            logger.error(f"  [Fact Verifier Error]: Could not process fact with LLM. Error: {e}")
            return None
    
    def synthesize(
        self,
        structured_facts: str | List[RelationData] | List[str],
        original_question: Optional[str] = None,
        mode: str = "statement",
    ) -> str:
        """Convert a structured, internal representation into natural language.
        
        Args:
            structured_facts: The internal data to be verbalized.
            original_question: The user's question, used for context.
            mode: The synthesis mode ('statement' or 'clarification_question').
            
        Returns:
            A natural language string representing the synthesized response.
        """
        if isinstance(structured_facts, list):
            try:
                structured_facts = json.dumps([str(f) for f in structured_facts])
            except Exception:
                structured_facts = str(structured_facts)
        
        cache_key = f"{mode}|{original_question}|{structured_facts}"
        
        if cache_key in self.synthesis_cache:
            logger.debug("[Synthesizer Cache]: Hit!")
            return self.synthesis_cache[cache_key]
        
        logger.info(f"[Synthesizer]: Running LLM for synthesis in '{mode}' mode.")
        
        # Check if full thinking should be shown
        show_thinking = self.config.get("show_full_thinking", False)
        
        if show_thinking:
            rephrasing_prompt = (
                "You are a language rephrasing engine. Your task is to convert the given 'Facts' into a natural English response. "
                "You can show your thinking process and reasoning. Follow these rules:\n"
                "1.  **Use the information given in the 'Facts' string as the primary source.**\n"
                "2.  **You can show your reasoning process, step-by-step analysis, and how you arrived at your answer.**\n"
                "3.  **Be informative and explain your thought process clearly.**\n"
                "4.  **If helpful, you can mention what information you found, what connections you made, and how you synthesized the answer.**\n"
                "5.  **Provide a complete response that includes both your reasoning and the final answer.**"
            )
        else:
            rephrasing_prompt = (
                "You are a language rephrasing engine. Your task is to convert the given 'Facts' into a single, natural English sentence. "
                "You are a fluent parrot. You must follow these rules STRICTLY:\n"
                "1.  **ONLY use the information given in the 'Facts' string.**\n"
                "2.  **DO NOT add any extra information, commentary, or meta-analysis.**\n"
                "3.  **DO NOT apologize or mention your own limitations.**\n"
                "4.  **DO NOT show your thinking process, reasoning, or step-by-step analysis.**\n"
                "5.  **DO NOT start with 'Okay', 'First', 'Let me', 'Wait', 'The user', 'I need', or any reasoning phrases.**\n"
                "6.  **Your output must be ONLY the rephrased sentence and nothing else - no explanations, no reasoning, no thinking.**\n"
                "7.  **Start directly with the answer sentence. Do not explain how you arrived at it.**"
            )
        
        system_prompt = ""
        task_prompt = ""
        
        if mode == "clarification_question":
            system_prompt = (
                "You are an inquisitive AI agent. Your task is to ask a clarifying question. "
                "You have been given two conflicting facts. Formulate a single, polite, and simple question. "
                "Do not state the facts directly. Your output must be ONLY the question."
            )
            task_prompt = f"Conflicting Facts: '{structured_facts}'"
        else:
            system_prompt = rephrasing_prompt
            task_prompt = f"Facts to rephrase: '{structured_facts}'"
            if original_question:
                task_prompt = (
                    f"Using ONLY the facts provided, directly answer the question.\n"
                    f"Question: '{original_question}'\nFacts: '{structured_facts}'"
                )
        
        full_prompt = f"{system_prompt}\n\n{task_prompt}"
        
        try:
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.7 if mode == "clarification_question" else 0.1
            # Increase max_tokens if full thinking is enabled to allow longer, detailed responses
            if show_thinking:
                llm_config["max_tokens"] = 2048  # Allow much longer responses with full reasoning
            else:
                llm_config["max_tokens"] = 256
            
            response = get_response(
                prompt=full_prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if not response:
                return str(structured_facts)
            
            # Clean reasoning text from response (unless full thinking is enabled)
            synthesized_text = self._clean_reasoning_text(response)
            
            # Only clean up prefixes and parentheticals if full thinking is disabled
            if not self.config.get("show_full_thinking", False):
                # Remove quotes
                synthesized_text = synthesized_text.strip().replace('"', "")
                
                # Remove common prefixes
                phrases_to_remove = [
                    "rephrased sentence:",
                    "based on the provided facts,",
                    "the rephrased sentence is:",
                    "good output:",
                    "output:",
                    "answer:",
                    "response:",
                ]
                for phrase in phrases_to_remove:
                    if synthesized_text.lower().startswith(phrase):
                        synthesized_text = synthesized_text[len(phrase) :].strip()
                
                # Remove parenthetical explanations
                if "(" in synthesized_text:
                    synthesized_text = synthesized_text.split("(")[0].strip()
            
            self.synthesis_cache[cache_key] = synthesized_text
            return synthesized_text
            
        except Exception as e:
            logger.error(f"[Synthesizer Error]: Could not generate fluent text. Error: {e}")
            return str(structured_facts)
    
    def generate_curriculum(self, high_level_goal: str) -> List[str]:
        """Uses the LLM to break down a high-level learning goal into prerequisite topics.
        
        Args:
            high_level_goal: The high-level goal to break down.
            
        Returns:
            A list of prerequisite topic names.
        """
        logger.info(f"  [Interpreter]: Generating curriculum for goal '{high_level_goal}'...")
        
        system_prompt = (
            "You are a curriculum design expert. Your task is to break down a 'High-Level Goal' into a "
            "short, prioritized list of the most fundamental, prerequisite concepts needed to understand it. "
            "These concepts should be simple nouns or short noun phrases."
        )
        
        examples_prompt = (
            "RULES:\n"
            "1. Output ONLY a comma-separated list of topics.\n"
            "2. Prioritize the most foundational concepts first.\n"
            "3. Do not add numbers, bullets, or any other formatting.\n\n"
            "Example 1:\n"
            "High-Level Goal: Become an expert on ancient Rome\n"
            "Output: Roman Republic, Roman Empire, Julius Caesar, Augustus, Colosseum, Latin\n\n"
            "Example 2:\n"
            "High-Level Goal: Understand photosynthesis\n"
            "Output: plant, cell, sunlight, chlorophyll, water, carbon dioxide, oxygen, glucose"
        )
        
        full_prompt = (
            f"{system_prompt}\n\n{examples_prompt}\n\n"
            f"High-Level Goal: {high_level_goal}\n"
            f"Output:"
        )
        
        try:
            llm_config = self.config.copy()
            llm_config["temperature"] = 0.3
            llm_config["max_tokens"] = 128
            
            response = get_response(
                prompt=full_prompt,
                config=llm_config,
                system_context=system_prompt,
            )
            
            if not response:
                return []
            
            topics = [
                topic.strip() for topic in response.split(",") if topic.strip()
            ]
            logger.info(f"    - Generated curriculum with {len(topics)} topics.")
            return topics
            
        except Exception as e:
            logger.error(f"  [Interpreter Error]: Could not generate curriculum. Error: {e}")
            return []

