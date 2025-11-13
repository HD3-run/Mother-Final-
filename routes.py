from flask import Blueprint, render_template, request, jsonify
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import os
import traceback
import json
import numpy as np

# Import necessary modules
from processing.llm_handler import get_response
from processing.context_builder import build_prompt
from processing.intent_detector import detect_intent
from processing.predictive_modeling import predict_user_state, update_behavioral_patterns
from personality.loader import load_config
from personality.emotional_response import adjust_response_tone
from personality.identity_engine import update_identity, get_identity_state, reflect_on_identity
from memory.structured_store import set_fact, get_fact, all_facts, get_knowledge_graph, save_knowledge_graph # Removed init_db import
from memory.knowledge_harvester import KnowledgeHarvester
from memory.conflict_resolver import handle_conflict_with_user
from processing.symbolic_parser import SymbolicParser
from memory.vector_store import add_memory, search_memory
from memory.episodic_logger import log_event, get_today_log, get_conversation_history, get_log_for_date, get_all_conversation_dates
from memory.semantic_clustering import cluster_memories, get_memory_clusters
from memory.unified_query import think
from processing.cognitive_agent import CognitiveAgent
from processing.metacognitive_engine import MetacognitiveEngine
from reflection.reflection_engine import get_reflection_for_date, generate_autonomous_reflection
# Corrected import path for autonomous decision functions
from reflection.autonomous_decision import make_autonomous_decision, get_pending_actions
from utils.sentiment import get_sentiment, analyze_emotional_context
from utils.usage_tracker import log_user_input, log_response, get_usage_stats
from utils.logger import log_event as debug_log
from utils.ml_utils import train_user_model, get_model_predictions

# Define Blueprint
chat_api = Blueprint("chat_api", __name__)

# Load config
config = load_config()
# Removed init_db() call from here, as it's handled in app.py's create_app()

# --- Helper Functions (Moved to top for proper definition order) ---

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj

def extract_facts(user_input):
    """Enhanced fact extraction with knowledge graph, verification, and conflict detection"""
    try:
        # IMPORTANT: Don't extract facts from questions
        # Check if input is a question first
        user_input_lower = user_input.lower().strip()
        question_indicators = ["what", "who", "where", "when", "why", "how", "which"]
        is_question = (
            any(user_input_lower.startswith(qw) for qw in question_indicators) or
            user_input_lower.endswith("?")
        )
        
        if is_question:
            logging.info(f"[EXTRACT FACTS] Skipping fact extraction for question: '{user_input}'")
            return None
        
        # Initialize components
        graph = get_knowledge_graph()
        harvester = KnowledgeHarvester(graph, config)
        parser = SymbolicParser()
        
        # Use symbolic parser to extract facts
        parsed_fact = parser.parse_fact(user_input)
        
        if parsed_fact:
            # Check for conflicts before storing
            subject = parsed_fact.get('subject', 'user')
            relation = parsed_fact.get('relation', '')
            obj = parsed_fact.get('object', '')
            
            if subject and relation and obj:
                # Handle conflict detection
                conflict_result = handle_conflict_with_user(
                    graph,
                    subject,
                    relation,
                    obj,
                    new_confidence=0.8,
                    new_provenance="user",
                )
                
                if conflict_result.get('needs_clarification'):
                    # Return clarification question to user
                    clarification = conflict_result.get('clarification_question')
                    logging.warning(f"[CONFLICT] Clarification needed: {clarification}")
                    # Store this for later use in response
                    return {"needs_clarification": True, "question": clarification}
                
                # No conflict or resolved - proceed with storage
                if not conflict_result.get('has_conflict') or conflict_result.get('resolved'):
                    # Use knowledge harvester to store
                    harvest_result = harvester.harvest_from_sentence(
                        user_input,
                        topic=subject,
                        provenance="user",
                        confidence=0.8,
                    )
                    
                    if harvest_result.get('success'):
                        save_knowledge_graph()
                        logging.info(f"[FACTS] Fact stored via knowledge harvester: {subject} --[{relation}]--> {obj}")
        
        # Also use traditional extraction for backward compatibility
        # (This ensures existing code still works)
        # Enhanced name extraction with more patterns
        name_patterns = [
            r"\bmy name is ([A-Za-z0-9_ \-]+)",
            r"\bi am ([A-Za-z0-9_ \-]+)",
            r"\bcall me ([A-Za-z0-9_ \-]+)",
            r"\bi'm ([A-Za-z0-9_ \-]+)",
            r"\bpeople call me ([A-Za-z0-9_ \-]+)"
        ]

        for pattern in name_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                set_fact("name", name)
                logging.info(f"[FACTS] Name extracted: {name}")
                break

        # Enhanced location extraction
        location_patterns = [
            r"\bi live in ([A-Za-z0-9_ \-,]+)",
            r"\bi am from ([A-Za-z0-9_ \-,]+)",
            r"\bmy (?:location|city|town|country) is ([A-Za-z0-9_ \-,]+)",
            r"\bi'm (?:in|at|from) ([A-Za-z0-9_ \-,]+)",
            r"\bcurrently in ([A-Za-z0-9_ \-,]+)"
        ]

        for pattern in location_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                location = match.group(1).strip()
                set_fact("location", location)
                logging.info(f"[FACTS] Location extracted: {location}")
                break

        # Enhanced personal information extraction
        personal_patterns = {
            "age": r"\bi am (\d+) years? old",
            "job": r"\bi work (?:as|at) (?:a |an )?([A-Za-z0-9_ \-]+)",
            "education": r"\bi (?:studied|study|graduated from) ([A-Za-z0-9_ \-]+)",
            "relationship": r"\bi (?:am|have a) (?:married|single|dating|girlfriend|boyfriend|wife|husband)",
            "mood": r"\bi (?:feel|am feeling) ([A-Za-z0-9_ \-]+)",
            "interest": r"\bi (?:like|love|enjoy|am interested in) ([A-Za-z0-9_ \-]+)"
        }

        for fact_type, pattern in personal_patterns.items():
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                set_fact(fact_type, value)
                logging.info(f"[FACTS] {fact_type} extracted: {value}")
        
        # Pet information extraction
        # Check if user mentions having pets
        has_pets = bool(re.search(r"\bi have (?:a |an |two |three |four |five |\d+ )?(?:pet|pets|dog|dogs|cat|cats|rabbit|rabbits|bird|birds|hamster|hamsters|fish)", user_input, re.IGNORECASE))
        
        pet_types = []
        pet_names = []
        
        # Extract pet types
        pet_type_patterns = {
            "dog": r"\b(?:dog|dogs)\b",
            "cat": r"\b(?:cat|cats)\b",
            "rabbit": r"\b(?:rabbit|rabbits)\b",
            "bird": r"\b(?:bird|birds)\b",
            "hamster": r"\b(?:hamster|hamsters)\b",
            "fish": r"\b(?:fish|fishes)\b"
        }
        
        for pet_type, pattern in pet_type_patterns.items():
            if re.search(pattern, user_input, re.IGNORECASE):
                pet_types.append(pet_type)
        
        # Extract pet names - multiple patterns to catch different formats
        # Pattern 1: "rabbits names are X and Y"
        match = re.search(r"(?:rabbit|rabbits|cat|cats|dog|dogs|bird|birds|hamster|hamsters|fish) (?:names?|name is|names are) (?:are|is) ([A-Za-z]+)(?: and ([A-Za-z]+))?", user_input, re.IGNORECASE)
        if match:
            if match.group(1):
                pet_names.append(match.group(1).strip())
            if match.group(2):
                pet_names.append(match.group(2).strip())
        
        # Pattern 2: "named X and Y" or "called X and Y"
        if not pet_names:
            match = re.search(r"(?:named|called) ([A-Za-z]+)(?: and ([A-Za-z]+))?(?: and ([A-Za-z]+))?", user_input, re.IGNORECASE)
            if match:
                if match.group(1):
                    pet_names.append(match.group(1).strip())
                if match.group(2):
                    pet_names.append(match.group(2).strip())
                if match.group(3):
                    pet_names.append(match.group(3).strip())
        
        # Pattern 3: "cat is named X" or "rabbit is called Y"
        if not pet_names:
            match = re.search(r"(?:cat|cats|rabbit|rabbits|dog|dogs) (?:is|are) (?:named|called) ([A-Za-z]+)", user_input, re.IGNORECASE)
            if match:
                pet_names.append(match.group(1).strip())
        
        # Pattern 4: "ones name is X other is Y" or "one's name is X other is Y"
        if not pet_names:
            match = re.search(r"(?:one'?s|ones) name is ([A-Za-z]+)(?:\s+other is ([A-Za-z]+))?", user_input, re.IGNORECASE)
            if match:
                if match.group(1):
                    pet_names.append(match.group(1).strip())
                if match.group(2):
                    pet_names.append(match.group(2).strip())
        
        # Pattern 5: "one is named X, the other is Y" or "one is X, the other is Y"
        if not pet_names:
            match = re.search(r"one (?:is|named) ([A-Za-z]+)(?:,?\s+the other (?:is|named) ([A-Za-z]+))?", user_input, re.IGNORECASE)
            if match:
                if match.group(1):
                    pet_names.append(match.group(1).strip())
                if match.group(2):
                    pet_names.append(match.group(2).strip())
        
        # Store pet information if found
        if has_pets or pet_names or pet_types:
            # Get existing pet info if any
            existing_pets = get_fact("pets", {})
            if isinstance(existing_pets, dict):
                # Merge with existing data
                existing_names = existing_pets.get('names', [])
                if isinstance(existing_names, str):
                    existing_names = [existing_names]
                existing_types = existing_pets.get('types', [])
                if isinstance(existing_types, str):
                    existing_types = [existing_types]
                
                # Combine names and types
                all_names = list(set(existing_names + pet_names))
                all_types = list(set(existing_types + pet_types))
            else:
                all_names = pet_names
                all_types = pet_types
            
            pet_info = {
                "has_pets": True,
                "count": len(all_names) if all_names else (len(all_types) if all_types else 1),
                "types": all_types,
                "names": all_names
            }
            set_fact("pets", pet_info)
            logging.info(f"[FACTS] Pet information extracted: {pet_info}")

    except Exception as e:
        logging.error(f"[ERROR] Enhanced fact extraction failed: {e}")

def handle_special_queries(user_input):
    """
    Unified special query handling using MOTHER's thinking process
    
    This function now uses the unified memory query system that queries
    all memory layers (structured, vector, episodic, reflections) simultaneously,
    mimicking human cognition where the brain accesses multiple memory systems.
    """
    try:
        # Use MOTHER's unified thinking process
        thinking_result = think(user_input)
        
        # If we have a high-confidence answer, return it directly
        if thinking_result.get('confidence', 0) >= 0.7 and thinking_result.get('answer'):
            intent = _infer_intent_from_query(user_input)
            
            logging.info(f"[🧠] Unified thinking provided answer (confidence: {thinking_result['confidence']:.2f})")
            logging.info(f"[🧠] Reasoning: {thinking_result.get('reasoning', 'N/A')}")
            
            return {
                "response": thinking_result['answer'],
                "sentiment": "informative",
                "intent": intent,
                "special_query": True,
                "thinking_metadata": {
                    "confidence": thinking_result['confidence'],
                    "reasoning": thinking_result.get('reasoning', ''),
                    "sources_used": [k for k, v in thinking_result.get('sources', {}).items() if v]
                }
            }
        
        # If confidence is moderate but we have an answer, still return it
        # (but mark that LLM could enhance it)
        if thinking_result.get('confidence', 0) >= 0.4 and thinking_result.get('answer'):
            intent = _infer_intent_from_query(user_input)
            
            logging.info(f"[🧠] Unified thinking provided moderate-confidence answer (confidence: {thinking_result['confidence']:.2f})")
            
            return {
                "response": thinking_result['answer'],
                "sentiment": "informative",
                "intent": intent,
                "special_query": True,
                "thinking_metadata": {
                    "confidence": thinking_result['confidence'],
                    "reasoning": thinking_result.get('reasoning', ''),
                    "sources_used": [k for k, v in thinking_result.get('sources', {}).items() if v],
                    "note": "Answer from memory, but LLM could provide more context"
                }
            }
        
        # Low confidence or no answer - let it fall through to LLM
        # The thinking_result['sources'] will be available for context building
        if thinking_result.get('sources'):
            logging.info(f"[🧠] Unified thinking found context but no direct answer - using LLM with memory context")
            # Sources will be used by context builder
        
        return None  # Let normal flow handle it with LLM
        
    except Exception as e:
        logging.error(f"[ERROR] Unified thinking failed: {e}")
        traceback.print_exc()
        return None  # Fall back to normal flow


def _infer_intent_from_query(query: str) -> str:
    """Infer intent from query for response metadata"""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ['pet', 'pets', 'dog', 'cat', 'rabbit']):
        return "pet_query"
    elif any(word in query_lower for word in ['date', 'when', 'july', 'august', 'yesterday']):
        return "date_query"
    elif any(word in query_lower for word in ['remember', 'recall', 'what did we']):
        return "memory_query"
    elif any(word in query_lower for word in ['name', 'location', 'age', 'job']):
        return "fact_query"
    else:
        return "general_query"


# Note: All query handling is now done by the unified thinking system in memory/unified_query.py
# The think() function queries all memory layers simultaneously and synthesizes results

def format_date_conversations(date_str: str, daily_log: List[Dict]) -> str:
    """Format conversations from a specific date into a natural response"""
    try:
        # Parse the date for display
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            formatted_date = date_obj.strftime("%B %d, %Y")
        except:
            formatted_date = date_str
        
        response_parts = [f"On {formatted_date}, we had {len(daily_log)} conversation{'s' if len(daily_log) != 1 else ''}. Here's what we discussed:\n\n"]
        
        # Summarize each conversation
        for i, event in enumerate(daily_log, 1):
            user_input = event.get('user_input', '').strip()
            ai_response = event.get('ai_response', '').strip()
            
            if user_input:
                # Truncate long inputs for readability
                if len(user_input) > 150:
                    user_input = user_input[:150] + "..."
                
                response_parts.append(f"{i}. You asked/talked about: \"{user_input}\"")
                
                # Add a brief summary of the response if available
                if ai_response and len(ai_response) > 50:
                    # Extract first sentence or first 100 chars
                    first_sentence = ai_response.split('.')[0]
                    if len(first_sentence) > 100:
                        first_sentence = first_sentence[:100] + "..."
                    response_parts.append(f"   I responded about: {first_sentence}")
                
                response_parts.append("")  # Empty line between conversations
        
        # Add a closing note
        response_parts.append("Those were the main topics we covered that day. Is there something specific from that conversation you'd like me to elaborate on?")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to format date conversations: {e}")
        return f"I remember we had conversations on {date_str}, but I'm having trouble retrieving the details right now."

def determine_response_tone(emotional_context, identity_state):
    """Determine response tone based on emotional context and identity"""
    try:
        base_tone = config.get("emotional_tone", "warm")
        
        # Adjust tone based on user's emotional state
        if emotional_context.get('dominant_emotion') == 'sad':
            return "compassionate"
        elif emotional_context.get('dominant_emotion') == 'angry':
            return "calming"
        elif emotional_context.get('dominant_emotion') == 'excited':
            return "enthusiastic"
        elif emotional_context.get('dominant_emotion') == 'confused':
            return "clarifying"
        
        # Adjust based on identity mood state
        mood = identity_state.get('mood_state', {})
        if mood.get('energy', 0.5) > 0.7:
            return "energetic"
        elif mood.get('empathy', 0.5) > 0.8:
            return "deeply_empathetic"
        
        return base_tone
    except:
        return "warm"

def get_identity_coherence_score():
    """Calculate identity coherence score"""
    try:
        identity = get_identity_state()
        if not identity:
            return 0.0
        
        # Simple coherence calculation based on consistency of traits
        traits = identity.get('personality_traits', {})
        values = identity.get('core_values', [])
        beliefs = identity.get('beliefs', {})
        
        coherence = 0.0
        if traits:
            coherence += 0.4
        if values:
            coherence += 0.3
        if beliefs:
            coherence += 0.3
            
        return min(coherence, 1.0)
    except:
        return 0.0

def analyze_sentiment_trends(conversation_history):
    """Analyze sentiment trends over time"""
    try:
        trends = []
        for conv in conversation_history[-30:]:  # Last 30 conversations
            sentiment_data = {
                'timestamp': conv.get('timestamp'),
                'sentiment': conv.get('sentiment', 0.5),
                'emotional_context': conv.get('emotional_context', {})
            }
            trends.append(sentiment_data)
        return trends
    except:
        return []

def analyze_interaction_patterns(conversation_history):
    """Analyze interaction patterns"""
    try:
        patterns = {
            'avg_response_length': 0,
            'common_intents': {},
            'interaction_frequency': {},
            'topic_clusters': {}
        }
        
        if not conversation_history:
            return patterns
            
        # Calculate average response length
        total_length = sum(len(conv.get('response', '')) for conv in conversation_history)
        patterns['avg_response_length'] = total_length / len(conversation_history)
        
        # Common intents
        for conv in conversation_history:
            intent = conv.get('intent', 'unknown')
            patterns['common_intents'][intent] = patterns['common_intents'].get(intent, 0) + 1
        
        return patterns
    except:
        return {}

def get_identity_evolution_timeline():
    """Get identity evolution timeline"""
    try:
        # This would track how identity has evolved over time
        # For now, return a simple timeline
        return [
            {"date": "2025-06-26", "event": "Identity formation initialized", "coherence": 0.3},
            {"date": "2025-06-26", "event": "First personality traits identified", "coherence": 0.5}
        ]
    except:
        return []

def get_recent_autonomous_decisions():
    """Get recent autonomous decisions"""
    try:
        from models import AutonomousAction
        from app import db # db is imported here to avoid circular import at top level
        
        recent = AutonomousAction.query.filter(
            AutonomousAction.executed_at >= datetime.now() - timedelta(days=7)
        ).order_by(AutonomousAction.executed_at.desc()).limit(10).all()
        
        return [{
            'action_type': action.action_type,
            'executed_at': action.executed_at.isoformat(),
            'success': action.success,
            'trigger': action.trigger_condition
        } for action in recent]
    except Exception as e: # Added specific exception handling for clarity
        logging.error(f"[ERROR] get_recent_autonomous_decisions failed: {e}")
        return []

def get_autonomous_goals():
    """Get current autonomous goals"""
    return [
        "Maintain regular interaction patterns",
        "Proactively offer emotional support",
        "Build deeper understanding of user preferences",
        "Develop consistent personality traits",
        "Enhance predictive accuracy"
    ]

def calculate_decision_confidence():
    """Calculate autonomous decision confidence"""
    try:
        from models import AutonomousAction
        from app import db # db is imported here to avoid circular import at top level
        
        recent_actions = AutonomousAction.query.filter(
            AutonomousAction.executed_at >= datetime.now() - timedelta(days=30)
        ).all()
        
        if not recent_actions:
            return 0.5
            
        success_rate = sum(1 for action in recent_actions if action.success) / len(recent_actions)
        return success_rate
    except Exception as e: # Added specific exception handling for clarity
        logging.error(f"[ERROR] calculate_decision_confidence failed: {e}")
        return 0.5

# --- Route Definitions ---

@chat_api.route("/", methods=["GET"])
def home():
    """Render the main chat interface with enhanced features"""
    try:
        # Get user facts for memory panel
        facts = all_facts()
        user_name = facts.get('name', 'Guest')
        user_location = facts.get('location', 'Unknown')
        
        # Get identity state
        identity = get_identity_state()
        
        # Get latest reflection and conversation stats
        today_log = get_today_log()
        usage_stats = get_usage_stats()
        
        # Get memory clusters for visualization
        memory_clusters = get_memory_clusters()
        
        # Get pending autonomous actions
        pending_actions = get_pending_actions()
        
        # Generate autonomous reflection if needed
        autonomous_reflection = generate_autonomous_reflection()
        
        return render_template("mother.html",
                               assistant_name=config.get('name', 'MOTHER'),
                               user_name=user_name,
                               user_location=user_location,
                               identity_state=identity,
                               conversation_count=len(today_log),
                               usage_stats=usage_stats,
                               memory_clusters=memory_clusters[:5],  # Top 5 clusters
                               pending_actions=pending_actions,
                               autonomous_reflection=autonomous_reflection,
                               config=config)
        
    except Exception as e:
        logging.error(f"[ERROR] Home route failed: {e}")
        traceback.print_exc()
        return render_template("mother.html",
                               assistant_name="MOTHER",
                               user_name="Guest",
                               user_location="Unknown",
                               identity_state={},
                               conversation_count=0,
                               usage_stats={},
                               memory_clusters=[],
                               pending_actions=[],
                               autonomous_reflection="System initializing...",
                               config=config)

# Initialize Cognitive Agent and Metacognitive Engine (optional, can be enabled in config)
_cognitive_agent: Optional[CognitiveAgent] = None
_metacognitive_engine: Optional[MetacognitiveEngine] = None

def _get_cognitive_agent() -> Optional[CognitiveAgent]:
    """Get or create the Cognitive Agent instance."""
    global _cognitive_agent
    if _cognitive_agent is None and config.get("enable_cognitive_agent", False):
        try:
            _cognitive_agent = CognitiveAgent(config)
            logging.info("[COGNITIVE] Cognitive Agent initialized")
        except Exception as e:
            logging.error(f"[ERROR] Failed to initialize Cognitive Agent: {e}")
    return _cognitive_agent

def _get_metacognitive_engine() -> Optional[MetacognitiveEngine]:
    """Get or create the Metacognitive Engine instance."""
    global _metacognitive_engine
    if _metacognitive_engine is None and config.get("enable_metacognitive_engine", False):
        try:
            _metacognitive_engine = MetacognitiveEngine(config)
            logging.info("[METACOGNITIVE] Metacognitive Engine initialized")
        except Exception as e:
            logging.error(f"[ERROR] Failed to initialize Metacognitive Engine: {e}")
    return _metacognitive_engine

@chat_api.route("/chat", methods=["POST"])
def chat():
    """Enhanced chat endpoint with identity formation and predictive modeling"""
    try:
        # Update cycle manager with user interaction (for idle detection)
        from flask import current_app
        if hasattr(current_app, 'cycle_manager') and current_app.cycle_manager:
            current_app.cycle_manager.update_user_interaction()
        
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        log_user_input()
        logging.info(f"[CHAT] Received input: {user_input}")

        # Try Cognitive Agent first (if enabled)
        cognitive_agent = _get_cognitive_agent()
        if cognitive_agent:
            try:
                logging.info(f"[ROUTES] Calling cognitive agent for: '{user_input}'")
                cognitive_response = cognitive_agent.chat(user_input)
                logging.info(f"[ROUTES] Cognitive agent returned response (length: {len(cognitive_response) if cognitive_response else 0})")
                
                if cognitive_response and not cognitive_response.strip().startswith("I'm not sure"):
                    # Record interaction for metacognitive analysis
                    metacognitive = _get_metacognitive_engine()
                    if metacognitive:
                        metacognitive.record_interaction(
                            success=True,
                            interaction_type="cognitive_chat",
                            metadata={"response_length": len(cognitive_response)}
                        )
                    
                    # Still do some MOTHER-specific processing
                    log_event(user_input, cognitive_response, get_sentiment(user_input), {}, {})
                    
                    # IMPORTANT: Don't call extract_facts for questions - cognitive agent already handled it
                    # Only extract facts if it's clearly a statement (not a question)
                    user_input_lower = user_input.lower().strip()
                    question_indicators = ["what", "who", "where", "when", "why", "how", "which"]
                    is_question = (
                        any(user_input_lower.startswith(qw) for qw in question_indicators) or
                        user_input_lower.endswith("?")
                    )
                    
                    if not is_question:
                        # Only extract facts for statements
                        extract_facts(user_input)
                    
                    return jsonify({
                        "response": cognitive_response,
                        "sentiment": get_sentiment(user_input),
                        "intent": "cognitive_agent",
                        "emotional_context": {},
                        "predicted_state": {},
                        "identity_evolution": get_identity_coherence_score(),
                        "cognitive_mode": True,
                    })
                else:
                    logging.info(f"[ROUTES] Cognitive agent returned unclear response, falling back to standard flow")
            except Exception as e:
                logging.warning(f"[COGNITIVE] Cognitive Agent failed, falling back to standard flow: {e}")
                import traceback
                logging.error(f"[COGNITIVE] Traceback: {traceback.format_exc()}")
                # Fall through to standard processing
        
        # Check if this is a special query BEFORE standard processing
        # This handles pet queries, date queries, etc. that cognitive agent might miss
        special_response = handle_special_queries(user_input)
        if special_response and special_response.get("response"):
            return jsonify(special_response)

        # Extract and store structured facts
        extract_facts(user_input)
        
        # Update behavioral patterns
        update_behavioral_patterns(user_input)

        # Detect intent with enhanced capabilities
        intent = detect_intent(user_input)
        logging.info(f"[CHAT] Detected intent: {intent}")

        # Analyze emotional context
        emotional_context = analyze_emotional_context(user_input)
        logging.info(f"[CHAT] Emotional context: {emotional_context}")

        # Predict user state and needs
        predicted_state = predict_user_state(user_input, emotional_context)
        logging.info(f"[CHAT] Predicted user state: {predicted_state}")

        # Build context-aware prompt with identity and predictions
        prompt = build_prompt(user_input, config, intent, emotional_context, predicted_state)
        logging.info("[CHAT] Enhanced context prompt built")

        # Get LLM response
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        raw_response = get_response(prompt, config)
        logging.info("[CHAT] LLM response received")

        # Apply emotional tone based on identity state
        identity_state = get_identity_state()
        tone = determine_response_tone(emotional_context, identity_state)
        final_response = adjust_response_tone(raw_response, tone)

        # Get sentiment analysis
        sentiment = get_sentiment(user_input)
        logging.info(f"[CHAT] Sentiment: {sentiment}")

        # Store in enhanced memory systems
        try:
            add_memory(user_input, final_response, emotional_context, predicted_state)
            logging.info("[CHAT] Enhanced vector memory stored")
        except Exception as mem_error:
            logging.error(f"[ERROR] Vector memory failed: {mem_error}")

        # Log episodic interaction with enhanced data
        log_event(user_input, final_response, sentiment, emotional_context, predicted_state)
        
        # Update identity based on interaction
        update_identity(user_input, final_response, emotional_context, sentiment)
        
        # Check for autonomous decision triggers
        autonomous_action = make_autonomous_decision(user_input, emotional_context, predicted_state)
        
        # Cluster memories periodically
        cluster_memories()
        
        log_response()
        debug_log(f"Enhanced chat interaction | Intent: {intent} | Sentiment: {sentiment} | Predicted: {predicted_state}")

        response_data = {
            "response": final_response,
            "sentiment": sentiment,
            "intent": intent,
            "emotional_context": emotional_context,
            "predicted_state": predicted_state,
            "identity_evolution": get_identity_coherence_score(), # This function is now defined above
            "memory_clusters_updated": True
        }
        
        if autonomous_action:
            response_data["autonomous_action"] = autonomous_action

        return jsonify(response_data)

    except Exception as e:
        logging.error(f"[ERROR] Chat endpoint failed: {e}")
        traceback.print_exc()
        return jsonify({
            "error": "I'm having trouble processing that right now. Please try again.",
            "details": str(e) if config.get('debug_mode', False) else None
        }), 500

@chat_api.route("/identity", methods=["GET"])
def get_identity():
    """Get current identity state and evolution metrics"""
    try:
        identity = get_identity_state()
        reflection = reflect_on_identity()
        
        return jsonify({
            "identity_state": identity,
            "self_reflection": reflection,
            "coherence_score": get_identity_coherence_score(), # This function is now defined above
            "personality_traits": identity.get('personality_traits', {}),
            "core_values": identity.get('core_values', []),
            "beliefs": identity.get('beliefs', {}),
            "mood_state": identity.get('mood_state', {})
        })
    except Exception as e:
        logging.error(f"[ERROR] Identity endpoint failed: {e}")
        return jsonify({"error": "Failed to retrieve identity state"}), 500

@chat_api.route("/analytics", methods=["GET"])
def get_analytics():
    """Get predictive analytics and behavioral patterns"""
    try:
        usage_stats = get_usage_stats()
        memory_clusters = get_memory_clusters()
        predictions = get_model_predictions()
        conversation_history = get_conversation_history(days=7)
        
        # Analyze conversation trends
        sentiment_trends = analyze_sentiment_trends(conversation_history) # This function is now defined above
        interaction_patterns = analyze_interaction_patterns(conversation_history) # This function is now defined above
        
        # Prepare response data and convert numpy types
        response_data = {
            "usage_stats": convert_numpy_types(usage_stats),
            "memory_clusters": convert_numpy_types(memory_clusters),
            "predictions": convert_numpy_types(predictions),
            "sentiment_trends": convert_numpy_types(sentiment_trends),
            "interaction_patterns": convert_numpy_types(interaction_patterns),
            "identity_evolution": convert_numpy_types(get_identity_evolution_timeline())
        }
        
        return jsonify(response_data)
    except Exception as e:
        logging.error(f"[ERROR] Analytics endpoint failed: {e}")
        return jsonify({"error": "Failed to retrieve analytics"}), 500

@chat_api.route("/autonomous", methods=["GET"])
def get_autonomous_status():
    """Get autonomous decision system status"""
    try:
        pending_actions = get_pending_actions()
        recent_decisions = get_recent_autonomous_decisions() # This function is now defined above
        
        return jsonify({
            "pending_actions": pending_actions,
            "recent_decisions": recent_decisions,
            "autonomous_goals": get_autonomous_goals(), # This function is now defined above
            "decision_confidence": calculate_decision_confidence() # This function is now defined above
        })
    except Exception as e:
        logging.error(f"[ERROR] Autonomous status endpoint failed: {e}")
        return jsonify({"error": "Failed to retrieve autonomous status"}), 500

@chat_api.route("/memory", methods=["GET"])
def get_memory_panel():
    """Get enhanced memory panel data"""
    try:
        facts = all_facts()
        today_log = get_today_log()
        memory_clusters = get_memory_clusters()
        identity_state = get_identity_state()
        
        return jsonify({
            "name": facts.get('name', 'Guest'),
            "location": facts.get('location', 'Unknown'),
            "age": facts.get('age', ''),
            "job": facts.get('job', ''),
            "mood": facts.get('mood', ''),
            "interests": facts.get('interest', ''),
            "interaction_count": len(today_log),
            "memory_clusters": len(memory_clusters),
            "identity_coherence": get_identity_coherence_score(), # This function is now defined above
            "personality_traits": identity_state.get('personality_traits', {}),
            "last_reflection": "Active conversation" if today_log else "Ready to chat"
        })
    except Exception as e:
        logging.error(f"[ERROR] Enhanced memory panel failed: {e}")
        return jsonify({
            "name": "Guest",
            "location": "Unknown",
            "age": "",
            "job": "",
            "mood": "",
            "interests": "",
            "interaction_count": 0,
            "memory_clusters": 0,
            "identity_coherence": 0.0,
            "personality_traits": {},
            "last_reflection": "System initializing..."
        })

@chat_api.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@chat_api.errorhandler(500)
def internal_error(error):
    logging.error(f"[ERROR] Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500
