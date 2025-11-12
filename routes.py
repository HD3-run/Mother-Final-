from flask import Blueprint, render_template, request, jsonify
import logging
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any
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
from memory.structured_store import set_fact, get_fact, all_facts # Removed init_db import
from memory.vector_store import add_memory, search_memory
from memory.episodic_logger import log_event, get_today_log, get_conversation_history, get_log_for_date, get_all_conversation_dates
from memory.semantic_clustering import cluster_memories, get_memory_clusters
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
    """Enhanced fact extraction with semantic understanding"""
    try:
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

    except Exception as e:
        logging.error(f"[ERROR] Enhanced fact extraction failed: {e}")

def handle_special_queries(user_input):
    """Enhanced special query handling with semantic search"""
    try:
        user_lower = user_input.lower()
        
        # Check for queries asking for all dates we've talked
        date_list_patterns = [
            r"(?:apart from|excluding|other than).*today.*(?:what|which).*dates?.*(?:did|do|have).*we.*talk",
            r"(?:what|which).*dates?.*(?:did|do|have).*we.*talk.*(?:on|about)",
            r"(?:list|tell me|show me).*dates?.*(?:we|you and i).*(?:talk|convers|chat)",
            r"(?:when|what dates).*(?:did|have).*we.*talk.*(?:before|apart from|excluding).*today",
            r"all.*dates?.*(?:we|you and i).*(?:talk|convers|chat)",
            # Follow-up queries asking for earlier dates
            r"before that\??",
            r"earlier\??",
            r"any.*before",
            r"dates?.*before"
        ]
        
        for pattern in date_list_patterns:
            if re.search(pattern, user_lower):
                # Get all conversation dates (excluding today if requested)
                exclude_today = any(word in user_lower for word in ['apart from', 'excluding', 'other than', 'before'])
                dates = get_all_conversation_dates(exclude_today=exclude_today)
                
                if dates:
                    # Format dates nicely
                    formatted_dates = []
                    for date_str in dates:
                        try:
                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                            formatted_dates.append(date_obj.strftime("%B %d, %Y"))
                        except:
                            formatted_dates.append(date_str)
                    
                    if len(formatted_dates) == 1:
                        response = f"We talked on {formatted_dates[0]}."
                        # If only one date and user asked "before that", clarify it's the earliest
                        if any(word in user_lower for word in ['before that', 'earlier', 'any before']):
                            response = f"{formatted_dates[0]} is the earliest date I have in my records. That's the first conversation I can recall before today."
                    elif len(formatted_dates) <= 5:
                        response = f"We've talked on {len(formatted_dates)} date{'s' if len(formatted_dates) > 1 else ''}: {', '.join(formatted_dates)}."
                        # If asking for earlier dates, mention the earliest
                        if any(word in user_lower for word in ['before that', 'earlier', 'any before']):
                            response += f" The earliest date in my records is {formatted_dates[0]}."
                    else:
                        # Show first few and last few
                        first_few = ', '.join(formatted_dates[:3])
                        last_few = ', '.join(formatted_dates[-2:])
                        response = f"We've talked on {len(formatted_dates)} dates. Some of them include: {first_few}... and more recently {last_few}."
                        # If asking for earlier dates, mention the earliest
                        if any(word in user_lower for word in ['before that', 'earlier', 'any before']):
                            response += f" The earliest date in my records is {formatted_dates[0]}."
                    
                    return {
                        "response": response,
                        "sentiment": "informative",
                        "intent": "date_list",
                        "special_query": True
                    }
                else:
                    return {
                        "response": "I don't have any recorded conversation dates yet. Today might be our first conversation, or the records might not be available.",
                        "sentiment": "informative",
                        "intent": "date_list",
                        "special_query": True
                    }
        
        # Enhanced date reflection queries - improved pattern matching
        date_patterns = [
            # Pattern: "what did we talk about on july 8th 2025" or "july 8, 2025"
            r"(?:what did we|what we|did we).*(?:talk about|discuss|say).*(?:on\s)?(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*)?(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s*,?\s*)?(\d{4})?",
            # Pattern: "what happened on july 8th"
            r"(?:what happened|remember|recall).*(?:on\s)?(\d{1,2})(?:st|nd|rd|th)?(?:\s+of)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s*,?\s*)?(\d{4})?",
            # Pattern: "7/8/2025" or "7-8-2025"
            r"(?:what did we|what we|did we).*(?:talk about|discuss).*(?:on\s)?(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})",
            # Pattern: "yesterday", "last week", "last month"
            r"(?:what did we|what we|did we).*(?:talk about|discuss).*(yesterday|last week|last month)",
            # Pattern: "our conversation on..."
            r"(?:our conversation|what we discussed).*(?:about|regarding|on)\s+(.+)"
        ]
        
        # Check for date queries first
        for pattern in date_patterns:
            match = re.search(pattern, user_lower)
            if match:
                date_str = None
                
                # Handle relative dates
                if "yesterday" in user_lower:
                    yesterday = datetime.now() - timedelta(days=1)
                    date_str = yesterday.strftime("%Y-%m-%d")
                elif "last week" in user_lower:
                    last_week = datetime.now() - timedelta(days=7)
                    date_str = last_week.strftime("%Y-%m-%d")
                elif "last month" in user_lower:
                    last_month = datetime.now() - timedelta(days=30)
                    date_str = last_month.strftime("%Y-%m-%d")
                else:
                    # Handle specific date formats
                    groups = match.groups()
                    
                    # Format: "july 8th 2025" or "july 8, 2025" (3 groups: day, month, year)
                    if len(groups) == 3 and groups[2] and groups[2].isdigit():
                        day = int(groups[0])
                        month = groups[1].capitalize()
                        year = int(groups[2])
                        try:
                            month_num = datetime.strptime(month, "%B").month
                            date_str = f"{year}-{month_num:02d}-{day:02d}"
                        except ValueError:
                            continue
                    # Format: "july 8th" without year (2 groups: day, month) - assume current year
                    elif len(groups) >= 2 and groups[0].isdigit():
                        day = int(groups[0])
                        month = groups[1].capitalize()
                        try:
                            month_num = datetime.strptime(month, "%B").month
                            year = datetime.now().year
                            date_str = f"{year}-{month_num:02d}-{day:02d}"
                        except ValueError:
                            continue
                    # Format: "7/8/2025" (3 groups: month, day, year)
                    elif len(groups) == 3 and all(g.isdigit() for g in groups if g):
                        month, day, year = groups
                        date_str = f"{year}-{int(month):02d}-{int(day):02d}"
                
                if date_str:
                    # Get actual conversations from that date
                    daily_log = get_log_for_date(date_str)
                    
                    if daily_log:
                        # Format conversations naturally
                        conversation_summary = format_date_conversations(date_str, daily_log)
                        return {
                            "response": conversation_summary,
                            "sentiment": "reflective",
                            "intent": "date_reflection",
                            "special_query": True
                        }
                    else:
                        # No conversations found for that date
                        return {
                            "response": f"I don't have any recorded conversations from {date_str}. We might not have talked that day, or the records might not be available.",
                            "sentiment": "informative",
                            "intent": "date_reflection",
                            "special_query": True
                        }

        user_lower = user_input.lower()
        
        # Enhanced memory search with semantic clustering
        memory_patterns = [
            r"(?:what did we|do you remember).*(?:talk about|discuss|say about)\s+(.+)",
            r"remember when we (?:talked about|discussed) (.+)",
            r"what do you know about (.+)",
            r"(?:tell me about|recall) (.+)",
            r"do you remember (.+)"
        ]

        for pattern in memory_patterns:
            match = re.search(pattern, user_input.lower())
            if match:
                topic = match.group(1).strip()
                memories = search_memory(topic, limit=5)
                clustered_memories = cluster_memories(memories)
                
                if memories:
                    memory_text = "\n".join([f"• {mem['content']}" for mem in memories])
                    cluster_info = f"\nI've organized {len(clustered_memories)} related memory clusters about this topic."
                    
                    return {
                        "response": f"Here's what I remember about {topic}:\n\n{memory_text}{cluster_info}",
                        "sentiment": "informative",
                        "intent": "memory_search",
                        "special_query": True,
                        "memory_clusters": clustered_memories
                    }

    except Exception as e:
        logging.error(f"[ERROR] Enhanced special query handling failed: {e}")
    
    return None

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

@chat_api.route("/chat", methods=["POST"])
def chat():
    """Enhanced chat endpoint with identity formation and predictive modeling"""
    try:
        user_input = request.json.get("message", "").strip()
        if not user_input:
            return jsonify({"error": "Empty message"}), 400

        log_user_input()
        logging.info(f"[CHAT] Received input: {user_input}")

        # Extract and store structured facts
        extract_facts(user_input)
        
        # Update behavioral patterns
        update_behavioral_patterns(user_input)

        # Handle special queries (date reflections, memory searches)
        special_response = handle_special_queries(user_input)
        if special_response:
            return jsonify(special_response)

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
