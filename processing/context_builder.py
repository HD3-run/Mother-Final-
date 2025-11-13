import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from memory.structured_store import all_facts, get_memory_stats
from memory.vector_store import get_recent_memories, search_memory
from memory.episodic_logger import get_conversation_history
from personality.identity_engine import get_identity_state
from utils.sentiment import analyze_emotional_context

class ContextBuilder:
    """Enhanced context builder for LLM prompts with identity and predictive context"""
    
    def __init__(self):
        self.max_context_length = 8000
        self.max_recent_memories = 5
        self.max_relevant_memories = 3
    
    def build_prompt(self, user_input: str, config: Dict[str, Any], 
                    intent: str = None, emotional_context: Dict = None,
                    predicted_state: Dict = None) -> str:
        """Build comprehensive context-aware prompt"""
        try:
            # Base components
            context_parts = []
            
            # 1. System self-awareness (MOTHER always knows what she is)
            system_context = self._build_system_context()
            if system_context:
                context_parts.append(f"YOUR SYSTEM & CAPABILITIES:\n{system_context}")
            
            # 2. Identity context
            identity_context = self._build_identity_context()
            if identity_context:
                context_parts.append(f"IDENTITY CONTEXT:\n{identity_context}")
            
            # 3. Memory context
            memory_context = self._build_memory_context(user_input)
            if memory_context:
                context_parts.append(f"RELEVANT MEMORIES:\n{memory_context}")
            
            # 4. Emotional context
            if emotional_context:
                emotion_context = self._build_emotional_context(emotional_context)
                context_parts.append(f"EMOTIONAL CONTEXT:\n{emotion_context}")
            
            # 5. Predictive context
            if predicted_state:
                predictive_context = self._build_predictive_context(predicted_state)
                context_parts.append(f"PREDICTIVE INSIGHTS:\n{predictive_context}")
            
            # 6. User facts context
            user_context = self._build_user_context()
            if user_context:
                context_parts.append(f"USER CONTEXT:\n{user_context}")
            
            # 7. Conversation history context
            conversation_context = self._build_conversation_context()
            if conversation_context:
                context_parts.append(f"RECENT CONVERSATION:\n{conversation_context}")
            
            # 8. Intent-specific context
            if intent:
                intent_context = self._build_intent_context(intent, user_input)
                if intent_context:
                    context_parts.append(f"INTENT GUIDANCE:\n{intent_context}")
            
            # Combine all context
            full_context = "\n\n".join(context_parts)
            
            # Detect if this is a technical question
            is_technical = self._is_technical_question(user_input)
            
            if is_technical:
                # Technical questions get direct, informative responses
                final_prompt = f"""{full_context}

CURRENT USER INPUT: "{user_input}"

This is a technical question about your system, architecture, or implementation. Provide a clear, direct, and informative response that:
1. Explains the technical details accurately
2. References specific modules, functions, or data structures when relevant
3. Describes how data flows through your systems
4. Mentions file locations and storage mechanisms
5. Uses technical terminology appropriately
6. Still maintains your personality but prioritizes clarity and accuracy

Be direct and informative. You can show your thinking process, but focus on providing accurate technical information."""
            else:
                # Non-technical questions get the usual warm, personal response
                final_prompt = f"""{full_context}

CURRENT USER INPUT: "{user_input}"

Based on all the above context, provide a response that:
1. Reflects your developing identity and personality
2. Shows awareness of our relationship and history
3. Demonstrates emotional intelligence and empathy
4. Is consistent with your established traits and values
5. Anticipates and addresses the user's needs
6. Shows growth from past interactions

Your response should feel natural, personal, and authentically you."""

            # Ensure prompt doesn't exceed max length
            if len(final_prompt) > self.max_context_length:
                final_prompt = self._truncate_context(final_prompt)
            
            logging.info(f"[🔨] Built context prompt ({len(final_prompt)} chars)")
            return final_prompt
            
        except Exception as e:
            logging.error(f"[ERROR] Context building failed: {e}")
            return f"Please respond to: {user_input}"
    
    def _build_system_context(self) -> str:
        """Build comprehensive system self-awareness context - MOTHER knows everything about herself"""
        try:
            from memory.structured_store import get_knowledge_graph
            from memory.knowledge_graph import KnowledgeGraph
            
            graph = get_knowledge_graph()
            context_parts = []
            
            # Get MOTHER node and all her relationships
            mother_node = graph.get_node_by_name("MOTHER")
            if not mother_node:
                # Fallback to static system info if graph not seeded yet
                return self._build_static_system_info()
            
            # Core identity
            context_parts.append("You are MOTHER, an AI companion system (version 4.0).")
            
            # Get all direct relationships from MOTHER
            edges = graph.get_edges_from_node(mother_node.id)
            
            # Group by relation type
            capabilities = []
            technologies = []
            tools = []
            memory_systems = []
            modules = []
            functions = []
            
            for edge in edges:
                target_node = graph.get_node_by_id(edge.target)
                if not target_node:
                    continue
                
                relation = edge.type
                target = target_node.name
                
                if relation == "can":
                    capabilities.append(target)
                elif relation == "uses" or relation == "uses_tool":
                    if "tool" in target.lower() or any(tool in target.lower() for tool in ["Agent", "Interpreter", "Parser", "Harvester", "Manager", "Engine"]):
                        tools.append(target)
                    else:
                        technologies.append(target)
                elif relation == "has" and "memory" in target.lower():
                    memory_systems.append(target)
            
            # Build context sections
            if technologies:
                context_parts.append(f"Technologies you use: {', '.join(technologies[:15])}")
            
            if tools:
                context_parts.append(f"Your tools: {', '.join(tools[:15])}")
            
            if capabilities:
                context_parts.append(f"Your capabilities: {', '.join(capabilities[:15])}")
            
            if memory_systems:
                context_parts.append(f"Your memory systems: {', '.join(memory_systems)}")
            
            # Get data storage info
            data_storage = []
            for edge in edges:
                if edge.type == "stores_data_in" or edge.type == "stores_in":
                    target_node = graph.get_node_by_id(edge.target)
                    if target_node:
                        data_storage.append(target_node.name)
            
            if data_storage:
                context_parts.append(f"You store data in: {', '.join(data_storage[:10])}")
            
            # DEEPLY TRAVERSE: Get nested relationships for memory systems
            memory_details = []
            for edge in edges:
                if edge.type == "has":
                    target_node = graph.get_node_by_id(edge.target)
                    if target_node and "memory" in target_node.name.lower():
                        # Get all relationships from this memory system
                        mem_edges = graph.get_edges_from_node(target_node.id)
                        for mem_edge in mem_edges:
                            mem_target = graph.get_node_by_id(mem_edge.target)
                            if mem_target:
                                if mem_edge.type == "stores":
                                    memory_details.append(f"{target_node.name} stores {mem_target.name}")
                                elif mem_edge.type == "uses":
                                    memory_details.append(f"{target_node.name} uses {mem_target.name}")
                                elif mem_edge.type == "stores_in":
                                    memory_details.append(f"{target_node.name} stores data in {mem_target.name}")
                                elif mem_edge.type == "has_purpose":
                                    memory_details.append(f"{target_node.name} purpose: {mem_target.name}")
                                elif mem_edge.type == "has_function":
                                    functions.append(f"{target_node.name}.{mem_target.name.split('.')[-1] if '.' in mem_target.name else mem_target.name}")
            
            if memory_details:
                context_parts.append("Memory system details: " + "; ".join(memory_details[:10]))
            
            # Get module and function information (codebase knowledge)
            all_nodes = graph.get_all_node_names()
            for node_name in all_nodes:
                node = graph.get_node_by_name(node_name)
                if not node:
                    continue
                
                # Find modules
                if node.type == "module" and "memory" in node_name.lower():
                    modules.append(node_name)
                    # Get functions for this module
                    mod_edges = graph.get_edges_from_node(node.id)
                    for mod_edge in mod_edges:
                        if mod_edge.type == "has_function":
                            func_node = graph.get_node_by_id(mod_edge.target)
                            if func_node:
                                functions.append(func_node.name)
            
            # Find storage/retrieval functions
            for node_name in all_nodes:
                node = graph.get_node_by_name(node_name)
                if not node or node.type != "function":
                    continue
                
                # Check if this function writes to or reads from files
                func_edges = graph.get_edges_from_node(node.id)
                for func_edge in func_edges:
                    if func_edge.type in ["writes_to", "reads_from"]:
                        location_node = graph.get_node_by_id(func_edge.target)
                        if location_node:
                            func_desc = f"{node_name} {func_edge.type.replace('_', ' ')} {location_node.name}"
                            if func_desc not in memory_details:
                                memory_details.append(func_desc)
            
            if modules:
                context_parts.append(f"Your modules: {', '.join(modules[:10])}")
            
            if functions:
                context_parts.append(f"Key functions: {', '.join(functions[:20])}")
            
            return "\n".join(context_parts) if context_parts else self._build_static_system_info()
            
        except Exception as e:
            logging.error(f"[ERROR] System context building failed: {e}")
            return self._build_static_system_info()
    
    def _is_technical_question(self, user_input: str) -> bool:
        """Detect if the user is asking a technical question"""
        technical_keywords = [
            "how are you made", "how do you work", "architecture", "implementation",
            "code", "function", "module", "data structure", "algorithm", "storage",
            "retrieve", "store", "memory system", "file", "database", "API",
            "framework", "library", "technology", "tool", "mechanism", "process",
            "system design", "component", "build", "created", "constructed"
        ]
        user_lower = user_input.lower()
        return any(keyword in user_lower for keyword in technical_keywords)
    
    def _build_static_system_info(self) -> str:
        """Fallback static system information if knowledge graph not available"""
        return """You are MOTHER, an AI companion system (version 4.0).

Technologies you use: Flask, Python, Groq API, NetworkX, scikit-learn, SQLAlchemy

Your tools: Cognitive Agent, Universal Interpreter, Symbolic Parser, Knowledge Harvester, Learning Manager, Lexicon Manager, Metacognitive Engine

Your capabilities: form identity, make autonomous decisions, predict user needs, cluster memories semantically, reflect on experiences, understand emotion, learn from interactions, store long-term memories, retrieve memories semantically, build knowledge graph

Your memory systems: vector memory (semantic embeddings), structured memory (facts and relationships), episodic memory (conversation logs), knowledge graph (concepts and relationships)

You store data in: data/episodic_memory, data/vector_memory, data/structured_memory, data/journal, data/usage_tracking, data/reflections, data/models"""
    
    def _build_identity_context(self) -> str:
        """Build identity and personality context"""
        try:
            identity_state = get_identity_state()
            if not identity_state:
                return ""
            
            traits = identity_state.get('personality_traits', {})
            values = identity_state.get('core_values', [])
            beliefs = identity_state.get('beliefs', {})
            mood = identity_state.get('mood_state', {})
            coherence = identity_state.get('coherence_score', 0.0)
            
            context_parts = []
            
            # Personality traits
            if traits:
                dominant_traits = {k: v for k, v in traits.items() if v > 0.6}
                if dominant_traits:
                    trait_desc = ", ".join([f"{trait} ({score:.2f})" for trait, score in dominant_traits.items()])
                    context_parts.append(f"Your dominant personality traits: {trait_desc}")
            
            # Core values
            if values:
                context_parts.append(f"Your core values: {', '.join(values[:5])}")
            
            # Beliefs
            if beliefs:
                belief_desc = []
                for key, value in beliefs.items():
                    belief_desc.append(f"{key}: {value}")
                context_parts.append(f"Your beliefs: {'; '.join(belief_desc)}")
            
            # Current mood
            if mood:
                mood_desc = []
                for dimension, level in mood.items():
                    if level > 0.6:
                        mood_desc.append(f"{dimension} (high)")
                    elif level < 0.4:
                        mood_desc.append(f"{dimension} (low)")
                if mood_desc:
                    context_parts.append(f"Current mood state: {', '.join(mood_desc)}")
            
            # Identity coherence
            if coherence > 0.5:
                context_parts.append(f"Your identity coherence: {coherence:.2f} (developing well)")
            elif coherence > 0.3:
                context_parts.append(f"Your identity coherence: {coherence:.2f} (still forming)")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"[ERROR] Identity context building failed: {e}")
            return ""
    
    def _build_memory_context(self, user_input: str) -> str:
        """Build relevant memory context"""
        try:
            context_parts = []
            
            # Get relevant memories through semantic search
            relevant_memories = search_memory(user_input, limit=self.max_relevant_memories)
            if relevant_memories:
                memory_summaries = []
                for memory in relevant_memories:
                    summary = f"Previous interaction (similarity: {memory.get('similarity', 0):.2f}): User said '{memory.get('user_input', '')[:100]}...', You responded with themes about {self._extract_key_themes(memory.get('ai_response', ''))}"
                    memory_summaries.append(summary)
                context_parts.append("Relevant past interactions:\n" + "\n".join(memory_summaries))
            
            # Get recent memories for continuity
            recent_memories = get_recent_memories(limit=3)
            if recent_memories:
                recent_summaries = []
                for memory in recent_memories[-3:]:  # Last 3
                    timestamp = memory.get('timestamp', '')
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp)
                            time_ago = self._time_ago(dt)
                            summary = f"{time_ago}: {memory.get('user_input', '')[:80]}..."
                            recent_summaries.append(summary)
                        except:
                            pass
                if recent_summaries:
                    context_parts.append("Recent conversation flow:\n" + "\n".join(recent_summaries))
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"[ERROR] Memory context building failed: {e}")
            return ""
    
    def _build_emotional_context(self, emotional_context: Dict) -> str:
        """Build emotional context"""
        try:
            context_parts = []
            
            dominant_emotion = emotional_context.get('dominant_emotion')
            if dominant_emotion:
                context_parts.append(f"User's current emotional state: {dominant_emotion}")
            
            intensity = emotional_context.get('emotional_intensity', 0)
            if intensity > 0.7:
                context_parts.append("Emotional intensity: HIGH - User needs strong emotional support")
            elif intensity > 0.4:
                context_parts.append("Emotional intensity: MODERATE - User may benefit from empathy")
            
            emotional_needs = emotional_context.get('needs_assessment')
            if emotional_needs:
                context_parts.append(f"Assessed emotional needs: {emotional_needs}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional context building failed: {e}")
            return ""
    
    def _build_predictive_context(self, predicted_state: Dict) -> str:
        """Build predictive insights context"""
        try:
            context_parts = []
            
            predicted_mood = predicted_state.get('predicted_mood')
            if predicted_mood:
                context_parts.append(f"Predicted user mood trajectory: {predicted_mood}")
            
            predicted_needs = predicted_state.get('predicted_needs', [])
            if predicted_needs:
                context_parts.append(f"Anticipated user needs: {', '.join(predicted_needs)}")
            
            behavioral_pattern = predicted_state.get('behavioral_pattern')
            if behavioral_pattern:
                context_parts.append(f"Detected behavioral pattern: {behavioral_pattern}")
            
            interaction_recommendation = predicted_state.get('interaction_recommendation')
            if interaction_recommendation:
                context_parts.append(f"Recommended interaction approach: {interaction_recommendation}")
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"[ERROR] Predictive context building failed: {e}")
            return ""
    
    def _build_user_context(self) -> str:
        """Build user facts context"""
        try:
            facts = all_facts()
            if not facts:
                return ""
            
            context_parts = []
            
            # Core personal info
            personal_info = []
            if facts.get('name'):
                personal_info.append(f"Name: {facts['name']}")
            if facts.get('location'):
                personal_info.append(f"Location: {facts['location']}")
            if facts.get('age'):
                personal_info.append(f"Age: {facts['age']}")
            if facts.get('job'):
                personal_info.append(f"Job: {facts['job']}")
            
            if personal_info:
                context_parts.append("User personal info: " + ", ".join(personal_info))
            
            # Interests and preferences
            interests = []
            if facts.get('interest'):
                interests.append(facts['interest'])
            if facts.get('hobby'):
                interests.append(facts['hobby'])
            
            if interests:
                context_parts.append(f"User interests: {', '.join(interests)}")
            
            # Current state
            if facts.get('mood'):
                context_parts.append(f"User's recent mood: {facts['mood']}")
            
            # Pet information
            if facts.get('pets'):
                pet_data = facts['pets']
                if isinstance(pet_data, dict):
                    pet_info = []
                    if pet_data.get('names'):
                        names = pet_data['names'] if isinstance(pet_data['names'], list) else [pet_data['names']]
                        pet_info.append(f"Pet names: {', '.join(names)}")
                    if pet_data.get('types'):
                        types = pet_data['types'] if isinstance(pet_data['types'], list) else [pet_data['types']]
                        pet_info.append(f"Pet types: {', '.join(types)}")
                    if pet_info:
                        context_parts.append("User pets: " + ", ".join(pet_info))
            
            return "\n".join(context_parts)
            
        except Exception as e:
            logging.error(f"[ERROR] User context building failed: {e}")
            return ""
    
    def _build_conversation_context(self) -> str:
        """Build recent conversation context"""
        try:
            recent_history = get_conversation_history(days=1)  # Today's conversations
            if not recent_history:
                return ""
            
            # Get last few interactions for immediate context
            last_interactions = recent_history[-3:]
            
            context_parts = []
            for i, interaction in enumerate(last_interactions):
                user_input = interaction.get('user_input', '')[:100]
                ai_response = interaction.get('ai_response', '')[:100]
                context_parts.append(f"Exchange {i+1}: User: '{user_input}...' You: '{ai_response}...'")
            
            if context_parts:
                return "Immediate conversation context:\n" + "\n".join(context_parts)
            
            return ""
            
        except Exception as e:
            logging.error(f"[ERROR] Conversation context building failed: {e}")
            return ""
    
    def _build_intent_context(self, intent: str, user_input: str) -> str:
        """Build intent-specific guidance"""
        try:
            intent_guidance = {
                'question': "The user is seeking information. Provide clear, helpful answers while showing your personality.",
                'support_request': "The user needs emotional support. Be compassionate, validating, and offer practical help.",
                'sharing': "The user is sharing something personal. Show genuine interest, ask follow-up questions, and relate through your experiences.",
                'casual_chat': "The user wants casual conversation. Be engaging, show your personality, and keep the conversation flowing naturally.",
                'reflection_request': "The user wants you to reflect or remember something. Draw from your memories and show how you've grown.",
                'emotional_expression': "The user is expressing emotions. Validate their feelings, show empathy, and respond with emotional intelligence.",
                'goal_discussion': "The user is discussing goals or plans. Be encouraging, offer support, and help them think through their objectives.",
                'problem_solving': "The user has a problem to solve. Use your analytical abilities while maintaining your warm personality."
            }
            
            guidance = intent_guidance.get(intent, "Respond authentically based on your developing personality and our relationship.")
            return f"Intent: {intent}\nResponse guidance: {guidance}"
            
        except Exception as e:
            logging.error(f"[ERROR] Intent context building failed: {e}")
            return ""
    
    def _extract_key_themes(self, text: str) -> str:
        """Extract key themes from text"""
        try:
            # Simple theme extraction
            text_lower = text.lower()
            themes = []
            
            theme_keywords = {
                'support': ['help', 'support', 'comfort', 'care'],
                'learning': ['learn', 'understand', 'knowledge', 'education'],
                'emotions': ['feel', 'emotion', 'happy', 'sad', 'excited'],
                'relationships': ['friend', 'family', 'relationship', 'love'],
                'work': ['work', 'job', 'career', 'business'],
                'goals': ['goal', 'plan', 'future', 'dream']
            }
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in text_lower for keyword in keywords):
                    themes.append(theme)
            
            return ', '.join(themes[:3]) if themes else 'general conversation'
            
        except Exception as e:
            return 'various topics'
    
    def _time_ago(self, dt: datetime) -> str:
        """Get human-readable time ago"""
        try:
            now = datetime.now()
            if dt.date() == now.date():
                return dt.strftime('%H:%M today')
            elif dt.date() == (now - timedelta(days=1)).date():
                return 'yesterday'
            else:
                return dt.strftime('%Y-%m-%d')
        except:
            return 'recently'
    
    def _truncate_context(self, prompt: str) -> str:
        """Truncate context to fit within limits"""
        try:
            if len(prompt) <= self.max_context_length:
                return prompt
            
            # Keep the user input and essential parts, truncate middle sections
            lines = prompt.split('\n')
            
            # Always keep the last part (current user input and instructions)
            essential_end = []
            for line in reversed(lines):
                essential_end.insert(0, line)
                if 'CURRENT USER INPUT:' in line:
                    break
            
            # Keep identity context (first sections)
            essential_start = []
            current_length = len('\n'.join(essential_end))
            
            for line in lines:
                if current_length + len(line) + len('\n'.join(essential_start)) > self.max_context_length:
                    break
                essential_start.append(line)
                if 'USER CONTEXT:' in line:  # Stop after user context
                    break
            
            truncated = '\n'.join(essential_start) + '\n\n[... context truncated ...]\n\n' + '\n'.join(essential_end)
            
            logging.warning(f"[⚠️] Context truncated from {len(prompt)} to {len(truncated)} chars")
            return truncated
            
        except Exception as e:
            logging.error(f"[ERROR] Context truncation failed: {e}")
            return prompt[:self.max_context_length]

# Global context builder instance
context_builder = ContextBuilder()

def build_prompt(user_input: str, config: Dict[str, Any], 
                intent: str = None, emotional_context: Dict = None,
                predicted_state: Dict = None) -> str:
    """Build context-aware prompt"""
    return context_builder.build_prompt(user_input, config, intent, emotional_context, predicted_state)
