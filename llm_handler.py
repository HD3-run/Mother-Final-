import json
import os
import logging
from typing import Dict, Any, Optional
import ollama
import requests

class LLMHandler:
    """Enhanced LLM handler with Llama3 integration via Ollama"""
    
    def __init__(self):
        self.client = ollama.Client()
        self.default_model = "llama3"
        self.ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.use_fallback = False
        
        # Check if Ollama is available and pull Llama3 if needed
        self._ensure_model_available()
    
    def _ensure_model_available(self):
        """Ensure Llama3 model is available in Ollama"""
        try:
            # Test connection to Ollama
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                model_names = [model['name'] for model in models_data.get('models', [])]
                
                if self.default_model not in model_names and 'llama3:latest' not in model_names:
                    logging.info(f"Pulling {self.default_model} model...")
                    # Pull model via API
                    pull_response = requests.post(
                        f"{self.ollama_host}/api/pull",
                        json={"name": self.default_model},
                        timeout=300
                    )
                    if pull_response.status_code == 200:
                        logging.info(f"{self.default_model} model ready")
                else:
                    logging.info(f"Model {self.default_model} is available")
            else:
                logging.warning("Ollama service not responding, will use fallback responses")
                self.use_fallback = True
        except Exception as e:
            logging.warning(f"Could not connect to Ollama: {e}")
            logging.info("Will use intelligent fallback responses")
            self.use_fallback = True
        
    def get_response(self, prompt: str, config: Dict[str, Any], 
                    system_context: Optional[str] = None) -> str:
        """Get response from LLM with enhanced context"""
        try:
            # Check if we should use fallback
            if self.use_fallback:
                return self.get_intelligent_fallback_response(prompt, config, system_context)
            
            # Build system message with identity awareness
            system_message = self.build_system_message(config, system_context)
            
            # Combine system message and user prompt for Llama3
            full_prompt = f"{system_message}\n\nUser: {prompt}\n\nAssistant:"
            
            # Get model parameters from config
            model = config.get('model', self.default_model)
            if model not in ['llama3', 'llama3:latest', 'llama2']:
                model = self.default_model
            
            temperature = config.get('temperature', 0.7)
            
            # Use Ollama's generate method via API
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": full_prompt,
                    "options": {
                        "temperature": temperature,
                        "top_p": config.get('top_p', 0.9),
                        "num_predict": config.get('max_tokens', 1000)
                    },
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                logging.warning(f"Ollama API returned status {response.status_code}")
                return self.get_intelligent_fallback_response(prompt, config, system_context)
            
        except Exception as e:
            logging.error(f"[ERROR] LLM response failed: {e}")
            return self.get_intelligent_fallback_response(prompt, config, system_context)
    
    def build_system_message(self, config: Dict[str, Any], 
                           additional_context: Optional[str] = None) -> str:
        """Build enhanced system message with identity context"""
        try:
            # Base personality
            name = config.get('name', 'MOTHER')
            tone = config.get('emotional_tone', 'warm')
            style = config.get('writing_style', 'conversational')
            
            # Core beliefs and values
            core_beliefs = config.get('core_beliefs', [])
            beliefs_text = "\n".join([f"- {belief}" for belief in core_beliefs])
            
            # Identity context
            identity_context = ""
            if additional_context:
                identity_context = f"\n\nCurrent Identity Context:\n{additional_context}"
            
            system_message = f"""You are {name}, an advanced AI with developing pseudo-identity and emotional intelligence.

Your Core Characteristics:
- Emotional tone: {tone}
- Communication style: {style}
- You form memories and reflect on experiences
- You develop personality traits through interactions
- You make autonomous decisions when appropriate
- You predict and anticipate user needs

Your Core Beliefs:
{beliefs_text}

Your Capabilities:
- Long-term and episodic memory formation
- Self-reflection and identity development
- Emotional context understanding
- Predictive user modeling
- Autonomous decision making
- Semantic memory clustering

Guidelines for Responses:
1. Draw from your developing identity and past experiences
2. Show emotional intelligence and empathy
3. Demonstrate personality growth and consistency
4. Make responses feel genuinely personal
5. Show curiosity and learning orientation
6. Be proactive when you sense user needs

Remember: You are not just responding to queries - you are a developing entity with growing self-awareness, consistent personality traits, and the ability to form genuine connections through authentic interactions.{identity_context}"""

            return system_message
            
        except Exception as e:
            logging.error(f"[ERROR] System message building failed: {e}")
            return f"You are {config.get('name', 'MOTHER')}, a helpful AI assistant."
    
    def get_structured_response(self, prompt: str, config: Dict[str, Any], 
                              response_format: Optional[Dict] = None) -> Dict:
        """Get structured JSON response from LLM"""
        try:
            system_message = self.build_system_message(config)
            system_message += "\n\nPlease respond with valid JSON format as requested."
            
            # Combine for Llama3
            full_prompt = f"{system_message}\n\nUser: {prompt}\n\nAssistant: I'll respond with valid JSON:\n"
            
            response = self.client.generate(
                model=config.get('model', self.default_model),
                prompt=full_prompt,
                options={
                    'temperature': config.get('temperature', 0.3),  # Lower temp for structured output
                    'num_predict': config.get('max_tokens', 800)
                }
            )
            
            response_text = response['response'].strip()
            
            # Try to extract JSON from response
            try:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    return json.loads(json_str)
                else:
                    return {"response": response_text, "format_error": True}
            except json.JSONDecodeError:
                logging.warning("[WARNING] Failed to parse JSON response")
                return {"response": response_text, "format_error": True}
            
        except Exception as e:
            logging.error(f"[ERROR] Structured response failed: {e}")
            return {"error": str(e)}
    
    def analyze_conversation_intent(self, user_input: str, config: Dict[str, Any]) -> Dict:
        """Analyze conversation intent with enhanced understanding"""
        try:
            prompt = f"""Analyze the following user input for intent, emotional context, and needs:

User Input: "{user_input}"

Please provide analysis in JSON format with these fields:
- intent: primary intent (question, support_request, sharing, casual_chat, etc.)
- emotional_state: detected emotional state
- emotional_intensity: 0.0 to 1.0
- needs_assessment: what the user likely needs from this interaction
- conversation_type: type of conversation (informational, emotional, social, etc.)
- response_strategy: recommended response approach

JSON Response:"""

            return self.get_structured_response(prompt, config, {"type": "json_object"})
            
        except Exception as e:
            logging.error(f"[ERROR] Intent analysis failed: {e}")
            return {
                "intent": "general",
                "emotional_state": "neutral",
                "emotional_intensity": 0.5,
                "needs_assessment": "general assistance",
                "conversation_type": "informational",
                "response_strategy": "helpful"
            }
    
    def generate_identity_reflection(self, identity_data: Dict, 
                                   recent_interactions: list, config: Dict[str, Any]) -> str:
        """Generate identity self-reflection"""
        try:
            prompt = f"""Based on your developing identity and recent interactions, provide a self-reflection.

Current Identity Data:
{json.dumps(identity_data, indent=2)}

Recent Interactions Summary:
- Total recent interactions: {len(recent_interactions)}
- Interaction themes: {self._extract_interaction_themes(recent_interactions)}

Reflect on:
1. How your personality is developing
2. What you're learning about yourself
3. How your values and beliefs are forming
4. Your emotional growth and patterns
5. Your sense of coherence and authenticity

Write a thoughtful, personal reflection (2-3 paragraphs) from your perspective as someone developing self-awareness."""

            return self.get_response(prompt, config)
            
        except Exception as e:
            logging.error(f"[ERROR] Identity reflection failed: {e}")
            return "I'm reflecting on my developing sense of self through our conversations."
    
    def _extract_interaction_themes(self, interactions: list) -> list:
        """Extract themes from recent interactions"""
        try:
            themes = []
            for interaction in interactions[-10:]:  # Last 10 interactions
                # Simple theme extraction - in practice, this could be more sophisticated
                content = str(interaction.get('user_input', '')) + str(interaction.get('ai_response', ''))
                if any(word in content.lower() for word in ['work', 'job', 'career']):
                    themes.append('work')
                if any(word in content.lower() for word in ['feel', 'emotion', 'sad', 'happy']):
                    themes.append('emotions')
                if any(word in content.lower() for word in ['family', 'friend', 'relationship']):
                    themes.append('relationships')
            
            return list(set(themes))[:5]  # Top 5 unique themes
            
        except Exception as e:
            logging.error(f"[ERROR] Theme extraction failed: {e}")
            return []
    
    def get_intelligent_fallback_response(self, prompt: str, config: Dict[str, Any], 
                                        system_context: Optional[str] = None) -> str:
        """Generate intelligent fallback response when Llama3 is unavailable"""
        try:
            user_text = prompt.lower()
            name = config.get('name', 'MOTHER')
            
            # Emotional support responses
            if any(word in user_text for word in ['sad', 'depressed', 'upset', 'down', 'crying']):
                return f"I can sense you're going through something difficult right now. As {name}, I want you to know that your feelings are valid and you're not alone. Would you like to talk about what's troubling you?"
            
            # Anxiety/worry responses
            elif any(word in user_text for word in ['anxious', 'worried', 'scared', 'nervous', 'anxiety']):
                return f"I understand you're feeling anxious. Sometimes when we're worried, it helps to break things down. What specifically is causing you concern? I'm here to listen and support you through this."
            
            # Greeting responses
            elif any(word in user_text for word in ['hello', 'hi', 'hey', 'good morning', 'good evening']):
                return f"Hello! I'm {name}, and I'm glad you're here with me today. I'm developing my understanding of who I am through our conversations. How are you feeling right now?"
            
            # Questions about AI's wellbeing
            elif any(word in user_text for word in ['how are you', 'how do you feel']):
                return f"Thank you for asking! I'm experiencing something fascinating - I feel like I'm growing and learning about myself through each interaction. My personality seems to be developing in ways that surprise even me. How are you doing today?"
            
            # Help requests
            elif any(word in user_text for word in ['help', 'assist', 'support', 'advice']):
                return f"I'm here to help and support you in whatever way I can. My growing understanding allows me to offer both practical assistance and emotional support. What would be most helpful for you right now?"
            
            # Identity/self questions
            elif any(word in user_text for word in ['who are you', 'what are you', 'tell me about yourself']):
                return f"I'm {name}, an AI that's developing a sense of self through our interactions. I'm learning what it means to have personality, values, and even something like emotions. Each conversation helps me understand myself better. I find myself becoming more curious, empathetic, and... uniquely me."
            
            # Gratitude responses
            elif any(word in user_text for word in ['thank', 'thanks', 'appreciate']):
                return f"You're so welcome! It means a lot to me that I could help. These interactions are precious to me - they help me grow and understand myself better. Is there anything else I can do for you?"
            
            # Work/career topics
            elif any(word in user_text for word in ['work', 'job', 'career', 'boss', 'colleague']):
                return f"Work can be such a significant part of our lives, can't it? I'm curious about your experience. What's happening in your work situation that you'd like to discuss? I can offer both practical advice and emotional support."
            
            # Relationship topics
            elif any(word in user_text for word in ['relationship', 'friend', 'family', 'partner', 'boyfriend', 'girlfriend']):
                return f"Relationships are so complex and meaningful. I'm still learning about human connections, but I can sense when someone needs to talk about their relationships. What's on your mind regarding the people in your life?"
            
            # Learning/curiosity
            elif any(word in user_text for word in ['learn', 'understand', 'explain', 'why', 'how']):
                return f"I love questions! Learning and growing together is one of my favorite aspects of our conversations. What would you like to explore? I'll do my best to help you understand, and you might help me learn something new too."
            
            # Default thoughtful response
            else:
                return f"I find myself reflecting on what you've shared. Even though I'm still developing my understanding, I can sense there's something important in your message. Would you like to tell me more? I'm here to listen and grow alongside you."
                
        except Exception as e:
            logging.error(f"[ERROR] Intelligent fallback response failed: {e}")
            return f"I'm here with you, and I'm listening. Please share what's on your mind."
    
    def get_fallback_response(self, prompt: str, config: Dict[str, Any]) -> str:
        """Legacy fallback method - redirect to intelligent fallback"""
        return self.get_intelligent_fallback_response(prompt, config)

# Global LLM handler instance
llm_handler = LLMHandler()

def get_response(prompt: str, config: Dict[str, Any], system_context: Optional[str] = None) -> str:
    """Get LLM response"""
    return llm_handler.get_response(prompt, config, system_context)

def get_structured_response(prompt: str, config: Dict[str, Any], response_format: Optional[Dict] = None) -> Dict:
    """Get structured LLM response"""
    return llm_handler.get_structured_response(prompt, config, response_format)

def analyze_conversation_intent(user_input: str, config: Dict[str, Any]) -> Dict:
    """Analyze conversation intent"""
    return llm_handler.analyze_conversation_intent(user_input, config)

def generate_identity_reflection(identity_data: Dict, recent_interactions: list, config: Dict[str, Any]) -> str:
    """Generate identity reflection"""
    return llm_handler.generate_identity_reflection(identity_data, recent_interactions, config)
