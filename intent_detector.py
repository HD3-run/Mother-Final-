import re
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from collections import Counter
from processing.llm_handler import analyze_conversation_intent
from personality.loader import load_config

class IntentDetector:
    """Enhanced intent detection with pattern matching and LLM analysis"""
    
    def __init__(self):
        self.intent_patterns = {
            'question': [
                r'\?',
                r'\b(what|how|why|when|where|who|which)\b',
                r'\b(can you|could you|would you|will you)\b',
                r'\b(do you know|tell me|explain)\b'
            ],
            'support_request': [
                r'\b(help|support|assist|advice|guidance)\b',
                r'\b(i need|i want|i\'m struggling|i\'m having trouble)\b',
                r'\b(problem|issue|difficulty|challenge)\b',
                r'\b(don\'t know what to do|feeling lost|confused)\b'
            ],
            'emotional_expression': [
                r'\b(feel|feeling|felt|emotion|emotional)\b',
                r'\b(sad|happy|angry|excited|worried|stressed|anxious|depressed)\b',
                r'\b(i\'m|i am).*(sad|happy|angry|excited|worried|stressed|anxious|upset)\b',
                r'\b(having a hard time|going through|dealing with)\b'
            ],
            'sharing': [
                r'\b(tell you|share|happened|today|yesterday)\b',
                r'\b(guess what|you know what|something interesting)\b',
                r'\b(i did|i went|i saw|i met)\b',
                r'\b(story|experience|event)\b'
            ],
            'casual_chat': [
                r'\b(hi|hello|hey|good morning|good evening)\b',
                r'\b(how are you|how\'s it going|what\'s up)\b',
                r'\b(nice weather|how was your day)\b',
                r'\b(just saying hi|wanted to chat)\b'
            ],
            'reflection_request': [
                r'\b(remember|recall|think back|what did we)\b',
                r'\b(our conversation|we talked about|you said)\b',
                r'\b(reflect on|think about|consider)\b',
                r'\b(what do you think about|your thoughts on)\b'
            ],
            'goal_discussion': [
                r'\b(goal|plan|planning|future|dream|ambition)\b',
                r'\b(want to|going to|thinking about|considering)\b',
                r'\b(achieve|accomplish|work towards)\b',
                r'\b(next step|move forward|progress)\b'
            ],
            'problem_solving': [
                r'\b(solve|solution|figure out|work out)\b',
                r'\b(problem|issue|challenge|difficulty)\b',
                r'\b(how can i|what should i|what would you)\b',
                r'\b(options|alternatives|suggestions)\b'
            ],
            'learning_request': [
                r'\b(learn|teach|explain|understand|knowledge)\b',
                r'\b(how does|how to|what is|what are)\b',
                r'\b(tutorial|guide|instruction)\b',
                r'\b(i want to learn|help me understand)\b'
            ],
            'appreciation': [
                r'\b(thank|thanks|grateful|appreciate)\b',
                r'\b(you\'re helpful|you help me|you\'re great)\b',
                r'\b(i appreciate|i\'m grateful|thank you)\b'
            ]
        }
        
        self.emotional_indicators = {
            'positive': ['happy', 'excited', 'great', 'wonderful', 'amazing', 'good', 'fantastic'],
            'negative': ['sad', 'upset', 'angry', 'frustrated', 'worried', 'stressed', 'anxious', 'depressed'],
            'neutral': ['okay', 'fine', 'alright', 'normal', 'usual']
        }
        
        self.urgency_indicators = {
            'high': ['urgent', 'emergency', 'asap', 'immediately', 'right now', 'crisis'],
            'medium': ['soon', 'quickly', 'important', 'need help', 'struggling'],
            'low': ['when you can', 'no rush', 'eventually', 'sometime']
        }
    
    def detect_intent(self, user_input: str, use_llm: bool = True) -> str:
        """Detect user intent with enhanced analysis"""
        try:
            # Primary pattern-based detection
            primary_intent = self._pattern_based_detection(user_input)
            
            # Get additional context
            emotional_context = self._detect_emotional_context(user_input)
            urgency = self._detect_urgency(user_input)
            
            # Use LLM for enhanced analysis if available
            if use_llm:
                try:
                    config = load_config()
                    llm_analysis = analyze_conversation_intent(user_input, config)
                    
                    if llm_analysis and not llm_analysis.get('error'):
                        llm_intent = llm_analysis.get('intent', primary_intent)
                        
                        # Combine pattern-based and LLM results
                        final_intent = self._combine_intent_results(primary_intent, llm_intent, emotional_context)
                        
                        logging.info(f"[🎯] Intent detected: {final_intent} (pattern: {primary_intent}, LLM: {llm_intent})")
                        return final_intent
                        
                except Exception as e:
                    logging.warning(f"[WARNING] LLM intent analysis failed: {e}")
            
            # Adjust intent based on emotional context
            adjusted_intent = self._adjust_for_emotion(primary_intent, emotional_context)
            
            logging.info(f"[🎯] Intent detected: {adjusted_intent} (pattern-based with emotion adjustment)")
            return adjusted_intent
            
        except Exception as e:
            logging.error(f"[ERROR] Intent detection failed: {e}")
            return 'general'
    
    def _pattern_based_detection(self, user_input: str) -> str:
        """Detect intent using pattern matching"""
        try:
            user_text = user_input.lower().strip()
            intent_scores = {}
            
            # Score each intent based on pattern matches
            for intent, patterns in self.intent_patterns.items():
                score = 0
                for pattern in patterns:
                    matches = len(re.findall(pattern, user_text))
                    score += matches
                
                if score > 0:
                    intent_scores[intent] = score
            
            # Return highest scoring intent
            if intent_scores:
                return max(intent_scores, key=intent_scores.get)
            
            # Fallback to simple heuristics
            if '?' in user_input:
                return 'question'
            elif any(word in user_text for word in ['hi', 'hello', 'hey']):
                return 'casual_chat'
            elif any(word in user_text for word in ['help', 'support']):
                return 'support_request'
            else:
                return 'general'
                
        except Exception as e:
            logging.error(f"[ERROR] Pattern-based detection failed: {e}")
            return 'general'
    
    def _detect_emotional_context(self, user_input: str) -> Dict[str, Any]:
        """Detect emotional context in user input"""
        try:
            user_text = user_input.lower()
            emotional_context = {
                'dominant_emotion': 'neutral',
                'emotional_intensity': 0.5,
                'emotional_indicators': []
            }
            
            emotion_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
            
            # Count emotional indicators
            for emotion_type, indicators in self.emotional_indicators.items():
                for indicator in indicators:
                    if indicator in user_text:
                        emotion_scores[emotion_type] += 1
                        emotional_context['emotional_indicators'].append(indicator)
            
            # Determine dominant emotion
            if max(emotion_scores.values()) > 0:
                emotional_context['dominant_emotion'] = max(emotion_scores, key=emotion_scores.get)
            
            # Calculate emotional intensity
            total_indicators = sum(emotion_scores.values())
            if total_indicators > 0:
                # Intensity based on number of emotional words and their strength
                intensity = min(total_indicators * 0.3, 1.0)
                
                # Boost intensity for strong emotional words
                strong_emotions = ['devastated', 'thrilled', 'furious', 'ecstatic', 'terrified', 'overjoyed']
                if any(emotion in user_text for emotion in strong_emotions):
                    intensity = min(intensity + 0.3, 1.0)
                
                emotional_context['emotional_intensity'] = intensity
            
            return emotional_context
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional context detection failed: {e}")
            return {'dominant_emotion': 'neutral', 'emotional_intensity': 0.5, 'emotional_indicators': []}
    
    def _detect_urgency(self, user_input: str) -> str:
        """Detect urgency level in user input"""
        try:
            user_text = user_input.lower()
            
            for urgency_level, indicators in self.urgency_indicators.items():
                if any(indicator in user_text for indicator in indicators):
                    return urgency_level
            
            # Check for urgent punctuation patterns
            if '!!!' in user_input or user_input.count('!') > 2:
                return 'high'
            elif '!!' in user_input:
                return 'medium'
            
            return 'low'
            
        except Exception as e:
            logging.error(f"[ERROR] Urgency detection failed: {e}")
            return 'low'
    
    def _combine_intent_results(self, pattern_intent: str, llm_intent: str, 
                               emotional_context: Dict) -> str:
        """Combine pattern-based and LLM intent results"""
        try:
            # If both agree, use that intent
            if pattern_intent == llm_intent:
                return pattern_intent
            
            # Priority rules for disagreement
            priority_intents = ['support_request', 'emotional_expression', 'problem_solving']
            
            # If either detected a high-priority intent, use it
            if pattern_intent in priority_intents:
                return pattern_intent
            if llm_intent in priority_intents:
                return llm_intent
            
            # If emotional intensity is high, prioritize emotional expression
            if emotional_context.get('emotional_intensity', 0) > 0.7:
                if 'emotional' in llm_intent or 'support' in llm_intent:
                    return llm_intent
                if 'emotional' in pattern_intent or 'support' in pattern_intent:
                    return pattern_intent
            
            # Default to LLM result for ambiguous cases
            return llm_intent
            
        except Exception as e:
            logging.error(f"[ERROR] Intent combination failed: {e}")
            return pattern_intent
    
    def _adjust_for_emotion(self, intent: str, emotional_context: Dict) -> str:
        """Adjust intent based on emotional context"""
        try:
            dominant_emotion = emotional_context.get('dominant_emotion', 'neutral')
            intensity = emotional_context.get('emotional_intensity', 0.5)
            
            # High emotional intensity overrides some intents
            if intensity > 0.7:
                if dominant_emotion == 'negative':
                    if intent in ['question', 'general', 'casual_chat']:
                        return 'support_request'
                elif dominant_emotion == 'positive' and intent == 'general':
                    return 'sharing'
            
            # Emotional expressions should be recognized
            if dominant_emotion != 'neutral' and intent == 'general':
                return 'emotional_expression'
            
            return intent
            
        except Exception as e:
            logging.error(f"[ERROR] Emotion adjustment failed: {e}")
            return intent
    
    def get_intent_confidence(self, user_input: str, detected_intent: str) -> float:
        """Get confidence score for detected intent"""
        try:
            user_text = user_input.lower()
            
            # Count matching patterns for the detected intent
            if detected_intent in self.intent_patterns:
                patterns = self.intent_patterns[detected_intent]
                matches = 0
                for pattern in patterns:
                    matches += len(re.findall(pattern, user_text))
                
                # Calculate confidence based on matches and input length
                base_confidence = min(matches * 0.3, 1.0)
                
                # Adjust for input clarity
                if len(user_input.strip()) < 10:  # Very short inputs are less reliable
                    base_confidence *= 0.7
                elif '?' in user_input and detected_intent == 'question':
                    base_confidence = max(base_confidence, 0.8)
                
                return base_confidence
            
            return 0.5  # Default confidence
            
        except Exception as e:
            logging.error(f"[ERROR] Confidence calculation failed: {e}")
            return 0.5
    
    def analyze_conversation_flow(self, recent_intents: List[str]) -> Dict[str, Any]:
        """Analyze conversation flow patterns"""
        try:
            if not recent_intents:
                return {'pattern': 'new_conversation', 'trend': 'neutral'}
            
            # Count intent frequencies
            intent_counts = Counter(recent_intents)
            
            # Detect patterns
            patterns = []
            
            # Check for repeated support requests
            if intent_counts.get('support_request', 0) > 2:
                patterns.append('seeking_ongoing_support')
            
            # Check for learning progression
            if 'learning_request' in recent_intents and 'question' in recent_intents:
                patterns.append('learning_journey')
            
            # Check for emotional progression
            if 'emotional_expression' in recent_intents:
                if 'support_request' in recent_intents:
                    patterns.append('emotional_processing')
                elif 'sharing' in recent_intents:
                    patterns.append('emotional_sharing')
            
            # Analyze trend
            if len(recent_intents) >= 3:
                recent_3 = recent_intents[-3:]
                if all(intent in ['support_request', 'emotional_expression', 'problem_solving'] for intent in recent_3):
                    trend = 'support_intensive'
                elif all(intent in ['question', 'learning_request'] for intent in recent_3):
                    trend = 'learning_focused'
                elif 'casual_chat' in recent_3:
                    trend = 'social'
                else:
                    trend = 'mixed'
            else:
                trend = 'developing'
            
            return {
                'patterns': patterns,
                'trend': trend,
                'intent_distribution': dict(intent_counts),
                'dominant_intent': intent_counts.most_common(1)[0][0] if intent_counts else 'general'
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Conversation flow analysis failed: {e}")
            return {'pattern': 'unknown', 'trend': 'neutral'}

# Global intent detector instance
intent_detector = IntentDetector()

def detect_intent(user_input: str, use_llm: bool = True) -> str:
    """Detect user intent"""
    return intent_detector.detect_intent(user_input, use_llm)

def get_intent_confidence(user_input: str, detected_intent: str) -> float:
    """Get intent confidence score"""
    return intent_detector.get_intent_confidence(user_input, detected_intent)

def analyze_conversation_flow(recent_intents: List[str]) -> Dict[str, Any]:
    """Analyze conversation flow patterns"""
    return intent_detector.analyze_conversation_flow(recent_intents)
