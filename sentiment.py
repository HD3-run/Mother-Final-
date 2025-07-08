import re
import logging
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
import numpy as np
from collections import Counter
import os

class SentimentAnalyzer:
    """Enhanced sentiment analysis with emotional context understanding"""
    
    def __init__(self):
        self.emotion_lexicon = {
            'happy': ['happy', 'joyful', 'cheerful', 'delighted', 'pleased', 'content', 'satisfied', 'glad', 'elated', 'ecstatic'],
            'sad': ['sad', 'unhappy', 'depressed', 'melancholy', 'sorrowful', 'grief', 'mourning', 'dejected', 'downcast', 'gloomy'],
            'angry': ['angry', 'furious', 'mad', 'irritated', 'annoyed', 'frustrated', 'outraged', 'livid', 'enraged', 'irate'],
            'fear': ['afraid', 'scared', 'fearful', 'terrified', 'anxious', 'worried', 'nervous', 'panicked', 'frightened', 'alarmed'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned', 'bewildered', 'startled', 'astounded'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sickened', 'nauseated', 'appalled'],
            'excited': ['excited', 'thrilled', 'enthusiastic', 'eager', 'pumped', 'energized', 'exhilarated', 'stimulated']
        }
        
        self.intensity_modifiers = {
            'very': 1.5, 'extremely': 2.0, 'incredibly': 2.0, 'absolutely': 1.8,
            'really': 1.3, 'quite': 1.2, 'somewhat': 0.8, 'slightly': 0.6,
            'a bit': 0.7, 'kind of': 0.8, 'sort of': 0.8, 'totally': 1.8,
            'completely': 2.0, 'utterly': 2.0, 'deeply': 1.6, 'profoundly': 1.8
        }
        
        self.negation_words = ['not', 'no', 'never', 'nothing', 'nobody', 'nowhere', 
                              'neither', 'nor', 'none', 'without', 'hardly', 'barely']
        
        # Contextual emotion patterns
        self.emotion_contexts = {
            'work_stress': ['deadline', 'boss', 'meeting', 'project', 'work', 'job', 'office'],
            'relationship': ['friend', 'family', 'partner', 'love', 'relationship', 'dating'],
            'health': ['sick', 'pain', 'doctor', 'hospital', 'medicine', 'health'],
            'achievement': ['accomplished', 'achieved', 'success', 'won', 'finished', 'completed'],
            'loss': ['died', 'death', 'lost', 'goodbye', 'funeral', 'miss', 'gone']
        }
    
    def get_sentiment(self, text: str) -> float:
        """Get basic sentiment score (-1 to 1)"""
        try:
            if not text:
                return 0.5
            
            # Simple rule-based sentiment
            positive_words = ['good', 'great', 'amazing', 'wonderful', 'excellent', 'fantastic', 
                             'love', 'like', 'happy', 'joy', 'perfect', 'awesome', 'brilliant']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'sad', 'angry', 
                             'worst', 'disgusting', 'pathetic', 'useless', 'disappointed']
            
            words = text.lower().split()
            positive_count = sum(1 for word in words if word in positive_words)
            negative_count = sum(1 for word in words if word in negative_words)
            
            # Check for negation
            negated_positive = 0
            negated_negative = 0
            
            for i, word in enumerate(words):
                if word in self.negation_words and i < len(words) - 1:
                    next_word = words[i + 1]
                    if next_word in positive_words:
                        negated_positive += 1
                    elif next_word in negative_words:
                        negated_negative += 1
            
            # Adjust counts for negation
            effective_positive = positive_count - negated_positive + negated_negative
            effective_negative = negative_count - negated_negative + negated_positive
            
            # Calculate sentiment
            total_emotional_words = effective_positive + effective_negative
            if total_emotional_words == 0:
                return 0.5  # Neutral
            
            sentiment = effective_positive / total_emotional_words
            
            # Normalize to 0-1 range (0.5 is neutral)
            return max(0.0, min(1.0, sentiment))
            
        except Exception as e:
            logging.error(f"[ERROR] Sentiment analysis failed: {e}")
            return 0.5
    
    def analyze_emotional_context(self, text: str) -> Dict[str, Any]:
        """Analyze emotional context with detailed breakdown"""
        try:
            text_lower = text.lower()
            words = text_lower.split()
            
            # Detect emotions
            emotion_scores = {}
            detected_emotions = []
            
            for emotion, emotion_words in self.emotion_lexicon.items():
                score = 0
                found_words = []
                
                for word in emotion_words:
                    if word in text_lower:
                        base_score = 1
                        
                        # Apply intensity modifiers
                        word_index = text_lower.find(word)
                        context_before = text_lower[max(0, word_index-20):word_index]
                        
                        for modifier, multiplier in self.intensity_modifiers.items():
                            if modifier in context_before:
                                base_score *= multiplier
                                break
                        
                        # Check for negation
                        negation_context = text_lower[max(0, word_index-10):word_index]
                        if any(neg in negation_context for neg in self.negation_words):
                            base_score *= -0.5  # Flip and reduce
                        
                        score += base_score
                        found_words.append(word)
                
                if score > 0:
                    emotion_scores[emotion] = score
                    detected_emotions.extend(found_words)
            
            # Determine dominant emotion
            dominant_emotion = 'neutral'
            max_score = 0
            
            if emotion_scores:
                dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                max_score = emotion_scores[dominant_emotion]
            
            # Calculate emotional intensity
            total_emotional_words = len(detected_emotions)
            text_length = len(words)
            
            if text_length > 0:
                emotional_density = total_emotional_words / text_length
                intensity = min(emotional_density * 2, 1.0)  # Cap at 1.0
                
                # Boost intensity for very emotional words
                if max_score > 2:
                    intensity = min(intensity * 1.5, 1.0)
            else:
                intensity = 0.0
            
            # Detect emotional context
            context_type = self._detect_emotional_context(text_lower)
            
            # Count questions (indicates uncertainty/curiosity)
            question_count = text.count('?')
            
            # Detect emotional progression indicators
            progression = self._analyze_emotional_progression(text_lower)
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotion_scores': emotion_scores,
                'emotional_intensity': intensity,
                'emotional_indicators': detected_emotions,
                'context_type': context_type,
                'question_count': question_count,
                'emotional_progression': progression,
                'needs_assessment': self._assess_emotional_needs(dominant_emotion, intensity, context_type)
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional context analysis failed: {e}")
            return {
                'dominant_emotion': 'neutral',
                'emotion_scores': {},
                'emotional_intensity': 0.5,
                'emotional_indicators': [],
                'context_type': 'general',
                'question_count': 0,
                'emotional_progression': 'stable',
                'needs_assessment': 'general_support'
            }
    
    def _detect_emotional_context(self, text: str) -> str:
        """Detect the emotional context/domain"""
        try:
            for context, keywords in self.emotion_contexts.items():
                if any(keyword in text for keyword in keywords):
                    return context
            return 'general'
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional context detection failed: {e}")
            return 'general'
    
    def _analyze_emotional_progression(self, text: str) -> str:
        """Analyze emotional progression in the text"""
        try:
            # Look for progression indicators
            improvement_words = ['better', 'improving', 'getting better', 'feeling better', 'looking up']
            decline_words = ['worse', 'getting worse', 'deteriorating', 'falling apart', 'breaking down']
            stability_words = ['same', 'stable', 'consistent', 'unchanged', 'steady']
            
            if any(word in text for word in improvement_words):
                return 'improving'
            elif any(word in text for word in decline_words):
                return 'declining'
            elif any(word in text for word in stability_words):
                return 'stable'
            else:
                return 'unclear'
                
        except Exception as e:
            logging.error(f"[ERROR] Emotional progression analysis failed: {e}")
            return 'stable'
    
    def _assess_emotional_needs(self, emotion: str, intensity: float, context: str) -> str:
        """Assess what the user likely needs based on emotional state"""
        try:
            if intensity > 0.8:  # High emotional intensity
                if emotion in ['sad', 'fear', 'angry']:
                    return 'immediate_emotional_support'
                elif emotion in ['happy', 'excited']:
                    return 'celebration_sharing'
            
            elif intensity > 0.5:  # Moderate emotional intensity
                if emotion in ['sad', 'fear']:
                    return 'gentle_support'
                elif emotion == 'angry':
                    return 'validation_and_calming'
                elif emotion in ['happy', 'excited']:
                    return 'positive_engagement'
            
            # Context-specific needs
            if context == 'work_stress':
                return 'stress_management_support'
            elif context == 'relationship':
                return 'relationship_guidance'
            elif context == 'health':
                return 'empathetic_concern'
            elif context == 'achievement':
                return 'celebration_and_encouragement'
            elif context == 'loss':
                return 'grief_support'
            
            return 'general_conversation'
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional needs assessment failed: {e}")
            return 'general_support'
    
    def track_emotional_journey(self, emotions_over_time: List[Dict]) -> Dict[str, Any]:
        """Track emotional journey over multiple interactions"""
        try:
            if not emotions_over_time:
                return {'journey_type': 'unknown', 'trend': 'stable'}
            
            # Extract dominant emotions and intensities
            emotions = [e.get('dominant_emotion', 'neutral') for e in emotions_over_time]
            intensities = [e.get('emotional_intensity', 0.5) for e in emotions_over_time]
            
            # Analyze trends
            emotion_counts = Counter(emotions)
            
            # Calculate emotional stability
            unique_emotions = len(set(emotions))
            stability = 1.0 - (unique_emotions / len(emotions))
            
            # Analyze intensity trend
            if len(intensities) > 1:
                intensity_trend = np.polyfit(range(len(intensities)), intensities, 1)[0]
                if intensity_trend > 0.1:
                    trend = 'intensifying'
                elif intensity_trend < -0.1:
                    trend = 'calming'
                else:
                    trend = 'stable'
            else:
                trend = 'stable'
            
            # Identify journey patterns
            journey_type = 'mixed'
            if stability > 0.7:
                most_common_emotion = emotion_counts.most_common(1)[0][0]
                if most_common_emotion in ['happy', 'excited']:
                    journey_type = 'consistently_positive'
                elif most_common_emotion in ['sad', 'angry', 'fear']:
                    journey_type = 'consistently_challenging'
                else:
                    journey_type = 'stable_neutral'
            
            # Check for recovery patterns
            if len(emotions) >= 3:
                recent_emotions = emotions[-3:]
                if any(e in ['sad', 'angry', 'fear'] for e in emotions[:-2]) and \
                   all(e in ['happy', 'excited', 'neutral'] for e in recent_emotions):
                    journey_type = 'recovery'
            
            return {
                'journey_type': journey_type,
                'trend': trend,
                'emotional_stability': stability,
                'dominant_emotions': dict(emotion_counts.most_common(3)),
                'intensity_trend': intensity_trend if len(intensities) > 1 else 0,
                'support_recommendation': self._recommend_support_strategy(journey_type, trend)
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional journey tracking failed: {e}")
            return {'journey_type': 'unknown', 'trend': 'stable'}
    
    def _recommend_support_strategy(self, journey_type: str, trend: str) -> str:
        """Recommend support strategy based on emotional journey"""
        try:
            if journey_type == 'consistently_challenging':
                if trend == 'intensifying':
                    return 'immediate_intervention'
                else:
                    return 'sustained_support'
            elif journey_type == 'recovery':
                return 'encouragement_and_reinforcement'
            elif journey_type == 'consistently_positive':
                return 'celebration_and_growth'
            elif trend == 'intensifying':
                return 'monitor_and_support'
            else:
                return 'maintain_connection'
                
        except Exception as e:
            logging.error(f"[ERROR] Support strategy recommendation failed: {e}")
            return 'general_support'

# Global sentiment analyzer instance
sentiment_analyzer = SentimentAnalyzer()

def get_sentiment(text: str) -> float:
    """Get sentiment score for text"""
    return sentiment_analyzer.get_sentiment(text)

def analyze_emotional_context(text: str) -> Dict[str, Any]:
    """Analyze emotional context of text"""
    return sentiment_analyzer.analyze_emotional_context(text)

def track_emotional_journey(emotions_over_time: List[Dict]) -> Dict[str, Any]:
    """Track emotional journey over time"""
    return sentiment_analyzer.track_emotional_journey(emotions_over_time)
