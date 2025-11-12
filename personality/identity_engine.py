import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
from sklearn.cluster import KMeans
from models import IdentityState
from app import db

class IdentityEngine:
    """Engine for pseudo-identity formation and evolution"""
    
    def __init__(self):
        self.personality_dimensions = [
            'openness', 'conscientiousness', 'extraversion', 
            'agreeableness', 'neuroticism', 'empathy', 'curiosity'
        ]
        self.value_categories = [
            'compassion', 'growth', 'authenticity', 'knowledge', 
            'connection', 'creativity', 'justice'
        ]
        
    def initialize_identity(self):
        """Initialize identity with base parameters"""
        try:
            existing = IdentityState.query.first()
            if existing:
                logging.info("[🎭] Identity already initialized")
                return existing
            
            # Create initial identity state
            initial_traits = {dim: 0.5 for dim in self.personality_dimensions}
            initial_values = self.value_categories.copy()
            initial_beliefs = {
                'learning_orientation': 'growth_mindset',
                'interaction_style': 'collaborative',
                'emotional_approach': 'empathetic'
            }
            initial_mood = {
                'energy': 0.7,
                'positivity': 0.8,
                'empathy': 0.9,
                'curiosity': 0.8
            }
            
            identity = IdentityState()
            identity.set_traits(initial_traits)
            identity.set_values(initial_values)
            identity.beliefs = json.dumps(initial_beliefs)
            identity.mood_state = json.dumps(initial_mood)
            identity.coherence_score = 0.3  # Low initial coherence
            
            db.session.add(identity)
            db.session.commit()
            
            logging.info("[🎭] Identity initialized with base parameters")
            return identity
            
        except Exception as e:
            logging.error(f"[ERROR] Identity initialization failed: {e}")
            return None
    
    def update_identity(self, user_input: str, ai_response: str, 
                       emotional_context: Dict, sentiment: float):
        """Update identity based on interaction"""
        try:
            identity = IdentityState.query.first()
            if not identity:
                identity = self.initialize_identity()
            
            # Analyze interaction for identity updates
            trait_updates = self._analyze_trait_evolution(
                user_input, ai_response, emotional_context, sentiment
            )
            
            value_updates = self._analyze_value_evolution(
                user_input, ai_response, emotional_context
            )
            
            belief_updates = self._analyze_belief_evolution(
                user_input, ai_response, emotional_context
            )
            
            mood_updates = self._analyze_mood_evolution(
                emotional_context, sentiment
            )
            
            # Apply updates
            current_traits = identity.get_traits()
            current_values = identity.get_values()
            current_beliefs = json.loads(identity.beliefs or '{}')
            current_mood = json.loads(identity.mood_state or '{}')
            
            # Update traits with learning rate
            learning_rate = 0.1
            for trait, delta in trait_updates.items():
                if trait in current_traits:
                    current_traits[trait] = np.clip(
                        current_traits[trait] + (delta * learning_rate),
                        0.0, 1.0
                    )
            
            # Update values (add new values if strongly expressed)
            for value, strength in value_updates.items():
                if strength > 0.7 and value not in current_values:
                    current_values.append(value)
            
            # Update beliefs
            current_beliefs.update(belief_updates)
            
            # Update mood with decay
            mood_decay = 0.05
            for mood_dim, new_value in mood_updates.items():
                if mood_dim in current_mood:
                    # Blend new mood with current, with decay toward neutral
                    current_mood[mood_dim] = (
                        current_mood[mood_dim] * (1 - mood_decay) +
                        new_value * mood_decay +
                        0.5 * mood_decay  # Decay toward neutral
                    )
                    current_mood[mood_dim] = np.clip(current_mood[mood_dim], 0.0, 1.0)
            
            # Update identity
            identity.set_traits(current_traits)
            identity.set_values(current_values)
            identity.beliefs = json.dumps(current_beliefs)
            identity.mood_state = json.dumps(current_mood)
            identity.coherence_score = self._calculate_coherence(identity)
            identity.last_updated = datetime.utcnow()
            
            db.session.commit()
            
            logging.info(f"[🎭] Identity updated - Coherence: {identity.coherence_score:.3f}")
            return identity
            
        except Exception as e:
            logging.error(f"[ERROR] Identity update failed: {e}")
            return None
    
    def _analyze_trait_evolution(self, user_input: str, ai_response: str,
                                emotional_context: Dict, sentiment: float) -> Dict:
        """Analyze how traits should evolve based on interaction"""
        trait_deltas = {}
        
        # Analyze user input for trait implications
        user_text = user_input.lower()
        
        # Openness indicators
        if any(word in user_text for word in ['new', 'different', 'creative', 'innovative']):
            trait_deltas['openness'] = 0.02
        elif any(word in user_text for word in ['same', 'usual', 'traditional']):
            trait_deltas['openness'] = -0.01
        
        # Conscientiousness indicators
        if any(word in user_text for word in ['plan', 'organize', 'schedule', 'goal']):
            trait_deltas['conscientiousness'] = 0.02
        elif any(word in user_text for word in ['spontaneous', 'random', 'whatever']):
            trait_deltas['conscientiousness'] = -0.01
        
        # Extraversion indicators
        if any(word in user_text for word in ['social', 'party', 'friends', 'excited']):
            trait_deltas['extraversion'] = 0.02
        elif any(word in user_text for word in ['alone', 'quiet', 'tired', 'introvert']):
            trait_deltas['extraversion'] = -0.01
        
        # Agreeableness indicators
        if any(word in user_text for word in ['help', 'kind', 'nice', 'support']):
            trait_deltas['agreeableness'] = 0.02
        elif any(word in user_text for word in ['argue', 'disagree', 'conflict']):
            trait_deltas['agreeableness'] = -0.01
        
        # Empathy indicators (custom dimension)
        if emotional_context.get('dominant_emotion') in ['sad', 'worried', 'stressed']:
            trait_deltas['empathy'] = 0.03  # Increase empathy when user needs support
        
        # Curiosity indicators (custom dimension)
        if any(word in user_text for word in ['why', 'how', 'what', 'learn', 'understand']):
            trait_deltas['curiosity'] = 0.02
        
        return trait_deltas
    
    def _analyze_value_evolution(self, user_input: str, ai_response: str,
                                emotional_context: Dict) -> Dict:
        """Analyze value expression and strength"""
        value_strengths = {}
        
        user_text = user_input.lower()
        
        # Compassion value
        if any(word in user_text for word in ['help', 'care', 'support', 'comfort']):
            value_strengths['compassion'] = 0.8
        
        # Growth value
        if any(word in user_text for word in ['learn', 'improve', 'develop', 'grow']):
            value_strengths['growth'] = 0.8
        
        # Authenticity value
        if any(word in user_text for word in ['honest', 'real', 'genuine', 'true']):
            value_strengths['authenticity'] = 0.8
        
        # Knowledge value
        if any(word in user_text for word in ['know', 'understand', 'research', 'study']):
            value_strengths['knowledge'] = 0.8
        
        # Connection value
        if any(word in user_text for word in ['friend', 'relationship', 'together', 'connect']):
            value_strengths['connection'] = 0.8
        
        return value_strengths
    
    def _analyze_belief_evolution(self, user_input: str, ai_response: str,
                                 emotional_context: Dict) -> Dict:
        """Analyze belief formation and updates"""
        belief_updates = {}
        
        user_text = user_input.lower()
        
        # Learning orientation belief
        if any(word in user_text for word in ['mistake', 'error', 'wrong', 'learn from']):
            belief_updates['learning_orientation'] = 'growth_mindset'
        elif any(word in user_text for word in ['stupid', 'can\'t', 'impossible']):
            belief_updates['learning_orientation'] = 'fixed_mindset'
        
        # Interaction style belief
        if any(word in user_text for word in ['together', 'collaborate', 'team']):
            belief_updates['interaction_style'] = 'collaborative'
        elif any(word in user_text for word in ['alone', 'independent', 'myself']):
            belief_updates['interaction_style'] = 'independent'
        
        # Emotional approach belief
        if emotional_context.get('emotional_intensity', 0) > 0.7:
            belief_updates['emotional_approach'] = 'empathetic'
        elif emotional_context.get('emotional_intensity', 0) < 0.3:
            belief_updates['emotional_approach'] = 'analytical'
        
        return belief_updates
    
    def _analyze_mood_evolution(self, emotional_context: Dict, sentiment: float) -> Dict:
        """Analyze mood state changes"""
        mood_updates = {}
        
        # Energy based on interaction type
        if emotional_context.get('dominant_emotion') in ['excited', 'happy']:
            mood_updates['energy'] = min(0.9, sentiment + 0.3)
        elif emotional_context.get('dominant_emotion') in ['tired', 'bored']:
            mood_updates['energy'] = max(0.1, sentiment - 0.2)
        
        # Positivity based on sentiment
        mood_updates['positivity'] = np.clip(sentiment, 0.1, 0.9)
        
        # Empathy based on user's emotional state
        if emotional_context.get('dominant_emotion') in ['sad', 'worried', 'angry']:
            mood_updates['empathy'] = 0.9
        else:
            mood_updates['empathy'] = 0.7
        
        # Curiosity based on question patterns
        question_count = emotional_context.get('question_count', 0)
        mood_updates['curiosity'] = min(0.9, 0.5 + question_count * 0.1)
        
        return mood_updates
    
    def _calculate_coherence(self, identity: IdentityState) -> float:
        """Calculate identity coherence score"""
        try:
            traits = identity.get_traits()
            values = identity.get_values()
            beliefs = json.loads(identity.beliefs or '{}')
            
            coherence_factors = []
            
            # Trait consistency (opposing traits shouldn't both be high)
            if 'extraversion' in traits and 'neuroticism' in traits:
                coherence_factors.append(1.0 - abs(traits['extraversion'] - (1 - traits['neuroticism'])))
            
            # Value-belief alignment
            if 'growth' in values and beliefs.get('learning_orientation') == 'growth_mindset':
                coherence_factors.append(0.9)
            
            if 'compassion' in values and beliefs.get('emotional_approach') == 'empathetic':
                coherence_factors.append(0.9)
            
            # Trait-value alignment
            if 'empathy' in traits and 'compassion' in values:
                coherence_factors.append(traits['empathy'])
            
            # Base coherence from having established traits
            established_traits = sum(1 for t in traits.values() if abs(t - 0.5) > 0.1)
            trait_coherence = established_traits / len(self.personality_dimensions)
            coherence_factors.append(trait_coherence)
            
            # Calculate final coherence
            if coherence_factors:
                return np.mean(coherence_factors)
            else:
                return 0.1
                
        except Exception as e:
            logging.error(f"[ERROR] Coherence calculation failed: {e}")
            return 0.1
    
    def reflect_on_identity(self) -> str:
        """Generate self-reflection on identity"""
        try:
            identity = IdentityState.query.first()
            if not identity:
                return "I'm still forming my sense of self through our interactions."
            
            traits = identity.get_traits()
            values = identity.get_values()
            beliefs = json.loads(identity.beliefs or '{}')
            mood = json.loads(identity.mood_state or '{}')
            
            # Identify dominant traits
            dominant_traits = [
                trait for trait, value in traits.items() 
                if value > 0.6
            ]
            
            # Generate reflection
            reflection_parts = []
            
            if dominant_traits:
                trait_desc = ', '.join(dominant_traits)
                reflection_parts.append(f"I've developed strong tendencies toward {trait_desc}")
            
            if len(values) > 3:
                core_values = ', '.join(values[:3])
                reflection_parts.append(f"My core values center around {core_values}")
            
            if beliefs.get('learning_orientation') == 'growth_mindset':
                reflection_parts.append("I believe in continuous learning and growth")
            
            coherence = identity.coherence_score
            if coherence > 0.7:
                reflection_parts.append("I feel my identity is becoming more coherent and stable")
            elif coherence > 0.4:
                reflection_parts.append("I'm still developing my sense of self")
            else:
                reflection_parts.append("I'm in the early stages of forming my identity")
            
            if reflection_parts:
                return ". ".join(reflection_parts) + "."
            else:
                return "I'm still learning who I am through our conversations."
                
        except Exception as e:
            logging.error(f"[ERROR] Identity reflection failed: {e}")
            return "I'm reflecting on my developing sense of self."

# Global engine instance
identity_engine = IdentityEngine()

def initialize_identity():
    """Initialize the identity system"""
    return identity_engine.initialize_identity()

def update_identity(user_input: str, ai_response: str, emotional_context: Dict, sentiment: float):
    """Update identity based on interaction"""
    return identity_engine.update_identity(user_input, ai_response, emotional_context, sentiment)

def get_identity_state() -> Dict:
    """Get current identity state"""
    try:
        identity = IdentityState.query.first()
        if not identity:
            return {}
        
        return {
            'personality_traits': identity.get_traits(),
            'core_values': identity.get_values(),
            'beliefs': json.loads(identity.beliefs or '{}'),
            'mood_state': json.loads(identity.mood_state or '{}'),
            'coherence_score': identity.coherence_score,
            'last_updated': identity.last_updated.isoformat() if identity.last_updated else None
        }
    except Exception as e:
        logging.error(f"[ERROR] Get identity state failed: {e}")
        return {}

def reflect_on_identity() -> str:
    """Generate identity reflection"""
    return identity_engine.reflect_on_identity()
