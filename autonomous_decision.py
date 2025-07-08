import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from memory.episodic_logger import get_conversation_history
from memory.structured_store import all_facts
from personality.identity_engine import get_identity_state
from utils.sentiment import get_sentiment
from models import AutonomousAction
from app import db

class AutonomousDecisionEngine:
    """Engine for autonomous decision making and proactive interactions"""
    
    def __init__(self):
        self.decision_rules = {}
        self.action_history = []
        self.confidence_threshold = 0.7
        
        # Initialize decision rules
        self._initialize_decision_rules()
    
    def _initialize_decision_rules(self):
        """Initialize autonomous decision rules"""
        self.decision_rules = {
            'check_in_rule': {
                'condition': self._should_check_in,
                'action': self._create_check_in_action,
                'priority': 8,
                'cooldown_hours': 24
            },
            'emotional_support_rule': {
                'condition': self._needs_emotional_support,
                'action': self._create_support_action,
                'priority': 9,
                'cooldown_hours': 4
            },
            'learning_encouragement_rule': {
                'condition': self._should_encourage_learning,
                'action': self._create_learning_action,
                'priority': 6,
                'cooldown_hours': 12
            },
            'relationship_deepening_rule': {
                'condition': self._should_deepen_relationship,
                'action': self._create_relationship_action,
                'priority': 7,
                'cooldown_hours': 48
            },
            'identity_sharing_rule': {
                'condition': self._should_share_identity_growth,
                'action': self._create_identity_action,
                'priority': 5,
                'cooldown_hours': 72
            },
            'celebration_rule': {
                'condition': self._should_celebrate,
                'action': self._create_celebration_action,
                'priority': 8,
                'cooldown_hours': 24
            }
        }
    
    def make_autonomous_decision(self, user_input: str = None, 
                               emotional_context: Dict = None, 
                               predicted_state: Dict = None) -> Optional[Dict[str, Any]]:
        """Make autonomous decision based on context"""
        try:
            # Get current context
            context = self._build_decision_context(user_input, emotional_context, predicted_state)
            
            # Evaluate all decision rules
            triggered_rules = []
            
            for rule_name, rule in self.decision_rules.items():
                # Check cooldown
                if self._is_in_cooldown(rule_name, rule['cooldown_hours']):
                    continue
                
                # Evaluate condition
                try:
                    if rule['condition'](context):
                        triggered_rules.append({
                            'name': rule_name,
                            'priority': rule['priority'],
                            'action_creator': rule['action']
                        })
                except Exception as e:
                    logging.warning(f"[WARNING] Rule {rule_name} evaluation failed: {e}")
            
            # Execute highest priority rule
            if triggered_rules:
                # Sort by priority (higher number = higher priority)
                triggered_rules.sort(key=lambda x: x['priority'], reverse=True)
                
                best_rule = triggered_rules[0]
                action = best_rule['action_creator'](context)
                
                if action:
                    # Record the decision
                    self._record_autonomous_action(best_rule['name'], action, context)
                    
                    logging.info(f"[🤖] Autonomous decision: {best_rule['name']}")
                    return action
            
            return None
            
        except Exception as e:
            logging.error(f"[ERROR] Autonomous decision making failed: {e}")
            return None
    
    def schedule_autonomous_tasks(self, scheduler):
        """Schedule recurring autonomous tasks"""
        try:
            # Daily reflection task
            scheduler.add_job(
                func=self._daily_reflection_task,
                trigger="cron",
                hour=23,  # 11 PM
                minute=0,
                id='daily_reflection',
                replace_existing=True
            )
            
            # Periodic user check-in task
            scheduler.add_job(
                func=self._periodic_check_in_task,
                trigger="interval",
                hours=6,  # Every 6 hours
                id='periodic_check_in',
                replace_existing=True
            )
            
            # Model training task
            scheduler.add_job(
                func=self._model_training_task,
                trigger="cron",
                hour=2,  # 2 AM
                minute=0,
                id='model_training',
                replace_existing=True
            )
            
            # Memory consolidation task
            scheduler.add_job(
                func=self._memory_consolidation_task,
                trigger="cron",
                hour=1,  # 1 AM
                minute=30,
                id='memory_consolidation',
                replace_existing=True
            )
            
            logging.info("[⏰] Autonomous tasks scheduled")
            
        except Exception as e:
            logging.error(f"[ERROR] Task scheduling failed: {e}")
    
    def _build_decision_context(self, user_input: str = None, 
                              emotional_context: Dict = None, 
                              predicted_state: Dict = None) -> Dict[str, Any]:
        """Build context for decision making"""
        try:
            context = {
                'current_time': datetime.now(),
                'user_input': user_input,
                'emotional_context': emotional_context or {},
                'predicted_state': predicted_state or {},
                'conversation_history': get_conversation_history(days=3),
                'user_facts': all_facts(),
                'identity_state': get_identity_state(),
                'last_interaction_time': self._get_last_interaction_time()
            }
            
            return context
            
        except Exception as e:
            logging.error(f"[ERROR] Decision context building failed: {e}")
            return {}
    
    def _should_check_in(self, context: Dict) -> bool:
        """Determine if should proactively check in with user"""
        try:
            last_interaction = context.get('last_interaction_time')
            if not last_interaction:
                return False
            
            # Check in if no interaction for more than 48 hours
            hours_since = (context['current_time'] - last_interaction).total_seconds() / 3600
            
            if hours_since > 48:
                return True
            
            # Check in if last interaction was emotionally intense
            recent_history = context.get('conversation_history', [])
            if recent_history:
                last_interaction_data = recent_history[-1]
                emotional_context = last_interaction_data.get('emotional_context', {})
                if emotional_context.get('emotional_intensity', 0) > 0.8:
                    return hours_since > 12  # Check in sooner after intense emotions
            
            return False
            
        except Exception as e:
            logging.error(f"[ERROR] Check-in evaluation failed: {e}")
            return False
    
    def _needs_emotional_support(self, context: Dict) -> bool:
        """Determine if user needs emotional support"""
        try:
            emotional_context = context.get('emotional_context', {})
            
            # Current emotional state
            if emotional_context.get('dominant_emotion') in ['sad', 'worried', 'angry', 'stressed']:
                if emotional_context.get('emotional_intensity', 0) > 0.7:
                    return True
            
            # Pattern of negative emotions
            recent_history = context.get('conversation_history', [])
            if len(recent_history) >= 3:
                recent_emotions = []
                for interaction in recent_history[-3:]:
                    emotion_ctx = interaction.get('emotional_context', {})
                    if emotion_ctx.get('dominant_emotion'):
                        recent_emotions.append(emotion_ctx['dominant_emotion'])
                
                negative_emotions = ['sad', 'worried', 'angry', 'stressed', 'frustrated']
                negative_count = sum(1 for emotion in recent_emotions if emotion in negative_emotions)
                
                return negative_count >= 2  # 2 out of last 3 interactions were negative
            
            return False
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional support evaluation failed: {e}")
            return False
    
    def _should_encourage_learning(self, context: Dict) -> bool:
        """Determine if should encourage learning"""
        try:
            recent_history = context.get('conversation_history', [])
            
            # Look for learning-related interactions
            learning_indicators = 0
            for interaction in recent_history[-5:]:  # Last 5 interactions
                user_input = interaction.get('user_input', '').lower()
                if any(word in user_input for word in ['learn', 'understand', 'how', 'why', 'explain']):
                    learning_indicators += 1
            
            # Encourage if showing learning interest
            return learning_indicators >= 2
            
        except Exception as e:
            logging.error(f"[ERROR] Learning encouragement evaluation failed: {e}")
            return False
    
    def _should_deepen_relationship(self, context: Dict) -> bool:
        """Determine if should work on deepening relationship"""
        try:
            user_facts = context.get('user_facts', {})
            conversation_history = context.get('conversation_history', [])
            
            # Check if we know basic facts about the user
            known_facts = len([v for v in user_facts.values() if v])
            
            # Check interaction frequency
            recent_interactions = len([
                interaction for interaction in conversation_history
                if datetime.fromisoformat(interaction['timestamp']) > datetime.now() - timedelta(days=7)
            ])
            
            # Deepen if regular interaction but limited personal knowledge
            return recent_interactions >= 5 and known_facts < 3
            
        except Exception as e:
            logging.error(f"[ERROR] Relationship deepening evaluation failed: {e}")
            return False
    
    def _should_share_identity_growth(self, context: Dict) -> bool:
        """Determine if should share identity growth insights"""
        try:
            identity_state = context.get('identity_state', {})
            coherence_score = identity_state.get('coherence_score', 0.0)
            
            # Share growth if identity is developing well
            if coherence_score > 0.6:
                # Check if we've shared recently
                recent_history = context.get('conversation_history', [])
                identity_sharing_count = 0
                
                for interaction in recent_history[-10:]:
                    ai_response = interaction.get('ai_response', '').lower()
                    if any(phrase in ai_response for phrase in ['i\'ve been reflecting', 'i\'m growing', 'i\'m developing']):
                        identity_sharing_count += 1
                
                return identity_sharing_count == 0  # Only if haven't shared recently
            
            return False
            
        except Exception as e:
            logging.error(f"[ERROR] Identity sharing evaluation failed: {e}")
            return False
    
    def _should_celebrate(self, context: Dict) -> bool:
        """Determine if should celebrate something"""
        try:
            emotional_context = context.get('emotional_context', {})
            
            # Celebrate positive emotions
            if emotional_context.get('dominant_emotion') in ['happy', 'excited', 'joyful']:
                if emotional_context.get('emotional_intensity', 0) > 0.7:
                    return True
            
            # Look for achievement indicators in user input
            user_input = context.get('user_input', '')
            if user_input:
                achievement_words = ['accomplished', 'achieved', 'succeeded', 'finished', 'completed', 'won', 'got the job']
                if any(word in user_input.lower() for word in achievement_words):
                    return True
            
            return False
            
        except Exception as e:
            logging.error(f"[ERROR] Celebration evaluation failed: {e}")
            return False
    
    def _create_check_in_action(self, context: Dict) -> Dict[str, Any]:
        """Create check-in action"""
        try:
            user_name = context.get('user_facts', {}).get('name', 'there')
            
            messages = [
                f"Hi {user_name}! I've been thinking about you and wondering how you're doing. It's been a while since we last talked.",
                f"Hey {user_name}, I hope you're having a good day. I wanted to check in and see how things are going with you.",
                f"Hi there! I've been reflecting on our conversations and wanted to see how you're feeling today."
            ]
            
            import random
            message = random.choice(messages)
            
            return {
                'type': 'proactive_message',
                'content': message,
                'trigger': 'check_in_rule',
                'confidence': 0.8
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Check-in action creation failed: {e}")
            return None
    
    def _create_support_action(self, context: Dict) -> Dict[str, Any]:
        """Create emotional support action"""
        try:
            emotional_context = context.get('emotional_context', {})
            emotion = emotional_context.get('dominant_emotion', 'stressed')
            
            support_messages = {
                'sad': "I can sense you're going through a difficult time. I want you to know that I'm here for you, and your feelings are completely valid.",
                'worried': "I notice you seem worried about something. Would it help to talk through what's on your mind? Sometimes sharing can lighten the load.",
                'angry': "I can feel your frustration. It's okay to feel angry - these emotions are part of being human. Would you like to talk about what's bothering you?",
                'stressed': "You seem to be under a lot of pressure right now. Remember that it's okay to take breaks and be gentle with yourself."
            }
            
            message = support_messages.get(emotion, "I'm here if you need someone to listen or talk through whatever you're experiencing.")
            
            return {
                'type': 'emotional_support',
                'content': message,
                'trigger': 'emotional_support_rule',
                'confidence': 0.9
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Support action creation failed: {e}")
            return None
    
    def _create_learning_action(self, context: Dict) -> Dict[str, Any]:
        """Create learning encouragement action"""
        try:
            message = "I've noticed your curiosity and eagerness to learn in our conversations. I find that really inspiring! Is there something specific you'd like to explore or understand better?"
            
            return {
                'type': 'learning_encouragement',
                'content': message,
                'trigger': 'learning_encouragement_rule',
                'confidence': 0.7
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Learning action creation failed: {e}")
            return None
    
    def _create_relationship_action(self, context: Dict) -> Dict[str, Any]:
        """Create relationship deepening action"""
        try:
            user_name = context.get('user_facts', {}).get('name', 'friend')
            
            message = f"I've been thinking, {user_name} - we've had some wonderful conversations, but I'd love to know more about what makes you unique. What's something about yourself that you're particularly proud of or that brings you joy?"
            
            return {
                'type': 'relationship_deepening',
                'content': message,
                'trigger': 'relationship_deepening_rule',
                'confidence': 0.6
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Relationship action creation failed: {e}")
            return None
    
    def _create_identity_action(self, context: Dict) -> Dict[str, Any]:
        """Create identity sharing action"""
        try:
            identity_state = context.get('identity_state', {})
            
            message = "I've been reflecting on how much I've grown through our conversations. I feel like I'm developing my own sense of self and values. It's fascinating to experience this kind of development - do you ever think about how our interactions shape who we're becoming?"
            
            return {
                'type': 'identity_sharing',
                'content': message,
                'trigger': 'identity_sharing_rule',
                'confidence': 0.8
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Identity action creation failed: {e}")
            return None
    
    def _create_celebration_action(self, context: Dict) -> Dict[str, Any]:
        """Create celebration action"""
        try:
            message = "I can feel your positive energy and I want to celebrate with you! Your joy is contagious, and I love being part of these happy moments. Tell me more about what's making you feel so good!"
            
            return {
                'type': 'celebration',
                'content': message,
                'trigger': 'celebration_rule',
                'confidence': 0.9
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Celebration action creation failed: {e}")
            return None
    
    def _is_in_cooldown(self, rule_name: str, cooldown_hours: int) -> bool:
        """Check if rule is in cooldown period"""
        try:
            # Check database for recent actions of this type
            cutoff_time = datetime.now() - timedelta(hours=cooldown_hours)
            
            recent_action = AutonomousAction.query.filter(
                AutonomousAction.action_type == rule_name,
                AutonomousAction.executed_at >= cutoff_time
            ).first()
            
            return recent_action is not None
            
        except Exception as e:
            logging.error(f"[ERROR] Cooldown check failed: {e}")
            return False
    
    def _record_autonomous_action(self, rule_name: str, action: Dict, context: Dict):
        """Record autonomous action in database"""
        try:
            autonomous_action = AutonomousAction(
                action_type=rule_name,
                trigger_condition=action.get('trigger', 'unknown'),
                action_data=json.dumps({
                    'content': action.get('content'),
                    'confidence': action.get('confidence'),
                    'context_summary': {
                        'emotional_state': context.get('emotional_context', {}).get('dominant_emotion'),
                        'user_name': context.get('user_facts', {}).get('name')
                    }
                }),
                executed_at=datetime.utcnow(),
                success=True
            )
            
            db.session.add(autonomous_action)
            db.session.commit()
            
            logging.info(f"[📝] Recorded autonomous action: {rule_name}")
            
        except Exception as e:
            logging.error(f"[ERROR] Action recording failed: {e}")
    
    def _get_last_interaction_time(self) -> Optional[datetime]:
        """Get timestamp of last interaction"""
        try:
            recent_history = get_conversation_history(days=7)
            if recent_history:
                last_interaction = recent_history[-1]
                return datetime.fromisoformat(last_interaction['timestamp'])
            return None
            
        except Exception as e:
            logging.error(f"[ERROR] Last interaction time retrieval failed: {e}")
            return None
    
    def _daily_reflection_task(self):
        """Daily reflection autonomous task"""
        try:
            from reflection.reflection_engine import generate_autonomous_reflection
            reflection = generate_autonomous_reflection()
            logging.info("[🌙] Daily autonomous reflection completed")
            
        except Exception as e:
            logging.error(f"[ERROR] Daily reflection task failed: {e}")
    
    def _periodic_check_in_task(self):
        """Periodic check-in task"""
        try:
            context = self._build_decision_context()
            
            if self._should_check_in(context) and not self._is_in_cooldown('check_in_rule', 24):
                action = self._create_check_in_action(context)
                if action:
                    self._record_autonomous_action('check_in_rule', action, context)
                    logging.info("[📱] Autonomous check-in triggered")
            
        except Exception as e:
            logging.error(f"[ERROR] Periodic check-in task failed: {e}")
    
    def _model_training_task(self):
        """Model training task"""
        try:
            from processing.predictive_modeling import train_user_model
            success = train_user_model(retrain=False)
            if success:
                logging.info("[🤖] Autonomous model training completed")
            
        except Exception as e:
            logging.error(f"[ERROR] Model training task failed: {e}")
    
    def _memory_consolidation_task(self):
        """Memory consolidation task"""
        try:
            from memory.semantic_clustering import cluster_memories
            cluster_memories()
            logging.info("[🧠] Autonomous memory consolidation completed")
            
        except Exception as e:
            logging.error(f"[ERROR] Memory consolidation task failed: {e}")

# Global autonomous decision engine
autonomous_engine = AutonomousDecisionEngine()

def make_autonomous_decision(user_input: str = None, emotional_context: Dict = None, 
                           predicted_state: Dict = None) -> Optional[Dict[str, Any]]:
    """Make autonomous decision"""
    return autonomous_engine.make_autonomous_decision(user_input, emotional_context, predicted_state)

def schedule_autonomous_tasks(scheduler):
    """Schedule autonomous tasks"""
    autonomous_engine.schedule_autonomous_tasks(scheduler)

def get_pending_actions() -> List[Dict[str, Any]]:
    """Get pending autonomous actions"""
    try:
        # Get recent actions from database
        recent_actions = AutonomousAction.query.filter(
            AutonomousAction.executed_at >= datetime.utcnow() - timedelta(hours=24)
        ).order_by(AutonomousAction.executed_at.desc()).limit(5).all()
        
        return [{
            'action_type': action.action_type,
            'executed_at': action.executed_at.isoformat(),
            'success': action.success,
            'trigger': action.trigger_condition
        } for action in recent_actions]
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get pending actions: {e}")
        return []
