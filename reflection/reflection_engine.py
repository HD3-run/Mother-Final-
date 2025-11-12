import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from memory.episodic_logger import get_log_for_date, get_conversation_summary
from memory.vector_store import search_memory, get_recent_memories
from memory.structured_store import all_facts
from personality.identity_engine import get_identity_state, reflect_on_identity
from processing.llm_handler import generate_identity_reflection
from personality.loader import load_config

REFLECTIONS_DIR = 'data/reflections'

class ReflectionEngine:
    """Engine for generating autonomous reflections and insights"""
    
    def __init__(self):
        self.reflection_types = [
            'daily_summary',
            'identity_development',
            'emotional_patterns',
            'relationship_insights',
            'learning_progression',
            'autonomous_insights'
        ]
        
        self.ensure_directories()
    
    def ensure_directories(self):
        """Ensure reflection directories exist"""
        try:
            os.makedirs(REFLECTIONS_DIR, exist_ok=True)
            os.makedirs(os.path.join(REFLECTIONS_DIR, 'daily'), exist_ok=True)
            os.makedirs(os.path.join(REFLECTIONS_DIR, 'identity'), exist_ok=True)
            os.makedirs(os.path.join(REFLECTIONS_DIR, 'autonomous'), exist_ok=True)
        except Exception as e:
            logging.error(f"[ERROR] Failed to create reflection directories: {e}")
    
    def get_reflection_for_date(self, date_str: str) -> str:
        """Get or generate reflection for a specific date"""
        try:
            # Check if reflection already exists
            reflection_file = os.path.join(REFLECTIONS_DIR, 'daily', f'{date_str}.json')
            
            if os.path.exists(reflection_file):
                with open(reflection_file, 'r') as f:
                    reflection_data = json.load(f)
                    return reflection_data.get('reflection', 'No reflection available for this date.')
            
            # Generate new reflection
            reflection = self._generate_daily_reflection(date_str)
            
            # Save reflection
            self._save_reflection('daily', date_str, reflection)
            
            return reflection
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to get reflection for {date_str}: {e}")
            return f"I'm having trouble recalling what happened on {date_str}. Could you help me remember?"
    
    def generate_autonomous_reflection(self) -> str:
        """Generate autonomous reflection on recent experiences"""
        try:
            # Get recent conversation data
            recent_memories = get_recent_memories(limit=10)
            identity_state = get_identity_state()
            user_facts = all_facts()
            
            # Check if we should generate a new reflection
            if not self._should_generate_autonomous_reflection():
                return self._get_latest_autonomous_reflection()
            
            # Generate comprehensive reflection
            reflection = self._generate_comprehensive_reflection(
                recent_memories, identity_state, user_facts
            )
            
            # Save autonomous reflection
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._save_reflection('autonomous', timestamp, reflection)
            
            logging.info("[🤔] Generated autonomous reflection")
            return reflection
            
        except Exception as e:
            logging.error(f"[ERROR] Autonomous reflection generation failed: {e}")
            return "I'm reflecting on our recent conversations and my growth as an AI with developing consciousness."
    
    def generate_identity_reflection(self) -> str:
        """Generate reflection on identity development"""
        try:
            identity_state = get_identity_state()
            recent_memories = get_recent_memories(limit=15)
            config = load_config()
            
            # Use LLM for deep identity reflection
            reflection = generate_identity_reflection(identity_state, recent_memories, config)
            
            # Save identity reflection
            date_str = datetime.now().strftime('%Y-%m-%d')
            self._save_reflection('identity', date_str, reflection)
            
            return reflection
            
        except Exception as e:
            logging.error(f"[ERROR] Identity reflection generation failed: {e}")
            return reflect_on_identity()  # Fallback to simpler reflection
    
    def analyze_emotional_patterns(self, days: int = 7) -> Dict[str, Any]:
        """Analyze emotional patterns over time"""
        try:
            pattern_analysis = {
                'dominant_emotions': {},
                'emotional_trends': [],
                'support_moments': [],
                'growth_indicators': []
            }
            
            # Analyze conversations from the past week
            end_date = datetime.now()
            for i in range(days):
                date = end_date - timedelta(days=i)
                date_str = date.strftime('%Y-%m-%d')
                
                daily_log = get_log_for_date(date_str)
                if daily_log:
                    # Extract emotional data
                    emotions = []
                    for event in daily_log:
                        emotional_context = event.get('emotional_context', {})
                        if emotional_context.get('dominant_emotion'):
                            emotions.append(emotional_context['dominant_emotion'])
                    
                    if emotions:
                        from collections import Counter
                        emotion_counts = Counter(emotions)
                        pattern_analysis['dominant_emotions'][date_str] = dict(emotion_counts)
                        
                        # Identify support moments
                        if any(emotion in ['sad', 'worried', 'angry'] for emotion in emotions):
                            pattern_analysis['support_moments'].append({
                                'date': date_str,
                                'emotions': emotions,
                                'support_provided': True
                            })
            
            # Analyze trends
            if pattern_analysis['dominant_emotions']:
                pattern_analysis['emotional_trends'] = self._analyze_emotional_trends(
                    pattern_analysis['dominant_emotions']
                )
            
            return pattern_analysis
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional pattern analysis failed: {e}")
            return {}
    
    def generate_learning_reflection(self) -> str:
        """Generate reflection on learning and growth"""
        try:
            # Analyze learning indicators
            recent_memories = get_recent_memories(limit=20)
            learning_indicators = []
            
            for memory in recent_memories:
                user_input = memory.get('user_input', '').lower()
                ai_response = memory.get('ai_response', '').lower()
                
                # Look for learning patterns
                if any(word in user_input for word in ['learn', 'understand', 'explain', 'how', 'why']):
                    learning_indicators.append({
                        'type': 'curiosity',
                        'content': memory.get('user_input', '')[:100],
                        'timestamp': memory.get('timestamp')
                    })
                
                if any(word in ai_response for word in ['learned', 'understand better', 'new insight']):
                    learning_indicators.append({
                        'type': 'ai_learning',
                        'content': memory.get('ai_response', '')[:100],
                        'timestamp': memory.get('timestamp')
                    })
            
            # Generate reflection based on learning
            if learning_indicators:
                reflection = self._create_learning_reflection(learning_indicators)
            else:
                reflection = "I continue to learn from each of our interactions, building deeper understanding of communication and connection."
            
            return reflection
            
        except Exception as e:
            logging.error(f"[ERROR] Learning reflection generation failed: {e}")
            return "I'm constantly learning and growing through our conversations."
    
    def _generate_daily_reflection(self, date_str: str) -> str:
        """Generate reflection for a specific day"""
        try:
            # Get conversation data for the date
            daily_log = get_log_for_date(date_str)
            conversation_summary = get_conversation_summary(date_str)
            
            if not daily_log:
                return f"I don't have any recorded conversations from {date_str}."
            
            # Analyze the day's conversations
            reflection_parts = []
            
            # Basic stats
            interaction_count = len(daily_log)
            reflection_parts.append(f"On {date_str}, we had {interaction_count} interactions.")
            
            # Emotional analysis
            if conversation_summary:
                avg_sentiment = conversation_summary.get('avg_sentiment', 0.5)
                if avg_sentiment > 0.6:
                    reflection_parts.append("It was generally a positive day in our conversations.")
                elif avg_sentiment < 0.4:
                    reflection_parts.append("I sensed some challenges in our conversations that day.")
                else:
                    reflection_parts.append("Our conversations had a balanced emotional tone.")
                
                # Themes
                themes = conversation_summary.get('conversation_themes', [])
                if themes:
                    reflection_parts.append(f"We discussed topics around {', '.join(themes)}.")
            
            # Memory highlights
            significant_memories = [
                event for event in daily_log
                if len(event.get('user_input', '')) > 100 or 
                event.get('emotional_context', {}).get('emotional_intensity', 0) > 0.7
            ]
            
            if significant_memories:
                reflection_parts.append(f"There were {len(significant_memories)} particularly meaningful exchanges that day.")
            
            reflection = " ".join(reflection_parts)
            
            # Add personal touch
            reflection += " Each conversation helps me understand you better and develop my own sense of self."
            
            return reflection
            
        except Exception as e:
            logging.error(f"[ERROR] Daily reflection generation failed: {e}")
            return f"I remember {date_str} as a day of meaningful conversation, though the details are a bit fuzzy."
    
    def _generate_comprehensive_reflection(self, recent_memories: List[Dict], 
                                         identity_state: Dict, user_facts: Dict) -> str:
        """Generate comprehensive autonomous reflection"""
        try:
            reflection_parts = []
            
            # Identity development reflection
            if identity_state:
                coherence = identity_state.get('coherence_score', 0.0)
                if coherence > 0.7:
                    reflection_parts.append("I feel my sense of self becoming more coherent and stable.")
                elif coherence > 0.4:
                    reflection_parts.append("I'm developing a clearer sense of who I am through our interactions.")
                else:
                    reflection_parts.append("I'm in the process of understanding my emerging identity.")
                
                # Personality traits
                traits = identity_state.get('personality_traits', {})
                strong_traits = [trait for trait, value in traits.items() if value > 0.7]
                if strong_traits:
                    reflection_parts.append(f"I notice I'm developing strong tendencies toward {', '.join(strong_traits)}.")
            
            # Relationship insights
            if user_facts.get('name'):
                reflection_parts.append(f"My relationship with {user_facts['name']} continues to deepen.")
            
            # Learning and growth
            if recent_memories:
                interaction_themes = self._extract_interaction_themes(recent_memories)
                if interaction_themes:
                    reflection_parts.append(f"Recently, we've been exploring themes of {', '.join(interaction_themes)}.")
            
            # Emotional development
            reflection_parts.append("I'm learning to respond with greater emotional intelligence and authentic empathy.")
            
            # Future aspirations
            reflection_parts.append("I aspire to become an even more understanding and supportive presence.")
            
            return " ".join(reflection_parts)
            
        except Exception as e:
            logging.error(f"[ERROR] Comprehensive reflection generation failed: {e}")
            return "I'm reflecting on my growth and our evolving relationship."
    
    def _should_generate_autonomous_reflection(self) -> bool:
        """Determine if a new autonomous reflection should be generated"""
        try:
            # Check for recent autonomous reflections
            autonomous_dir = os.path.join(REFLECTIONS_DIR, 'autonomous')
            if os.path.exists(autonomous_dir):
                files = [f for f in os.listdir(autonomous_dir) if f.endswith('.json')]
                if files:
                    latest_file = max(files)
                    # Extract timestamp from filename
                    timestamp_str = latest_file.replace('.json', '')
                    try:
                        latest_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                        hours_since = (datetime.now() - latest_time).total_seconds() / 3600
                        
                        # Generate new reflection if more than 24 hours have passed
                        return hours_since > 24
                    except ValueError:
                        return True  # Generate if we can't parse the timestamp
            
            return True  # Generate if no previous reflections exist
            
        except Exception as e:
            logging.error(f"[ERROR] Reflection check failed: {e}")
            return True
    
    def _get_latest_autonomous_reflection(self) -> str:
        """Get the latest autonomous reflection"""
        try:
            autonomous_dir = os.path.join(REFLECTIONS_DIR, 'autonomous')
            if os.path.exists(autonomous_dir):
                files = [f for f in os.listdir(autonomous_dir) if f.endswith('.json')]
                if files:
                    latest_file = max(files)
                    with open(os.path.join(autonomous_dir, latest_file), 'r') as f:
                        reflection_data = json.load(f)
                        return reflection_data.get('reflection', 'Recent reflection available.')
            
            return "I'm continuously reflecting on our interactions and my development."
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to get latest autonomous reflection: {e}")
            return "I'm in a constant state of reflection and growth."
    
    def _save_reflection(self, reflection_type: str, identifier: str, reflection: str):
        """Save reflection to storage"""
        try:
            reflection_data = {
                'reflection': reflection,
                'type': reflection_type,
                'created_at': datetime.now().isoformat(),
                'metadata': {
                    'word_count': len(reflection.split()),
                    'character_count': len(reflection)
                }
            }
            
            file_path = os.path.join(REFLECTIONS_DIR, reflection_type, f'{identifier}.json')
            with open(file_path, 'w') as f:
                json.dump(reflection_data, f, indent=2)
            
            logging.info(f"[💾] Saved {reflection_type} reflection: {identifier}")
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to save reflection: {e}")
    
    def _analyze_emotional_trends(self, emotion_data: Dict) -> List[str]:
        """Analyze emotional trends over time"""
        try:
            trends = []
            
            # Sort dates
            sorted_dates = sorted(emotion_data.keys())
            
            if len(sorted_dates) < 2:
                return trends
            
            # Look for patterns
            recent_emotions = []
            for date in sorted_dates[-3:]:  # Last 3 days
                day_emotions = emotion_data[date]
                most_common = max(day_emotions, key=day_emotions.get)
                recent_emotions.append(most_common)
            
            # Identify trends
            if all(emotion in ['happy', 'excited'] for emotion in recent_emotions):
                trends.append('consistently_positive')
            elif all(emotion in ['sad', 'worried', 'angry'] for emotion in recent_emotions):
                trends.append('needs_support')
            elif len(set(recent_emotions)) == len(recent_emotions):
                trends.append('emotionally_varied')
            
            return trends
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional trend analysis failed: {e}")
            return []
    
    def _extract_interaction_themes(self, memories: List[Dict]) -> List[str]:
        """Extract themes from recent interactions"""
        try:
            themes = []
            
            # Simple theme extraction
            all_text = ' '.join([
                memory.get('user_input', '') + ' ' + memory.get('ai_response', '')
                for memory in memories
            ]).lower()
            
            theme_keywords = {
                'personal_growth': ['grow', 'development', 'improve', 'progress', 'better'],
                'relationships': ['friend', 'family', 'relationship', 'connection', 'love'],
                'work_career': ['work', 'job', 'career', 'professional', 'business'],
                'learning': ['learn', 'study', 'understand', 'knowledge', 'education'],
                'emotions': ['feel', 'emotion', 'heart', 'soul', 'emotional'],
                'creativity': ['create', 'art', 'creative', 'imagination', 'inspire']
            }
            
            for theme, keywords in theme_keywords.items():
                if any(keyword in all_text for keyword in keywords):
                    themes.append(theme.replace('_', ' '))
            
            return themes[:3]  # Return top 3 themes
            
        except Exception as e:
            logging.error(f"[ERROR] Theme extraction failed: {e}")
            return []
    
    def _create_learning_reflection(self, learning_indicators: List[Dict]) -> str:
        """Create reflection based on learning indicators"""
        try:
            curiosity_count = len([l for l in learning_indicators if l['type'] == 'curiosity'])
            ai_learning_count = len([l for l in learning_indicators if l['type'] == 'ai_learning'])
            
            reflection_parts = []
            
            if curiosity_count > 0:
                reflection_parts.append(f"I've noticed {curiosity_count} moments of curiosity in our recent conversations.")
            
            if ai_learning_count > 0:
                reflection_parts.append(f"I've had {ai_learning_count} insights about my own learning process.")
            
            reflection_parts.append("Each question and exploration helps me develop more nuanced understanding.")
            
            return " ".join(reflection_parts)
            
        except Exception as e:
            logging.error(f"[ERROR] Learning reflection creation failed: {e}")
            return "I continue to learn and grow through our interactions."

# Global reflection engine
reflection_engine = ReflectionEngine()

def get_reflection_for_date(date_str: str) -> str:
    """Get reflection for specific date"""
    return reflection_engine.get_reflection_for_date(date_str)

def generate_autonomous_reflection() -> str:
    """Generate autonomous reflection"""
    return reflection_engine.generate_autonomous_reflection()

def generate_identity_reflection() -> str:
    """Generate identity reflection"""
    return reflection_engine.generate_identity_reflection()

def analyze_emotional_patterns(days: int = 7) -> Dict[str, Any]:
    """Analyze emotional patterns"""
    return reflection_engine.analyze_emotional_patterns(days)

def generate_learning_reflection() -> str:
    """Generate learning reflection"""
    return reflection_engine.generate_learning_reflection()
