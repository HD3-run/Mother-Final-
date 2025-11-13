import json
import logging
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

EPISODIC_LOG_DIR = 'data/episodic_memory'
CONVERSATION_INDEX_FILE = 'data/episodic_memory/conversation_index.json'

def init_episodic_logging():
    """Initialize episodic logging system"""
    try:
        os.makedirs(EPISODIC_LOG_DIR, exist_ok=True)
        
        if not os.path.exists(CONVERSATION_INDEX_FILE):
            with open(CONVERSATION_INDEX_FILE, 'w') as f:
                json.dump({
                    'total_conversations': 0,
                    'daily_logs': {},
                    'last_updated': datetime.now().isoformat()
                }, f)
        
        logging.info("[📖] Episodic logging initialized")
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Episodic logging initialization failed: {e}")
        return False

def log_event(user_input: str, ai_response: str, sentiment: float,
              emotional_context: Dict = None, predicted_state: Dict = None):
    """Log an episodic interaction event"""
    try:
        init_episodic_logging()
        
        timestamp = datetime.now()
        date_str = timestamp.strftime('%Y-%m-%d')
        
        # Create event record
        event = {
            'timestamp': timestamp.isoformat(),
            'user_input': user_input,
            'ai_response': ai_response,
            'sentiment': sentiment,
            'emotional_context': emotional_context or {},
            'predicted_state': predicted_state or {},
            'metadata': {
                'input_length': len(user_input),
                'response_length': len(ai_response),
                'interaction_id': f"{date_str}_{timestamp.strftime('%H%M%S')}"
            }
        }
        
        # Load or create daily log
        daily_log_file = os.path.join(EPISODIC_LOG_DIR, f"{date_str}.json")
        
        if os.path.exists(daily_log_file):
            with open(daily_log_file, 'r') as f:
                daily_log = json.load(f)
        else:
            daily_log = {
                'date': date_str,
                'events': [],
                'summary': {
                    'total_interactions': 0,
                    'avg_sentiment': 0.0,
                    'dominant_emotions': [],
                    'conversation_themes': []
                }
            }
        
        # Add event to daily log
        daily_log['events'].append(event)
        daily_log['summary']['total_interactions'] += 1
        
        # Update daily summary
        update_daily_summary(daily_log)
        
        # Save daily log
        with open(daily_log_file, 'w') as f:
            json.dump(daily_log, f, indent=2)
        
        # Update conversation index
        update_conversation_index(date_str, event)
        
        logging.info(f"[📖] Logged episodic event for {date_str}")
        
    except Exception as e:
        logging.error(f"[ERROR] Episodic logging failed: {e}")

def get_today_log() -> List[Dict[str, Any]]:
    """Get today's conversation log"""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        return get_log_for_date(today)
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get today's log: {e}")
        return []

def get_log_for_date(date_str: str) -> List[Dict[str, Any]]:
    """Get conversation log for specific date"""
    try:
        daily_log_file = os.path.join(EPISODIC_LOG_DIR, f"{date_str}.json")
        
        if os.path.exists(daily_log_file):
            with open(daily_log_file, 'r') as f:
                daily_log = json.load(f)
                return daily_log.get('events', [])
        
        return []
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get log for {date_str}: {e}")
        return []

def get_conversation_history(days: int = 7) -> List[Dict[str, Any]]:
    """Get conversation history for specified number of days"""
    try:
        history = []
        
        for i in range(days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%Y-%m-%d')
            daily_events = get_log_for_date(date_str)
            history.extend(daily_events)
        
        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])
        
        return history
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get conversation history: {e}")
        return []

def get_conversation_summary(date_str: str) -> Dict[str, Any]:
    """Get conversation summary for specific date"""
    try:
        daily_log_file = os.path.join(EPISODIC_LOG_DIR, f"{date_str}.json")
        
        if os.path.exists(daily_log_file):
            with open(daily_log_file, 'r') as f:
                daily_log = json.load(f)
                return daily_log.get('summary', {})
        
        return {}
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get conversation summary: {e}")
        return {}

def update_daily_summary(daily_log: Dict[str, Any]):
    """Update daily conversation summary"""
    try:
        events = daily_log.get('events', [])
        if not events:
            return
        
        # Calculate average sentiment
        sentiments = [event.get('sentiment', 0.5) for event in events]
        daily_log['summary']['avg_sentiment'] = sum(sentiments) / len(sentiments)
        
        # Identify dominant emotions
        emotions = []
        for event in events:
            emotional_context = event.get('emotional_context', {})
            emotion = emotional_context.get('dominant_emotion')
            if emotion:
                emotions.append(emotion)
        
        # Count emotion frequencies
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Get top 3 emotions
        top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        daily_log['summary']['dominant_emotions'] = [emotion for emotion, count in top_emotions]
        
        # Extract conversation themes (simple keyword extraction)
        all_text = ' '.join([event.get('user_input', '') for event in events])
        themes = extract_conversation_themes(all_text)
        daily_log['summary']['conversation_themes'] = themes
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to update daily summary: {e}")

def extract_conversation_themes(text: str) -> List[str]:
    """Extract main themes from conversation text"""
    try:
        # Simple theme extraction based on keywords
        theme_keywords = {
            'work': ['work', 'job', 'career', 'office', 'meeting', 'project'],
            'family': ['family', 'parent', 'child', 'sibling', 'relative'],
            'health': ['health', 'doctor', 'medicine', 'exercise', 'sick'],
            'relationship': ['relationship', 'friend', 'partner', 'love', 'dating'],
            'education': ['school', 'study', 'learn', 'education', 'class'],
            'hobby': ['hobby', 'interest', 'fun', 'enjoy', 'play'],
            'emotion': ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited'],
            'goal': ['goal', 'plan', 'future', 'dream', 'ambition']
        }
        
        text_lower = text.lower()
        detected_themes = []
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_themes.append(theme)
        
        return detected_themes[:5]  # Return top 5 themes
        
    except Exception as e:
        logging.error(f"[ERROR] Theme extraction failed: {e}")
        return []

def update_conversation_index(date_str: str, event: Dict[str, Any]):
    """Update the conversation index"""
    try:
        if os.path.exists(CONVERSATION_INDEX_FILE):
            with open(CONVERSATION_INDEX_FILE, 'r') as f:
                index = json.load(f)
        else:
            index = {
                'total_conversations': 0,
                'daily_logs': {},
                'last_updated': datetime.now().isoformat()
            }
        
        # Update index
        index['total_conversations'] += 1
        index['last_updated'] = datetime.now().isoformat()
        
        # Update daily log entry
        if date_str not in index['daily_logs']:
            index['daily_logs'][date_str] = {
                'interaction_count': 0,
                'first_interaction': event['timestamp'],
                'last_interaction': event['timestamp']
            }
        
        index['daily_logs'][date_str]['interaction_count'] += 1
        index['daily_logs'][date_str]['last_interaction'] = event['timestamp']
        
        # Save updated index
        with open(CONVERSATION_INDEX_FILE, 'w') as f:
            json.dump(index, f, indent=2)
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to update conversation index: {e}")

def get_all_conversation_dates(exclude_today: bool = False) -> List[str]:
    """Get all dates we've had conversations, sorted chronologically"""
    try:
        dates = []
        date_set = set()
        
        # Method 1: Check conversation index
        if os.path.exists(CONVERSATION_INDEX_FILE):
            with open(CONVERSATION_INDEX_FILE, 'r') as f:
                index = json.load(f)
            for date_str in index.get('daily_logs', {}).keys():
                # Validate date format (YYYY-MM-DD)
                try:
                    datetime.strptime(date_str, '%Y-%m-%d')
                    if date_str not in date_set:
                        dates.append(date_str)
                        date_set.add(date_str)
                except ValueError:
                    # Skip invalid date formats
                    continue
        
        # Method 2: Also check episodic log directory for any missing dates
        if os.path.exists(EPISODIC_LOG_DIR):
            for filename in os.listdir(EPISODIC_LOG_DIR):
                if filename.endswith('.json'):
                    date_str = filename.replace('.json', '')
                    # Only include valid YYYY-MM-DD format dates
                    try:
                        datetime.strptime(date_str, '%Y-%m-%d')
                        if date_str not in date_set:
                            dates.append(date_str)
                            date_set.add(date_str)
                    except ValueError:
                        # Skip invalid date formats (like timestamps)
                        continue
        
        # Sort dates chronologically
        dates.sort()
        
        # Exclude today if requested
        if exclude_today:
            today = datetime.now().strftime('%Y-%m-%d')
            dates = [d for d in dates if d != today]
        
        return dates
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get conversation dates: {e}")
        return []

def get_episodic_statistics() -> Dict[str, Any]:
    """Get episodic memory statistics"""
    try:
        if os.path.exists(CONVERSATION_INDEX_FILE):
            with open(CONVERSATION_INDEX_FILE, 'r') as f:
                index = json.load(f)
            
            # Calculate additional stats
            daily_logs = index.get('daily_logs', {})
            total_days = len(daily_logs)
            avg_interactions_per_day = 0
            
            if total_days > 0:
                total_interactions = sum(log.get('interaction_count', 0) for log in daily_logs.values())
                avg_interactions_per_day = total_interactions / total_days
            
            return {
                'total_conversations': index.get('total_conversations', 0),
                'total_days': total_days,
                'avg_interactions_per_day': avg_interactions_per_day,
                'last_updated': index.get('last_updated'),
                'most_active_day': max(daily_logs.items(), key=lambda x: x[1]['interaction_count'])[0] if daily_logs else None
            }
        
        return {
            'total_conversations': 0,
            'total_days': 0,
            'avg_interactions_per_day': 0,
            'last_updated': None,
            'most_active_day': None
        }
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get episodic statistics: {e}")
        return {}

# Initialize on import
init_episodic_logging()
