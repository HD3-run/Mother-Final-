import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter
import threading

USAGE_STATS_FILE = 'data/usage_stats.json'
INTERACTION_LOG_FILE = 'data/interaction_log.jsonl'

class UsageTracker:
    """Track system usage and interaction patterns"""
    
    def __init__(self):
        self.session_stats = {
            'session_start': datetime.now().isoformat(),
            'user_inputs': 0,
            'ai_responses': 0,
            'total_characters_input': 0,
            'total_characters_output': 0,
            'interaction_types': defaultdict(int),
            'emotional_states': defaultdict(int)
        }
        
        self.daily_stats = {}
        self.load_usage_stats()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def load_usage_stats(self):
        """Load existing usage statistics"""
        try:
            if os.path.exists(USAGE_STATS_FILE):
                with open(USAGE_STATS_FILE, 'r') as f:
                    data = json.load(f)
                    self.daily_stats = data.get('daily_stats', {})
                    
                logging.info(f"[📊] Loaded usage stats for {len(self.daily_stats)} days")
            else:
                self.daily_stats = {}
                
        except Exception as e:
            logging.error(f"[ERROR] Failed to load usage stats: {e}")
            self.daily_stats = {}
    
    def save_usage_stats(self):
        """Save usage statistics to file"""
        try:
            os.makedirs('data', exist_ok=True)
            
            stats_data = {
                'daily_stats': self.daily_stats,
                'last_updated': datetime.now().isoformat(),
                'total_days_tracked': len(self.daily_stats),
                'current_session': self.session_stats
            }
            
            with open(USAGE_STATS_FILE, 'w') as f:
                json.dump(stats_data, f, indent=2)
                
        except Exception as e:
            logging.error(f"[ERROR] Failed to save usage stats: {e}")
    
    def log_user_input(self, user_input: str = "", intent: str = "general", 
                      emotional_context: Dict = None):
        """Log user input event"""
        try:
            with self._lock:
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Update session stats
                self.session_stats['user_inputs'] += 1
                self.session_stats['total_characters_input'] += len(user_input)
                self.session_stats['interaction_types'][intent] += 1
                
                if emotional_context:
                    emotion = emotional_context.get('dominant_emotion', 'neutral')
                    self.session_stats['emotional_states'][emotion] += 1
                
                # Update daily stats
                if today not in self.daily_stats:
                    self.daily_stats[today] = {
                        'user_inputs': 0,
                        'ai_responses': 0,
                        'total_characters_input': 0,
                        'total_characters_output': 0,
                        'unique_sessions': set(),
                        'interaction_types': defaultdict(int),
                        'emotional_states': defaultdict(int),
                        'hourly_distribution': defaultdict(int),
                        'conversation_quality_scores': []
                    }
                
                daily_stats = self.daily_stats[today]
                daily_stats['user_inputs'] += 1
                daily_stats['total_characters_input'] += len(user_input)
                daily_stats['interaction_types'][intent] += 1
                daily_stats['hourly_distribution'][datetime.now().hour] += 1
                
                if emotional_context:
                    emotion = emotional_context.get('dominant_emotion', 'neutral')
                    daily_stats['emotional_states'][emotion] += 1
                
                # Log detailed interaction
                self._log_interaction('user_input', {
                    'content_length': len(user_input),
                    'intent': intent,
                    'emotional_context': emotional_context
                })
                
                # Periodic save
                if self.session_stats['user_inputs'] % 10 == 0:
                    self.save_usage_stats()
                
        except Exception as e:
            logging.error(f"[ERROR] User input logging failed: {e}")
    
    def log_response(self, ai_response: str = "", processing_time: float = 0.0,
                    model_used: str = "unknown"):
        """Log AI response event"""
        try:
            with self._lock:
                today = datetime.now().strftime('%Y-%m-%d')
                
                # Update session stats
                self.session_stats['ai_responses'] += 1
                self.session_stats['total_characters_output'] += len(ai_response)
                
                # Update daily stats
                if today in self.daily_stats:
                    daily_stats = self.daily_stats[today]
                    daily_stats['ai_responses'] += 1
                    daily_stats['total_characters_output'] += len(ai_response)
                
                # Log detailed interaction
                self._log_interaction('ai_response', {
                    'content_length': len(ai_response),
                    'processing_time': processing_time,
                    'model_used': model_used
                })
                
        except Exception as e:
            logging.error(f"[ERROR] AI response logging failed: {e}")
    
    def log_conversation_quality(self, quality_score: float):
        """Log conversation quality assessment"""
        try:
            with self._lock:
                today = datetime.now().strftime('%Y-%m-%d')
                
                if today in self.daily_stats:
                    self.daily_stats[today]['conversation_quality_scores'].append(quality_score)
                
        except Exception as e:
            logging.error(f"[ERROR] Conversation quality logging failed: {e}")
    
    def _log_interaction(self, interaction_type: str, data: Dict):
        """Log detailed interaction to JSONL file"""
        try:
            interaction_entry = {
                'timestamp': datetime.now().isoformat(),
                'type': interaction_type,
                'data': data,
                'session_id': self.session_stats['session_start']
            }
            
            os.makedirs('data', exist_ok=True)
            with open(INTERACTION_LOG_FILE, 'a') as f:
                f.write(json.dumps(interaction_entry) + '\n')
                
        except Exception as e:
            logging.error(f"[ERROR] Detailed interaction logging failed: {e}")
    
    def get_usage_stats(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive usage statistics"""
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Aggregate stats for the period
            period_stats = {
                'total_user_inputs': 0,
                'total_ai_responses': 0,
                'total_characters_input': 0,
                'total_characters_output': 0,
                'daily_averages': {},
                'interaction_types': defaultdict(int),
                'emotional_states': defaultdict(int),
                'hourly_patterns': defaultdict(int),
                'conversation_quality': {
                    'average_score': 0.0,
                    'score_distribution': []
                },
                'engagement_metrics': {},
                'growth_metrics': {}
            }
            
            valid_days = 0
            all_quality_scores = []
            
            # Aggregate daily stats
            for i in range(days):
                date = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
                
                if date in self.daily_stats:
                    day_stats = self.daily_stats[date]
                    valid_days += 1
                    
                    period_stats['total_user_inputs'] += day_stats.get('user_inputs', 0)
                    period_stats['total_ai_responses'] += day_stats.get('ai_responses', 0)
                    period_stats['total_characters_input'] += day_stats.get('total_characters_input', 0)
                    period_stats['total_characters_output'] += day_stats.get('total_characters_output', 0)
                    
                    # Aggregate interaction types
                    for itype, count in day_stats.get('interaction_types', {}).items():
                        period_stats['interaction_types'][itype] += count
                    
                    # Aggregate emotional states
                    for emotion, count in day_stats.get('emotional_states', {}).items():
                        period_stats['emotional_states'][emotion] += count
                    
                    # Aggregate hourly patterns
                    for hour, count in day_stats.get('hourly_distribution', {}).items():
                        period_stats['hourly_patterns'][int(hour)] += count
                    
                    # Collect quality scores
                    quality_scores = day_stats.get('conversation_quality_scores', [])
                    all_quality_scores.extend(quality_scores)
            
            # Calculate averages
            if valid_days > 0:
                period_stats['daily_averages'] = {
                    'user_inputs': period_stats['total_user_inputs'] / valid_days,
                    'ai_responses': period_stats['total_ai_responses'] / valid_days,
                    'characters_input': period_stats['total_characters_input'] / valid_days,
                    'characters_output': period_stats['total_characters_output'] / valid_days
                }
            
            # Calculate conversation quality metrics
            if all_quality_scores:
                period_stats['conversation_quality']['average_score'] = sum(all_quality_scores) / len(all_quality_scores)
                period_stats['conversation_quality']['score_distribution'] = all_quality_scores
            
            # Calculate engagement metrics
            period_stats['engagement_metrics'] = self._calculate_engagement_metrics(period_stats)
            
            # Calculate growth metrics
            period_stats['growth_metrics'] = self._calculate_growth_metrics(days)
            
            # Add current session info
            period_stats['current_session'] = self.session_stats.copy()
            period_stats['period_info'] = {
                'days_requested': days,
                'days_with_data': valid_days,
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
            
            return period_stats
            
        except Exception as e:
            logging.error(f"[ERROR] Usage stats retrieval failed: {e}")
            return {
                'total_user_inputs': 0,
                'total_ai_responses': 0,
                'error': str(e)
            }
    
    def _calculate_engagement_metrics(self, stats: Dict) -> Dict[str, Any]:
        """Calculate user engagement metrics"""
        try:
            metrics = {}
            
            # Response rate
            total_inputs = stats['total_user_inputs']
            total_outputs = stats['total_ai_responses']
            
            if total_inputs > 0:
                metrics['response_rate'] = total_outputs / total_inputs
            else:
                metrics['response_rate'] = 0.0
            
            # Average interaction length
            if total_inputs > 0:
                metrics['avg_input_length'] = stats['total_characters_input'] / total_inputs
            else:
                metrics['avg_input_length'] = 0.0
            
            if total_outputs > 0:
                metrics['avg_output_length'] = stats['total_characters_output'] / total_outputs
            else:
                metrics['avg_output_length'] = 0.0
            
            # Interaction diversity
            interaction_types = stats['interaction_types']
            if interaction_types:
                total_interactions = sum(interaction_types.values())
                entropy = 0
                for count in interaction_types.values():
                    if count > 0:
                        p = count / total_interactions
                        entropy -= p * (p ** 0.5)  # Simple diversity measure
                metrics['interaction_diversity'] = entropy
            else:
                metrics['interaction_diversity'] = 0.0
            
            # Emotional engagement
            emotional_states = stats['emotional_states']
            if emotional_states:
                total_emotional = sum(emotional_states.values())
                emotional_variety = len(emotional_states)
                metrics['emotional_engagement'] = emotional_variety / max(total_emotional, 1)
            else:
                metrics['emotional_engagement'] = 0.0
            
            return metrics
            
        except Exception as e:
            logging.error(f"[ERROR] Engagement metrics calculation failed: {e}")
            return {}
    
    def _calculate_growth_metrics(self, days: int) -> Dict[str, Any]:
        """Calculate growth and trend metrics"""
        try:
            metrics = {}
            
            if len(self.daily_stats) < 2:
                return {'insufficient_data': True}
            
            # Get sorted dates
            sorted_dates = sorted(self.daily_stats.keys())
            
            # Calculate recent vs. earlier periods
            mid_point = len(sorted_dates) // 2
            earlier_period = sorted_dates[:mid_point]
            recent_period = sorted_dates[mid_point:]
            
            # Calculate averages for each period
            earlier_avg = self._calculate_period_average(earlier_period)
            recent_avg = self._calculate_period_average(recent_period)
            
            # Calculate growth rates
            if earlier_avg['user_inputs'] > 0:
                metrics['interaction_growth_rate'] = (
                    (recent_avg['user_inputs'] - earlier_avg['user_inputs']) / 
                    earlier_avg['user_inputs']
                ) * 100
            else:
                metrics['interaction_growth_rate'] = 0.0
            
            if earlier_avg['avg_input_length'] > 0:
                metrics['engagement_depth_growth'] = (
                    (recent_avg['avg_input_length'] - earlier_avg['avg_input_length']) / 
                    earlier_avg['avg_input_length']
                ) * 100
            else:
                metrics['engagement_depth_growth'] = 0.0
            
            # Trend analysis
            daily_inputs = [self.daily_stats[date].get('user_inputs', 0) for date in sorted_dates[-7:]]
            if len(daily_inputs) > 1:
                # Simple trend calculation
                x = list(range(len(daily_inputs)))
                y = daily_inputs
                
                # Linear regression slope
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_x2 = sum(x[i] * x[i] for i in range(n))
                
                if n * sum_x2 - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    
                    if slope > 0.1:
                        metrics['trend'] = 'increasing'
                    elif slope < -0.1:
                        metrics['trend'] = 'decreasing'
                    else:
                        metrics['trend'] = 'stable'
                else:
                    metrics['trend'] = 'stable'
            else:
                metrics['trend'] = 'insufficient_data'
            
            return metrics
            
        except Exception as e:
            logging.error(f"[ERROR] Growth metrics calculation failed: {e}")
            return {}
    
    def _calculate_period_average(self, dates: List[str]) -> Dict[str, float]:
        """Calculate averages for a period"""
        try:
            if not dates:
                return {'user_inputs': 0.0, 'avg_input_length': 0.0}
            
            total_inputs = 0
            total_chars = 0
            
            for date in dates:
                if date in self.daily_stats:
                    day_stats = self.daily_stats[date]
                    total_inputs += day_stats.get('user_inputs', 0)
                    total_chars += day_stats.get('total_characters_input', 0)
            
            avg_inputs = total_inputs / len(dates)
            avg_length = total_chars / max(total_inputs, 1)
            
            return {
                'user_inputs': avg_inputs,
                'avg_input_length': avg_length
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Period average calculation failed: {e}")
            return {'user_inputs': 0.0, 'avg_input_length': 0.0}
    
    def get_interaction_patterns(self) -> Dict[str, Any]:
        """Analyze interaction patterns"""
        try:
            patterns = {
                'most_active_hours': [],
                'preferred_interaction_types': [],
                'emotional_patterns': [],
                'session_length_patterns': []
            }
            
            # Analyze hourly patterns across all days
            hourly_totals = defaultdict(int)
            interaction_type_totals = defaultdict(int)
            emotion_totals = defaultdict(int)
            
            for day_stats in self.daily_stats.values():
                for hour, count in day_stats.get('hourly_distribution', {}).items():
                    hourly_totals[int(hour)] += count
                
                for itype, count in day_stats.get('interaction_types', {}).items():
                    interaction_type_totals[itype] += count
                
                for emotion, count in day_stats.get('emotional_states', {}).items():
                    emotion_totals[emotion] += count
            
            # Most active hours
            if hourly_totals:
                sorted_hours = sorted(hourly_totals.items(), key=lambda x: x[1], reverse=True)
                patterns['most_active_hours'] = sorted_hours[:3]
            
            # Preferred interaction types
            if interaction_type_totals:
                sorted_types = sorted(interaction_type_totals.items(), key=lambda x: x[1], reverse=True)
                patterns['preferred_interaction_types'] = sorted_types[:5]
            
            # Emotional patterns
            if emotion_totals:
                sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
                patterns['emotional_patterns'] = sorted_emotions[:5]
            
            return patterns
            
        except Exception as e:
            logging.error(f"[ERROR] Interaction pattern analysis failed: {e}")
            return {}
    
    def cleanup_old_data(self, retention_days: int = 90):
        """Clean up old usage data"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=retention_days)).strftime('%Y-%m-%d')
            
            # Remove old daily stats
            dates_to_remove = [date for date in self.daily_stats.keys() if date < cutoff_date]
            
            for date in dates_to_remove:
                del self.daily_stats[date]
            
            if dates_to_remove:
                self.save_usage_stats()
                logging.info(f"[🧹] Cleaned up usage data for {len(dates_to_remove)} old days")
            
            # Clean up interaction log
            if os.path.exists(INTERACTION_LOG_FILE):
                temp_file = INTERACTION_LOG_FILE + '.tmp'
                cutoff_timestamp = datetime.now() - timedelta(days=retention_days)
                
                with open(INTERACTION_LOG_FILE, 'r') as infile, open(temp_file, 'w') as outfile:
                    for line in infile:
                        try:
                            entry = json.loads(line.strip())
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            
                            if entry_time > cutoff_timestamp:
                                outfile.write(line)
                        except (json.JSONDecodeError, KeyError):
                            continue
                
                os.replace(temp_file, INTERACTION_LOG_FILE)
                logging.info("[🧹] Cleaned up old interaction logs")
            
            return True
            
        except Exception as e:
            logging.error(f"[ERROR] Usage data cleanup failed: {e}")
            return False

# Global usage tracker instance
usage_tracker = UsageTracker()

def log_user_input(user_input: str = "", intent: str = "general", 
                  emotional_context: Dict = None):
    """Log user input"""
    usage_tracker.log_user_input(user_input, intent, emotional_context)

def log_response(ai_response: str = "", processing_time: float = 0.0,
                model_used: str = "unknown"):
    """Log AI response"""
    usage_tracker.log_response(ai_response, processing_time, model_used)

def log_conversation_quality(quality_score: float):
    """Log conversation quality score"""
    usage_tracker.log_conversation_quality(quality_score)

def get_usage_stats(days: int = 30) -> Dict[str, Any]:
    """Get usage statistics"""
    return usage_tracker.get_usage_stats(days)

def get_interaction_patterns() -> Dict[str, Any]:
    """Get interaction patterns"""
    return usage_tracker.get_interaction_patterns()

def cleanup_old_usage_data(retention_days: int = 90):
    """Clean up old usage data"""
    return usage_tracker.cleanup_old_data(retention_days)
