import logging
import os
import json
from datetime import datetime,timedelta
from typing import Dict, Any, Optional

# Logging configuration
LOG_LEVEL = logging.DEBUG
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DIR = 'logs'

def setup_logging():
    """Setup enhanced logging system"""
    try:
        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Configure root logger
        logging.basicConfig(
            level=LOG_LEVEL,
            format=LOG_FORMAT,
            handlers=[
                logging.FileHandler(os.path.join(LOG_DIR, 'mother_ai.log')),
                logging.StreamHandler()  # Console output
            ]
        )
        
        # Create specialized loggers
        create_specialized_loggers()
        
        logging.info("[📝] Enhanced logging system initialized")
        return True
        
    except Exception as e:
        print(f"[ERROR] Logging setup failed: {e}")
        return False

def create_specialized_loggers():
    """Create specialized loggers for different components"""
    try:
        # Identity development logger
        identity_logger = logging.getLogger('identity')
        identity_handler = logging.FileHandler(os.path.join(LOG_DIR, 'identity_development.log'))
        identity_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        identity_logger.addHandler(identity_handler)
        identity_logger.setLevel(logging.INFO)
        
        # Autonomous decisions logger
        autonomous_logger = logging.getLogger('autonomous')
        autonomous_handler = logging.FileHandler(os.path.join(LOG_DIR, 'autonomous_decisions.log'))
        autonomous_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        autonomous_logger.addHandler(autonomous_handler)
        autonomous_logger.setLevel(logging.INFO)
        
        # Predictive modeling logger
        ml_logger = logging.getLogger('ml')
        ml_handler = logging.FileHandler(os.path.join(LOG_DIR, 'machine_learning.log'))
        ml_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        ml_logger.addHandler(ml_handler)
        ml_logger.setLevel(logging.INFO)
        
        # Emotional analysis logger
        emotion_logger = logging.getLogger('emotion')
        emotion_handler = logging.FileHandler(os.path.join(LOG_DIR, 'emotional_analysis.log'))
        emotion_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        emotion_logger.addHandler(emotion_handler)
        emotion_logger.setLevel(logging.INFO)
        
        # Memory system logger
        memory_logger = logging.getLogger('memory')
        memory_handler = logging.FileHandler(os.path.join(LOG_DIR, 'memory_systems.log'))
        memory_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        memory_logger.addHandler(memory_handler)
        memory_logger.setLevel(logging.INFO)
        
    except Exception as e:
        logging.error(f"[ERROR] Specialized logger creation failed: {e}")

def log_event(message: str, event_type: str = 'general', 
              metadata: Dict[str, Any] = None, logger_name: str = None):
    """Log enhanced event with metadata"""
    try:
        # Choose appropriate logger
        if logger_name:
            logger = logging.getLogger(logger_name)
        else:
            logger = logging.getLogger()
        
        # Create structured log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'message': message,
            'metadata': metadata or {}
        }
        
        # Log based on event type
        if event_type in ['error', 'critical']:
            logger.error(json.dumps(log_entry))
        elif event_type == 'warning':
            logger.warning(json.dumps(log_entry))
        else:
            logger.info(json.dumps(log_entry))
        
        # Save to specialized event log
        save_event_log(log_entry)
        
    except Exception as e:
        logging.error(f"[ERROR] Event logging failed: {e}")

def save_event_log(log_entry: Dict[str, Any]):
    """Save event to structured event log"""
    try:
        event_log_file = os.path.join(LOG_DIR, 'events.jsonl')
        
        with open(event_log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
            
    except Exception as e:
        logging.error(f"[ERROR] Event log saving failed: {e}")

def log_identity_development(event: str, identity_data: Dict, coherence_score: float):
    """Log identity development events"""
    try:
        metadata = {
            'coherence_score': coherence_score,
            'personality_traits': identity_data.get('personality_traits', {}),
            'core_values': identity_data.get('core_values', []),
            'identity_change': True
        }
        
        log_event(
            message=f"Identity development: {event}",
            event_type='identity_development',
            metadata=metadata,
            logger_name='identity'
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Identity development logging failed: {e}")

def log_autonomous_decision(decision_type: str, action_data: Dict, confidence: float):
    """Log autonomous decision making"""
    try:
        metadata = {
            'decision_type': decision_type,
            'confidence': confidence,
            'action_content': action_data.get('content', ''),
            'trigger': action_data.get('trigger', ''),
            'autonomous_action': True
        }
        
        log_event(
            message=f"Autonomous decision: {decision_type}",
            event_type='autonomous_decision',
            metadata=metadata,
            logger_name='autonomous'
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Autonomous decision logging failed: {e}")

def log_predictive_analysis(prediction_type: str, features: Dict, prediction: Dict):
    """Log predictive modeling results"""
    try:
        metadata = {
            'prediction_type': prediction_type,
            'feature_count': len(features),
            'prediction_result': prediction,
            'model_used': True
        }
        
        log_event(
            message=f"Predictive analysis: {prediction_type}",
            event_type='predictive_analysis',
            metadata=metadata,
            logger_name='ml'
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Predictive analysis logging failed: {e}")

def log_emotional_analysis(emotion_data: Dict, analysis_result: Dict):
    """Log emotional analysis events"""
    try:
        metadata = {
            'dominant_emotion': emotion_data.get('dominant_emotion'),
            'emotional_intensity': emotion_data.get('emotional_intensity'),
            'analysis_result': analysis_result,
            'emotional_processing': True
        }
        
        log_event(
            message=f"Emotional analysis: {emotion_data.get('dominant_emotion', 'unknown')}",
            event_type='emotional_analysis',
            metadata=metadata,
            logger_name='emotion'
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Emotional analysis logging failed: {e}")

def log_memory_operation(operation: str, memory_data: Dict, result: Dict):
    """Log memory system operations"""
    try:
        metadata = {
            'operation': operation,
            'memory_type': memory_data.get('type', 'unknown'),
            'result': result,
            'memory_operation': True
        }
        
        log_event(
            message=f"Memory operation: {operation}",
            event_type='memory_operation',
            metadata=metadata,
            logger_name='memory'
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Memory operation logging failed: {e}")

def log_conversation_milestone(milestone: str, conversation_data: Dict):
    """Log significant conversation milestones"""
    try:
        metadata = {
            'milestone': milestone,
            'conversation_count': conversation_data.get('total_interactions', 0),
            'relationship_depth': conversation_data.get('relationship_score', 0),
            'milestone_reached': True
        }
        
        log_event(
            message=f"Conversation milestone: {milestone}",
            event_type='conversation_milestone',
            metadata=metadata
        )
        
    except Exception as e:
        logging.error(f"[ERROR] Conversation milestone logging failed: {e}")

def get_log_summary(hours: int = 24) -> Dict[str, Any]:
    """Get summary of log events for specified time period"""
    try:
        event_log_file = os.path.join(LOG_DIR, 'events.jsonl')
        
        if not os.path.exists(event_log_file):
            return {'total_events': 0, 'event_types': {}}
        
        # Read events from the last N hours
        cutoff_time = datetime.now() - timedelta(hours=hours)
        events = []
        
        with open(event_log_file, 'r') as f:
            for line in f:
                try:
                    event = json.loads(line.strip())
                    event_time = datetime.fromisoformat(event['timestamp'])
                    
                    if event_time > cutoff_time:
                        events.append(event)
                except (json.JSONDecodeError, KeyError):
                    continue
        
        # Analyze events
        event_types = {}
        identity_changes = 0
        autonomous_actions = 0
        emotional_analyses = 0
        memory_operations = 0
        
        for event in events:
            event_type = event.get('event_type', 'unknown')
            event_types[event_type] = event_types.get(event_type, 0) + 1
            
            metadata = event.get('metadata', {})
            if metadata.get('identity_change'):
                identity_changes += 1
            if metadata.get('autonomous_action'):
                autonomous_actions += 1
            if metadata.get('emotional_processing'):
                emotional_analyses += 1
            if metadata.get('memory_operation'):
                memory_operations += 1
        
        return {
            'total_events': len(events),
            'event_types': event_types,
            'identity_changes': identity_changes,
            'autonomous_actions': autonomous_actions,
            'emotional_analyses': emotional_analyses,
            'memory_operations': memory_operations,
            'time_period_hours': hours
        }
        
    except Exception as e:
        logging.error(f"[ERROR] Log summary generation failed: {e}")
        return {'total_events': 0, 'event_types': {}, 'error': str(e)}

def cleanup_old_logs(days: int = 30):
    """Clean up old log files"""
    try:
        cutoff_time = datetime.now() - timedelta(days=days)
        
        # Clean up main log files
        log_files = [
            'mother_ai.log',
            'identity_development.log',
            'autonomous_decisions.log',
            'machine_learning.log',
            'emotional_analysis.log',
            'memory_systems.log'
        ]
        
        for log_file in log_files:
            file_path = os.path.join(LOG_DIR, log_file)
            if os.path.exists(file_path):
                # Get file modification time
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if mod_time < cutoff_time:
                    # Archive old log file
                    archive_name = f"{log_file}.{mod_time.strftime('%Y%m%d')}.archived"
                    archive_path = os.path.join(LOG_DIR, 'archived', archive_name)
                    
                    os.makedirs(os.path.join(LOG_DIR, 'archived'), exist_ok=True)
                    os.rename(file_path, archive_path)
                    
                    logging.info(f"[🧹] Archived old log file: {log_file}")
        
        # Clean up event log (keep recent events only)
        event_log_file = os.path.join(LOG_DIR, 'events.jsonl')
        if os.path.exists(event_log_file):
            temp_file = event_log_file + '.tmp'
            
            with open(event_log_file, 'r') as infile, open(temp_file, 'w') as outfile:
                for line in infile:
                    try:
                        event = json.loads(line.strip())
                        event_time = datetime.fromisoformat(event['timestamp'])
                        
                        if event_time > cutoff_time:
                            outfile.write(line)
                    except (json.JSONDecodeError, KeyError):
                        continue
            
            os.replace(temp_file, event_log_file)
            logging.info("[🧹] Cleaned up old events from event log")
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Log cleanup failed: {e}")
        return False

# Initialize logging on import
setup_logging()
