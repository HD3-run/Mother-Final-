import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

FACTS_FILE = 'data/facts.json'
MEMORY_INDEX_FILE = 'data/memory_index.json'

def init_db():
    """Initialize the structured memory database"""
    try:
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Initialize facts file if it doesn't exist
        if not os.path.exists(FACTS_FILE):
            with open(FACTS_FILE, 'w') as f:
                json.dump({}, f)
            logging.info("[💾] Facts database initialized")
        
        # Initialize memory index if it doesn't exist
        if not os.path.exists(MEMORY_INDEX_FILE):
            with open(MEMORY_INDEX_FILE, 'w') as f:
                json.dump({
                    'total_interactions': 0,
                    'last_updated': datetime.now().isoformat(),
                    'memory_clusters': []
                }, f)
            logging.info("[💾] Memory index initialized")
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Database initialization failed: {e}")
        return False

def set_fact(key: str, value: Any) -> bool:
    """Store a structured fact"""
    try:
        facts = load_facts()
        
        # Store fact with metadata
        facts[key] = {
            'value': value,
            'updated_at': datetime.now().isoformat(),
            'confidence': 1.0,
            'source': 'user_input'
        }
        
        return save_facts(facts)
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to set fact {key}: {e}")
        return False

def get_fact(key: str, default=None) -> Any:
    """Retrieve a structured fact"""
    try:
        facts = load_facts()
        fact_data = facts.get(key)
        
        if fact_data:
            return fact_data.get('value', default)
        return default
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get fact {key}: {e}")
        return default

def all_facts() -> Dict[str, Any]:
    """Get all stored facts"""
    try:
        facts = load_facts()
        return {key: data.get('value') for key, data in facts.items()}
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get all facts: {e}")
        return {}

def load_facts() -> Dict[str, Any]:
    """Load facts from storage"""
    try:
        if os.path.exists(FACTS_FILE):
            with open(FACTS_FILE, 'r') as f:
                return json.load(f)
        return {}
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to load facts: {e}")
        return {}

def save_facts(facts: Dict[str, Any]) -> bool:
    """Save facts to storage"""
    try:
        with open(FACTS_FILE, 'w') as f:
            json.dump(facts, f, indent=2)
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to save facts: {e}")
        return False

def update_memory_index(interaction_data: Dict) -> bool:
    """Update the memory index with new interaction"""
    try:
        if os.path.exists(MEMORY_INDEX_FILE):
            with open(MEMORY_INDEX_FILE, 'r') as f:
                index = json.load(f)
        else:
            index = {
                'total_interactions': 0,
                'last_updated': datetime.now().isoformat(),
                'memory_clusters': []
            }
        
        # Update index
        index['total_interactions'] += 1
        index['last_updated'] = datetime.now().isoformat()
        
        # Store recent interaction metadata
        if 'recent_interactions' not in index:
            index['recent_interactions'] = []
        
        interaction_summary = {
            'timestamp': interaction_data.get('timestamp', datetime.now().isoformat()),
            'intent': interaction_data.get('intent', 'unknown'),
            'sentiment': interaction_data.get('sentiment', 0.5),
            'emotional_context': interaction_data.get('emotional_context', {}),
            'topics': interaction_data.get('topics', [])
        }
        
        index['recent_interactions'].append(interaction_summary)
        
        # Keep only last 50 interactions in index
        if len(index['recent_interactions']) > 50:
            index['recent_interactions'] = index['recent_interactions'][-50:]
        
        # Save updated index
        with open(MEMORY_INDEX_FILE, 'w') as f:
            json.dump(index, f, indent=2)
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to update memory index: {e}")
        return False

def get_memory_stats() -> Dict[str, Any]:
    """Get memory system statistics"""
    try:
        if os.path.exists(MEMORY_INDEX_FILE):
            with open(MEMORY_INDEX_FILE, 'r') as f:
                index = json.load(f)
            
            return {
                'total_interactions': index.get('total_interactions', 0),
                'last_updated': index.get('last_updated'),
                'facts_count': len(load_facts()),
                'memory_clusters': len(index.get('memory_clusters', [])),
                'recent_activity': len(index.get('recent_interactions', []))
            }
        
        return {
            'total_interactions': 0,
            'last_updated': None,
            'facts_count': 0,
            'memory_clusters': 0,
            'recent_activity': 0
        }
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get memory stats: {e}")
        return {}

def search_facts(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search facts based on query"""
    try:
        facts = load_facts()
        results = []
        
        query_lower = query.lower()
        
        for key, data in facts.items():
            value = str(data.get('value', '')).lower()
            if query_lower in key.lower() or query_lower in value:
                results.append({
                    'key': key,
                    'value': data.get('value'),
                    'confidence': data.get('confidence', 1.0),
                    'updated_at': data.get('updated_at')
                })
        
        # Sort by confidence and recency
        results.sort(key=lambda x: (x['confidence'], x['updated_at']), reverse=True)
        
        return results[:limit]
        
    except Exception as e:
        logging.error(f"[ERROR] Fact search failed: {e}")
        return []

def cleanup_old_data(retention_days: int = 90):
    """Clean up old data based on retention policy"""
    try:
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        facts = load_facts()
        
        cleaned_facts = {}
        removed_count = 0
        
        for key, data in facts.items():
            updated_at = datetime.fromisoformat(data.get('updated_at', datetime.now().isoformat()))
            
            if updated_at > cutoff_date:
                cleaned_facts[key] = data
            else:
                removed_count += 1
        
        if removed_count > 0:
            save_facts(cleaned_facts)
            logging.info(f"[🧹] Cleaned up {removed_count} old facts")
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Data cleanup failed: {e}")
        return False
