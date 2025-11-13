"""
Structured Memory Store for MOTHER

This module provides a hybrid approach: it uses the KnowledgeGraph for
relationship-based facts while maintaining backward compatibility with
the JSON-based fact storage for simple key-value facts.
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional

from memory.knowledge_graph import KnowledgeGraph, ConceptNode, RelationshipEdge

FACTS_FILE = 'data/structured_memory/facts.json'
MEMORY_INDEX_FILE = 'data/structured_memory/memory_index.json'
KNOWLEDGE_GRAPH_FILE = 'data/structured_memory/knowledge_graph.json'

# Global knowledge graph instance
_knowledge_graph: Optional[KnowledgeGraph] = None


def _get_graph() -> KnowledgeGraph:
    """Get or create the global knowledge graph instance"""
    global _knowledge_graph
    if _knowledge_graph is None:
        _knowledge_graph = KnowledgeGraph.load_from_file(KNOWLEDGE_GRAPH_FILE)
    return _knowledge_graph


def init_db():
    """Initialize the structured memory database"""
    try:
        # Ensure data directory exists
        os.makedirs('data/structured_memory', exist_ok=True)
        
        # Initialize facts file if it doesn't exist (backward compatibility)
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
        
        # Initialize knowledge graph
        graph = _get_graph()
        graph.save_to_file(KNOWLEDGE_GRAPH_FILE)
        logging.info("[💾] Knowledge graph initialized")
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Database initialization failed: {e}")
        return False


def set_fact(key: str, value: Any) -> bool:
    """Store a structured fact (backward compatible, also stores in graph)"""
    try:
        # Store in JSON for backward compatibility
        facts = load_facts()
        facts[key] = {
            'value': value,
            'updated_at': datetime.now().isoformat(),
            'confidence': 1.0,
            'source': 'user_input'
        }
        save_facts(facts)
        
        # Also store in knowledge graph if it's a relationship-type fact
        graph = _get_graph()
        
        # Try to parse as relationship
        if isinstance(value, dict):
            # Handle complex facts (e.g., pets with names)
            if key == "pets" and isinstance(value, dict):
                user_node = ConceptNode("user", node_type="person")
                user_node = graph.add_node(user_node)
                
                if "names" in value:
                    names = value["names"] if isinstance(value["names"], list) else [value["names"]]
                    types = value.get("types", [])
                    if not types:
                        types = ["pet"] * len(names)
                    
                    for i, name in enumerate(names):
                        pet_node = ConceptNode(name.lower(), node_type="pet")
                        pet_node = graph.add_node(pet_node)
                        graph.add_edge(
                            user_node,
                            pet_node,
                            "has_pet",
                            weight=0.9,
                            properties={"provenance": "user", "type": types[i] if i < len(types) else "pet"},
                        )
        elif isinstance(value, str):
            # Simple string facts - store as user attribute
            user_node = ConceptNode("user", node_type="person")
            user_node = graph.add_node(user_node)
            
            # Map common keys to relation types
            relation_map = {
                "name": "has_name",
                "location": "lives_in",
                "age": "has_age",
                "job": "works_as",
                "hobby": "has_hobby",
            }
            
            relation_type = relation_map.get(key, "has_attribute")
            value_node = ConceptNode(value.lower(), node_type="value")
            value_node = graph.add_node(value_node)
            
            graph.add_edge(
                user_node,
                value_node,
                relation_type,
                weight=0.9,
                properties={"provenance": "user"},
            )
        
        # Save graph
        graph.save_to_file(KNOWLEDGE_GRAPH_FILE)
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to set fact {key}: {e}")
        return False


def get_fact(key: str, default=None) -> Any:
    """Retrieve a structured fact (checks both JSON and graph)"""
    try:
        # First check JSON (backward compatibility)
        facts = load_facts()
        fact_data = facts.get(key)
        
        if fact_data:
            return fact_data.get('value', default)
        
        # Also check knowledge graph
        graph = _get_graph()
        user_node = graph.get_node_by_name("user")
        
        if user_node:
            # Map keys to relation types
            relation_map = {
                "name": "has_name",
                "location": "lives_in",
                "age": "has_age",
                "job": "works_as",
                "hobby": "has_hobby",
            }
            
            relation_type = relation_map.get(key)
            if relation_type:
                edges = graph.get_edges_from_node(user_node.id)
                for edge in edges:
                    if edge.type == relation_type:
                        target_node = graph.get_node_by_id(edge.target)
                        if target_node:
                            return target_node.name
            
            # Special handling for pets
            if key == "pets":
                pet_edges = [e for e in graph.get_edges_from_node(user_node.id) if e.type == "has_pet"]
                if pet_edges:
                    pets = {"names": [], "types": []}
                    for edge in pet_edges:
                        pet_node = graph.get_node_by_id(edge.target)
                        if pet_node:
                            pets["names"].append(pet_node.name)
                            pet_type = edge.properties.get("type", "pet")
                            pets["types"].append(pet_type)
                    return pets
        
        return default
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get fact {key}: {e}")
        return default


def all_facts() -> Dict[str, Any]:
    """Get all stored facts (from both JSON and graph)"""
    try:
        # Get from JSON
        facts = load_facts()
        result = {key: data.get('value') for key, data in facts.items()}
        
        # Also get from graph
        graph = _get_graph()
        user_node = graph.get_node_by_name("user")
        
        if user_node:
            edges = graph.get_edges_from_node(user_node.id)
            for edge in edges:
                target_node = graph.get_node_by_id(edge.target)
                if target_node:
                    # Map relation types back to keys
                    key_map = {
                        "has_name": "name",
                        "lives_in": "location",
                        "has_age": "age",
                        "works_as": "job",
                        "has_hobby": "hobby",
                    }
                    key = key_map.get(edge.type)
                    if key and key not in result:
                        result[key] = target_node.name
        
        return result
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get all facts: {e}")
        return {}


def load_facts() -> Dict[str, Any]:
    """Load facts from storage (JSON)"""
    try:
        if os.path.exists(FACTS_FILE):
            with open(FACTS_FILE, 'r') as f:
                return json.load(f)
        return {}
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to load facts: {e}")
        return {}


def save_facts(facts: Dict[str, Any]) -> bool:
    """Save facts to storage (JSON)"""
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
        graph = _get_graph()
        
        if os.path.exists(MEMORY_INDEX_FILE):
            with open(MEMORY_INDEX_FILE, 'r') as f:
                index = json.load(f)
            
            return {
                'total_interactions': index.get('total_interactions', 0),
                'last_updated': index.get('last_updated'),
                'facts_count': len(load_facts()),
                'graph_nodes': len(graph.graph.nodes),
                'graph_edges': len(graph.graph.edges),
                'memory_clusters': len(index.get('memory_clusters', [])),
                'recent_activity': len(index.get('recent_interactions', []))
            }
        
        return {
            'total_interactions': 0,
            'last_updated': None,
            'facts_count': 0,
            'graph_nodes': len(graph.graph.nodes),
            'graph_edges': len(graph.graph.edges),
            'memory_clusters': 0,
            'recent_activity': 0
        }
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to get memory stats: {e}")
        return {}


def search_facts(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search facts based on query (searches both JSON and graph)"""
    try:
        results = []
        
        # Search JSON facts
        facts = load_facts()
        query_lower = query.lower()
        
        for key, data in facts.items():
            value = str(data.get('value', '')).lower()
            if query_lower in key.lower() or query_lower in value:
                results.append({
                    'key': key,
                    'value': data.get('value'),
                    'confidence': data.get('confidence', 1.0),
                    'updated_at': data.get('updated_at'),
                    'source': 'json'
                })
        
        # Search graph
        graph = _get_graph()
        all_nodes = graph.get_all_node_names()
        for node_name in all_nodes:
            if query_lower in node_name.lower():
                node = graph.get_node_by_name(node_name)
                if node:
                    # Find relationships
                    edges = graph.get_edges_from_node(node.id)
                    for edge in edges:
                        target_node = graph.get_node_by_id(edge.target)
                        if target_node:
                            results.append({
                                'key': f"{node_name} --[{edge.type}]--> {target_node.name}",
                                'value': target_node.name,
                                'confidence': edge.weight,
                                'updated_at': edge.properties.get('last_modified'),
                                'source': 'graph'
                            })
        
        # Sort by confidence and recency
        results.sort(key=lambda x: (x.get('confidence', 0), x.get('updated_at', '')), reverse=True)
        
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


def get_knowledge_graph() -> KnowledgeGraph:
    """Get the knowledge graph instance"""
    return _get_graph()


def save_knowledge_graph():
    """Save the knowledge graph to disk"""
    try:
        graph = _get_graph()
        graph.save_to_file(KNOWLEDGE_GRAPH_FILE)
        return True
    except Exception as e:
        logging.error(f"[ERROR] Failed to save knowledge graph: {e}")
        return False
