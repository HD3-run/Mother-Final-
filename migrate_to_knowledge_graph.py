#!/usr/bin/env python3
"""
Migration script to convert existing facts.json to knowledge graph format.

This script:
1. Loads existing facts from data/structured_memory/facts.json
2. Converts them to knowledge graph nodes and relationships
3. Saves the graph to data/structured_memory/knowledge_graph.json
4. Preserves backward compatibility by keeping facts.json
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

from memory.knowledge_graph import KnowledgeGraph, ConceptNode, RelationshipEdge

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_facts_to_graph():
    """Migrate existing facts.json to knowledge graph format"""
    logger.info("[🔄] Starting migration from facts.json to knowledge graph...")
    
    facts_file = Path('data/structured_memory/facts.json')
    graph_file = Path('data/structured_memory/knowledge_graph.json')
    
    # Check if facts file exists
    if not facts_file.exists():
        logger.info("[ℹ️] No facts.json found. Creating empty knowledge graph.")
        graph = KnowledgeGraph()
        graph.save_to_file(graph_file)
        return True
    
    # Load existing facts
    try:
        with open(facts_file, 'r') as f:
            facts_data = json.load(f)
        logger.info(f"[📖] Loaded {len(facts_data)} facts from facts.json")
    except Exception as e:
        logger.error(f"[❌] Failed to load facts.json: {e}")
        return False
    
    # Create knowledge graph
    graph = KnowledgeGraph.load_from_file(graph_file) if graph_file.exists() else KnowledgeGraph()
    
    # Convert facts to graph nodes and relationships
    migrated_count = 0
    
    for key, fact_data in facts_data.items():
        try:
            value = fact_data.get('value') if isinstance(fact_data, dict) else fact_data
            
            # Create user node
            user_node = ConceptNode("user", node_type="person")
            user_node = graph.add_node(user_node)
            
            # Handle different fact types
            if key == "pets" and isinstance(value, dict):
                # Handle pets with names and types
                names = value.get("names", [])
                types = value.get("types", [])
                
                if isinstance(names, str):
                    names = [names]
                if isinstance(types, str):
                    types = [types]
                
                for i, name in enumerate(names):
                    pet_node = ConceptNode(name.lower(), node_type="pet")
                    pet_node = graph.add_node(pet_node)
                    
                    pet_type = types[i] if i < len(types) else "pet"
                    graph.add_edge(
                        user_node,
                        pet_node,
                        "has_pet",
                        weight=0.9,
                        properties={
                            "provenance": "user",
                            "type": pet_type,
                            "migrated_from": "facts.json",
                        },
                    )
                    migrated_count += 1
            
            elif isinstance(value, str):
                # Simple string facts
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
                    properties={
                        "provenance": "user",
                        "migrated_from": "facts.json",
                    },
                )
                migrated_count += 1
            
            elif isinstance(value, (int, float)):
                # Numeric facts (like age)
                if key == "age":
                    value_node = ConceptNode(str(value), node_type="value")
                    value_node = graph.add_node(value_node)
                    graph.add_edge(
                        user_node,
                        value_node,
                        "has_age",
                        weight=0.9,
                        properties={
                            "provenance": "user",
                            "migrated_from": "facts.json",
                        },
                    )
                    migrated_count += 1
            
            elif isinstance(value, list):
                # List facts (like hobbies)
                for item in value:
                    if isinstance(item, str):
                        item_node = ConceptNode(item.lower(), node_type="value")
                        item_node = graph.add_node(item_node)
                        graph.add_edge(
                            user_node,
                            item_node,
                            f"has_{key}",
                            weight=0.9,
                            properties={
                                "provenance": "user",
                                "migrated_from": "facts.json",
                            },
                        )
                        migrated_count += 1
        
        except Exception as e:
            logger.warning(f"[⚠️] Failed to migrate fact '{key}': {e}")
            continue
    
    # Save graph
    try:
        graph.save_to_file(graph_file)
        logger.info(f"[✅] Migration complete! Migrated {migrated_count} facts to knowledge graph.")
        logger.info(f"[📊] Graph now has {len(graph.graph.nodes)} nodes and {len(graph.graph.edges)} edges.")
        return True
    except Exception as e:
        logger.error(f"[❌] Failed to save knowledge graph: {e}")
        return False


if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs('data/structured_memory', exist_ok=True)
    
    success = migrate_facts_to_graph()
    if success:
        logger.info("[✅] Migration completed successfully!")
    else:
        logger.error("[❌] Migration failed!")
        exit(1)

