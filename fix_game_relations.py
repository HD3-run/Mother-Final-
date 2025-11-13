#!/usr/bin/env python3
"""
Fix Incorrect Game Relations

This script fixes incorrectly stored game relationships.
Specifically, it moves games that were stored under "love" relation
to "play" relation when they should be games.
"""

import logging
from memory.structured_store import get_knowledge_graph, save_knowledge_graph
from memory.knowledge_graph import KnowledgeGraph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Games that should be under "play" relation, not "love"
GAME_NAMES = ["valorant", "pubg", "counter strike", "counterstrike", "call of duty", 
              "videogames", "video games", "csgo", "cs:go", "league of legends", "lol",
              "dota", "overwatch", "fortnite", "apex legends", "rainbow six", "r6"]

def fix_game_relations():
    """Fix incorrectly stored game relationships."""
    graph = get_knowledge_graph()
    
    # Find "i" or "user" node
    user_node = graph.get_node_by_name("i") or graph.get_node_by_name("user")
    if not user_node:
        logger.warning("No user node found in knowledge graph")
        return
    
    logger.info(f"Found user node: {user_node.name} (ID: {user_node.id})")
    
    # Get all edges from user node
    edges = graph.get_edges_from_node(user_node.id)
    
    # Find "love" edges that point to games
    games_to_fix = []
    for edge in edges:
        if edge.type.lower() in ["love", "loves"]:
            target_node = graph.get_node_by_id(edge.target)
            if target_node:
                target_name_lower = target_node.name.lower()
                # Check if this is a game
                is_game = any(game_name in target_name_lower or target_name_lower in game_name 
                             for game_name in GAME_NAMES)
                
                if is_game:
                    games_to_fix.append({
                        "edge": edge,
                        "target_node": target_node,
                        "target_name": target_node.name
                    })
                    logger.info(f"Found game incorrectly stored as 'love': {target_node.name}")
    
    if not games_to_fix:
        logger.info("No games found with incorrect 'love' relation")
        return
    
    logger.info(f"\nFound {len(games_to_fix)} games to fix:")
    for item in games_to_fix:
        logger.info(f"  - {item['target_name']} (currently: love)")
    
    # Fix each game
    fixed_count = 0
    for item in games_to_fix:
        edge = item["edge"]
        target_node = item["target_node"]
        
        # Check if "play" edge already exists
        existing_play_edges = [
            e for e in graph.get_edges_from_node(user_node.id)
            if e.type.lower() in ["play", "plays"] and e.target == target_node.id
        ]
        
        if existing_play_edges:
            logger.info(f"  '{target_node.name}' already has 'play' relation, removing 'love' relation")
            # Just remove the "love" edge
            graph.remove_edge(edge)
            fixed_count += 1
        else:
            # Remove "love" edge and add "play" edge
            logger.info(f"  Fixing '{target_node.name}': removing 'love', adding 'play'")
            
            # Remove old edge
            graph.remove_edge(edge)
            
            # Add new "play" edge with same properties
            graph.add_edge(
                source_node=user_node,
                target_node=target_node,
                relation_type="play",
                weight=edge.weight,
                properties=edge.properties.copy() if edge.properties else {}
            )
            fixed_count += 1
    
    # Save the graph
    if fixed_count > 0:
        save_knowledge_graph()
        logger.info(f"\n✅ Fixed {fixed_count} game relationships!")
        logger.info("Games are now correctly stored under 'play' relation instead of 'love'")
    else:
        logger.info("\n⚠️  No changes were made")

if __name__ == "__main__":
    try:
        fix_game_relations()
    except Exception as e:
        logger.error(f"Error fixing game relations: {e}")
        import traceback
        traceback.print_exc()

