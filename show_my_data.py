#!/usr/bin/env python3
"""
Show All Data MOTHER Has Stored About You

This script displays a comprehensive view of all information MOTHER has
collected about you, including:
- Personal facts (name, location, age, etc.)
- Games you play
- Things you like/enjoy
- Pets
- Episodic memories
- Vector memories
- Knowledge graph relationships
"""

import json
import logging
import sys
import io
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Import MOTHER's memory systems
from memory.structured_store import get_fact, all_facts, get_knowledge_graph
from memory.vector_store import search_memory
from memory.episodic_logger import get_all_conversation_dates, get_log_for_date
from memory.knowledge_graph import KnowledgeGraph


def get_user_facts_from_graph(graph: KnowledgeGraph) -> Dict[str, List[Dict[str, Any]]]:
    """Get all facts about the user from the knowledge graph."""
    user_facts = {
        "games": [],
        "likes": [],
        "enjoys": [],
        "loves": [],
        "plays": [],
        "has": [],
        "owns": [],
        "lives": [],
        "works": [],
        "pets": [],
        "other": []
    }
    
    # Try to find "i" or "user" node
    user_node_i = graph.get_node_by_name("i")
    user_node_user = graph.get_node_by_name("user")
    
    # Collect edges from both nodes if they exist
    all_edges = []
    if user_node_i:
        all_edges.extend(graph.get_edges_from_node(user_node_i.id))
    if user_node_user:
        all_edges.extend(graph.get_edges_from_node(user_node_user.id))
    
    if not all_edges:
        return user_facts
    
    # Get all edges from the user node(s)
    edges = all_edges
    
    for edge in edges:
        target_node = graph.get_node_by_id(edge.target)
        if not target_node:
            continue
        
        relation_type = edge.type.lower()
        target_name = target_node.name
        
        fact_info = {
            "relation": edge.type,
            "target": target_name,
            "confidence": edge.weight,
            "provenance": edge.properties.get("provenance", "unknown"),
            "properties": edge.properties or {}
        }
        
        # Categorize by relation type
        if relation_type in ["play", "plays"]:
            user_facts["games"].append(fact_info)
            user_facts["plays"].append(fact_info)
        elif relation_type in ["like", "likes"]:
            user_facts["likes"].append(fact_info)
        elif relation_type in ["enjoy", "enjoys"]:
            user_facts["enjoys"].append(fact_info)
        elif relation_type in ["love", "loves"]:
            user_facts["loves"].append(fact_info)
        elif relation_type in ["has_pet"]:
            user_facts["pets"].append(fact_info)
        elif relation_type in ["has", "has_hobby", "has_name"]:
            user_facts["has"].append(fact_info)
        elif relation_type in ["own", "owns"]:
            user_facts["owns"].append(fact_info)
        elif relation_type in ["lives_in", "lives"]:
            user_facts["lives"].append(fact_info)
        elif relation_type in ["works_at", "works_as", "works"]:
            user_facts["works"].append(fact_info)
        else:
            user_facts["other"].append(fact_info)
    
    return user_facts


def format_facts_section(title: str, facts: List[Dict[str, Any]], relation_display: str = None) -> str:
    """Format a section of facts for display."""
    if not facts:
        return ""
    
    lines = [f"\n{'='*60}"]
    lines.append(f"{title}")
    lines.append(f"{'='*60}")
    
    for fact in facts:
        relation = relation_display or fact.get("relation", "related_to")
        target = fact.get("target", "unknown")
        confidence = fact.get("confidence", 0.0)
        provenance = fact.get("provenance", "unknown")
        
        lines.append(f"  • {relation.replace('_', ' ').title()}: {target}")
        lines.append(f"    (Confidence: {confidence:.2f}, Source: {provenance})")
    
    return "\n".join(lines)


def show_all_user_data():
    """Display all data MOTHER has stored about the user."""
    print("\n" + "="*70)
    print("MOTHER'S KNOWLEDGE ABOUT YOU")
    print("="*70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # 1. Structured Facts (JSON-based)
    print("\n" + "="*70)
    print("STRUCTURED FACTS (Personal Information)")
    print("="*70)
    
    structured_facts = all_facts()
    if structured_facts:
        for key, value in structured_facts.items():
            if isinstance(value, dict):
                print(f"  {key.title()}:")
                for sub_key, sub_value in value.items():
                    print(f"    - {sub_key}: {sub_value}")
            else:
                print(f"  {key.title()}: {value}")
    else:
        print("  No structured facts found.")
    
    # 2. Knowledge Graph Facts
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH FACTS")
    print("="*70)
    
    try:
        graph = get_knowledge_graph()
        user_facts = get_user_facts_from_graph(graph)
        
        # Games you play
        if user_facts["games"]:
            print("\n[GAMES] GAMES YOU PLAY:")
            # Deduplicate and clean game names
            seen_games = set()
            for fact in user_facts["games"]:
                game_name = fact['target'].lower().strip()
                # Skip generic terms
                if game_name not in ['video games', 'videogames'] or len(user_facts["games"]) == 1:
                    if game_name not in seen_games:
                        seen_games.add(game_name)
                        print(f"  • {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        # Things you like
        if user_facts["likes"]:
            print("\n[LIKES] THINGS YOU LIKE:")
            for fact in user_facts["likes"]:
                print(f"  • {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        # Things you enjoy
        if user_facts["enjoys"]:
            print("\n[ENJOYS] THINGS YOU ENJOY:")
            for fact in user_facts["enjoys"]:
                print(f"  • {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        # Things you love
        if user_facts["loves"]:
            print("\n[LOVES] THINGS YOU LOVE:")
            # Filter out incomplete entries (like "i play", "and counterstrike")
            filtered_loves = [
                f for f in user_facts["loves"] 
                if not f['target'].lower().startswith(('i ', 'and ', 'to ')) 
                or len(f['target'].split()) > 2
            ]
            for fact in filtered_loves:
                print(f"  • {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        # Pets
        if user_facts["pets"]:
            print("\n[PETS] YOUR PETS:")
            for fact in user_facts["pets"]:
                pet_type = fact.get('properties', {}).get('type', 'pet') if fact.get('properties') else 'pet'
                print(f"  • {fact['target']} ({pet_type}) (Confidence: {fact['confidence']:.2f})")
        
        # Things you have
        if user_facts["has"]:
            print("\n[HAS] THINGS YOU HAVE:")
            for fact in user_facts["has"]:
                relation = fact['relation'].replace('_', ' ').title()
                print(f"  • {relation}: {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        # Things you own
        if user_facts["owns"]:
            print("\n[OWNS] THINGS YOU OWN:")
            for fact in user_facts["owns"]:
                print(f"  • {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        # Location
        if user_facts["lives"]:
            print("\n[LOCATION] WHERE YOU LIVE:")
            for fact in user_facts["lives"]:
                print(f"  • Lives in: {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        # Work
        if user_facts["works"]:
            print("\n[WORK] WORK INFORMATION:")
            for fact in user_facts["works"]:
                relation = fact['relation'].replace('_', ' ').title()
                print(f"  • {relation}: {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        # Other relationships
        if user_facts["other"]:
            print("\n[OTHER] OTHER RELATIONSHIPS:")
            for fact in user_facts["other"]:
                relation = fact['relation'].replace('_', ' ').title()
                print(f"  • {relation}: {fact['target']} (Confidence: {fact['confidence']:.2f})")
        
        if not any(user_facts.values()):
            print("  No user-specific facts found in knowledge graph.")
    
    except Exception as e:
        print(f"  Error loading knowledge graph: {e}")
    
    # 3. Episodic Memories
    print("\n" + "="*70)
    print("EPISODIC MEMORIES (Conversation History)")
    print("="*70)
    
    try:
        dates = get_all_conversation_dates()
        if dates:
            print(f"\n  Total conversation days: {len(dates)}")
            print(f"  Date range: {min(dates)} to {max(dates)}")
            
            # Show recent conversations
            recent_dates = sorted(dates, reverse=True)[:5]
            print(f"\n  Recent conversations:")
            for date_str in recent_dates:
                log_data = get_log_for_date(date_str)
                if log_data and isinstance(log_data, list):
                    event_count = len(log_data)
                    print(f"    • {date_str}: {event_count} interaction(s)")
                elif log_data and isinstance(log_data, dict):
                    events = log_data.get("events", [])
                    event_count = len(events)
                    print(f"    • {date_str}: {event_count} interaction(s)")
        else:
            print("  No episodic memories found.")
    except Exception as e:
        print(f"  Error loading episodic memories: {e}")
    
    # 4. Vector Memories (Semantic Search)
    print("\n" + "="*70)
    print("VECTOR MEMORIES (Semantic Memories)")
    print("="*70)
    
    try:
        # Search for user-related memories
        user_memories = search_memory("you", limit=10)
        if user_memories:
            print(f"\n  Found {len(user_memories)} relevant memories:")
            for i, memory in enumerate(user_memories, 1):
                content = memory.get("content", "N/A")
                similarity = memory.get("similarity", 0.0)
                timestamp = memory.get("timestamp", "Unknown")
                print(f"    {i}. {content[:100]}...")
                print(f"       (Similarity: {similarity:.2f}, Date: {timestamp})")
        else:
            print("  No vector memories found.")
    except Exception as e:
        print(f"  Error loading vector memories: {e}")
    
    # 5. Summary Statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    try:
        graph = get_knowledge_graph()
        user_node = graph.get_node_by_name("i") or graph.get_node_by_name("user")
        
        if user_node:
            edges = graph.get_edges_from_node(user_node.id)
            print(f"\n  Total relationships about you: {len(edges)}")
            
            # Count by type
            relation_counts = {}
            for edge in edges:
                rel_type = edge.type
                relation_counts[rel_type] = relation_counts.get(rel_type, 0) + 1
            
            print(f"\n  Relationships by type:")
            for rel_type, count in sorted(relation_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"    • {rel_type.replace('_', ' ').title()}: {count}")
        else:
            print("  No user node found in knowledge graph.")
    
    except Exception as e:
        print(f"  Error calculating statistics: {e}")
    
    print("\n" + "="*70)
    print("END OF REPORT")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        show_all_user_data()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

