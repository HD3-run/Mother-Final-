#!/usr/bin/env python3
"""
Simple script to view and explore the knowledge graph.

Usage:
    python view_knowledge_graph.py
    python view_knowledge_graph.py --stats
    python view_knowledge_graph.py --node "user"
    python view_knowledge_graph.py --visualize
"""

import json
import argparse
from pathlib import Path
from memory.knowledge_graph import KnowledgeGraph

def print_stats(graph: KnowledgeGraph):
    """Print statistics about the knowledge graph"""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*60)
    print(f"Total Nodes: {len(graph.graph.nodes)}")
    print(f"Total Edges: {len(graph.graph.edges)}")
    print(f"Total Concepts: {len(graph.name_to_id)}")
    
    # Count nodes by type
    node_types = {}
    for node_id in graph.graph.nodes:
        node_data = graph.graph.nodes[node_id]
        node_type = node_data.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1
    
    print("\nNodes by Type:")
    for node_type, count in sorted(node_types.items()):
        print(f"  {node_type}: {count}")
    
    # Count edges by relation type
    edge_types = {}
    for u, v, data in graph.graph.edges(data=True):
        edge_type = data.get('type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    
    print("\nEdges by Relation Type:")
    for edge_type, count in sorted(edge_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {edge_type}: {count}")
    
    print("="*60 + "\n")

def print_node_info(graph: KnowledgeGraph, node_name: str):
    """Print detailed information about a specific node"""
    node = graph.get_node_by_name(node_name)
    if not node:
        print(f"Node '{node_name}' not found in graph.")
        return
    
    print(f"\n{'='*60}")
    print(f"NODE: {node.name}")
    print(f"{'='*60}")
    print(f"ID: {node.id}")
    print(f"Type: {node.type}")
    print(f"Value: {node.value}")
    print(f"Activation: {node.activation}")
    if node.properties:
        print(f"Properties: {json.dumps(node.properties, indent=2)}")
    
    # Get outgoing edges
    outgoing = graph.get_edges_from_node(node.id)
    if outgoing:
        print(f"\nOutgoing Relationships ({len(outgoing)}):")
        for edge in outgoing:
            target_node = graph.get_node_by_id(edge.target)
            if target_node:
                print(f"  --[{edge.type}]--> {target_node.name} (confidence: {edge.weight:.2f})")
    
    # Get incoming edges
    incoming = graph.get_edges_to_node(node.id)
    if incoming:
        print(f"\nIncoming Relationships ({len(incoming)}):")
        for edge in incoming:
            source_node = graph.get_node_by_id(edge.source)
            if source_node:
                print(f"  {source_node.name} --[{edge.type}]--> (confidence: {edge.weight:.2f})")
    
    print(f"{'='*60}\n")

def list_all_nodes(graph: KnowledgeGraph):
    """List all nodes in the graph"""
    print("\n" + "="*60)
    print("ALL NODES IN GRAPH")
    print("="*60)
    
    nodes = []
    for node_id in graph.graph.nodes:
        node_data = graph.graph.nodes[node_id]
        nodes.append((node_data.get('name', 'unknown'), node_data.get('type', 'unknown')))
    
    nodes.sort()
    for name, node_type in nodes:
        print(f"  {name} ({node_type})")
    
    print(f"\nTotal: {len(nodes)} nodes")
    print("="*60 + "\n")

def visualize_graph(graph: KnowledgeGraph):
    """Create a simple text visualization of the graph"""
    print("\n" + "="*60)
    print("KNOWLEDGE GRAPH VISUALIZATION")
    print("="*60)
    
    # Get all edges
    edges = graph.get_all_edges()
    
    if not edges:
        print("Graph is empty - no relationships stored yet.")
        print("="*60 + "\n")
        return
    
    # Group by source node
    by_source = {}
    for edge in edges:
        source_node = graph.get_node_by_id(edge.source)
        target_node = graph.get_node_by_id(edge.target)
        if source_node and target_node:
            if source_node.name not in by_source:
                by_source[source_node.name] = []
            by_source[source_node.name].append((edge.type, target_node.name, edge.weight))
    
    # Print relationships
    for source_name in sorted(by_source.keys()):
        print(f"\n{source_name}:")
        for rel_type, target_name, weight in by_source[source_name]:
            print(f"  --[{rel_type}]--> {target_name} (confidence: {weight:.2f})")
    
    print("\n" + "="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="View and explore the knowledge graph")
    parser.add_argument('--stats', action='store_true', help='Show graph statistics')
    parser.add_argument('--node', type=str, help='Show details for a specific node')
    parser.add_argument('--list', action='store_true', help='List all nodes')
    parser.add_argument('--visualize', action='store_true', help='Visualize graph relationships')
    parser.add_argument('--file', type=str, default='data/structured_memory/knowledge_graph.json',
                       help='Path to knowledge graph file')
    
    args = parser.parse_args()
    
    # Load graph
    graph_file = Path(args.file)
    if not graph_file.exists():
        print(f"Knowledge graph file not found at: {graph_file}")
        print("The graph will be created automatically when you first use the system.")
        return
    
    try:
        graph = KnowledgeGraph.load_from_file(str(graph_file))
    except Exception as e:
        print(f"Error loading knowledge graph: {e}")
        return
    
    # Execute requested action
    if args.node:
        print_node_info(graph, args.node)
    elif args.list:
        list_all_nodes(graph)
    elif args.visualize:
        visualize_graph(graph)
    else:
        # Default: show stats
        print_stats(graph)
        print("\nUse --help to see all available options:")
        print("  --stats      : Show statistics")
        print("  --node NAME  : Show details for a specific node")
        print("  --list       : List all nodes")
        print("  --visualize  : Visualize relationships")

if __name__ == "__main__":
    main()

