#!/usr/bin/env python3
"""Check if pet nodes exist in the knowledge graph"""

from memory.structured_store import get_knowledge_graph

graph = get_knowledge_graph()

# Find "i" or "user" node
user_node = graph.get_node_by_name("i") or graph.get_node_by_name("user")
if not user_node:
    print("No user node found")
    exit(1)

print(f"User node: {user_node.name} (ID: {user_node.id})\n")

# Get all edges from user node
edges = graph.get_edges_from_node(user_node.id)

# Find has_pet edges
pet_edges = [e for e in edges if e.type.lower() in ["has_pet", "has"]]

print(f"Found {len(pet_edges)} pet-related edges:\n")

for edge in pet_edges:
    target_node = graph.get_node_by_id(edge.target)
    if target_node:
        pet_type = edge.properties.get("type", "unknown")
        print(f"  • {user_node.name} --[{edge.type}]--> {target_node.name}")
        print(f"    Type: {pet_type}")
        print(f"    Confidence: {edge.weight:.2f}")
        print(f"    Properties: {edge.properties}")
        print()

# Also check for "nia" node directly
nia_node = graph.get_node_by_name("nia")
if nia_node:
    print(f"\nFound 'nia' node directly:")
    print(f"  Name: {nia_node.name}")
    print(f"  Type: {nia_node.type}")
    print(f"  Properties: {nia_node.properties}")
    
    # Check incoming edges
    incoming = graph.get_edges_to_node(nia_node.id)
    print(f"\n  Incoming edges: {len(incoming)}")
    for edge in incoming:
        source_node = graph.get_node_by_id(edge.source)
        if source_node:
            print(f"    {source_node.name} --[{edge.type}]--> {nia_node.name}")
else:
    print("\n'nia' node NOT found in knowledge graph")

