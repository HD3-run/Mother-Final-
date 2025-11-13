"""
Conflict Resolution System for MOTHER

This module detects and resolves conflicts in the knowledge graph,
handling contradictory facts and generating clarification questions
when needed.
"""

import logging
from typing import List, Optional, Dict, Any

from memory.knowledge_graph import KnowledgeGraph, ConceptNode, RelationshipEdge

logger = logging.getLogger(__name__)


def detect_conflicts(
    graph: KnowledgeGraph,
    subject_name: str,
    relation_type: str,
    new_target_name: str,
) -> List[RelationshipEdge]:
    """
    Detects conflicts when adding a new fact to the graph.
    
    A conflict occurs when:
    - An exclusive relationship already exists (e.g., user can only have one name)
    - A contradictory fact exists (e.g., "Paris is in France" vs "Paris is in Germany")
    
    Args:
        graph: The KnowledgeGraph instance to check.
        subject_name: The name of the subject node.
        relation_type: The type of relationship being added.
        new_target_name: The name of the target node being added.
        
    Returns:
        A list of conflicting RelationshipEdge objects.
    """
    logger.info(f"[Conflict Resolver]: Checking for conflicts: '{subject_name}' --[{relation_type}]--> '{new_target_name}'")
    
    subject_node = graph.get_node_by_name(subject_name)
    if not subject_node:
        return []
    
    conflicts: List[RelationshipEdge] = []
    
    # Check for existing edges with the same relation type
    existing_edges = graph.get_edges_from_node(subject_node.id)
    for edge in existing_edges:
        if edge.type == relation_type:
            target_node = graph.get_node_by_id(edge.target)
            if target_node and target_node.name.lower() != new_target_name.lower():
                # Check if this is an exclusive relationship type
                if _is_exclusive_relation(relation_type):
                    conflicts.append(edge)
                # Check for direct contradiction
                elif _is_contradictory(relation_type, target_node.name, new_target_name):
                    conflicts.append(edge)
    
    if conflicts:
        logger.warning(f"    - Found {len(conflicts)} conflict(s)")
    else:
        logger.info("    - No conflicts detected")
    
    return conflicts


def _is_exclusive_relation(relation_type: str) -> bool:
    """
    Determines if a relation type is exclusive (only one can exist per subject).
    
    Examples: "has_name", "is_located_in" (for cities), "has_age"
    """
    exclusive_types = {
        "has_name",
        "has_age",
        "is_located_in",  # For cities/countries
        "is_from",
        "works_at",  # Assuming one primary job
    }
    return relation_type in exclusive_types


def _is_contradictory(
    relation_type: str,
    existing_target: str,
    new_target: str,
) -> bool:
    """
    Checks if two facts are contradictory.
    
    This is a simple heuristic - can be enhanced with LLM-based reasoning.
    """
    # For "is_a" relationships, different targets are usually not contradictory
    # (e.g., "dog is_a mammal" and "dog is_a pet" are both true)
    if relation_type == "is_a":
        return False
    
    # For location-based facts, different targets are contradictory
    location_relations = {"is_located_in", "is_from", "lives_in"}
    if relation_type in location_relations:
        return existing_target.lower() != new_target.lower()
    
    # Default: different targets are potentially contradictory
    return existing_target.lower() != new_target.lower()


def resolve_conflict(
    graph: KnowledgeGraph,
    existing_edge: RelationshipEdge,
    new_confidence: float,
    new_provenance: str = "user",
) -> str:
    """
    Resolves a conflict by updating or superseding the existing edge.
    
    Args:
        graph: The KnowledgeGraph instance.
        existing_edge: The existing conflicting edge.
        new_confidence: The confidence of the new fact.
        new_provenance: The provenance of the new fact.
        
    Returns:
        Status string: "replaced", "merged", "ignored", or "needs_clarification"
    """
    logger.info(f"[Conflict Resolver]: Resolving conflict for edge {existing_edge.id}")
    
    # Use the graph's built-in conflict resolution
    status = graph.revise_conflicting_edge(
        existing_edge,
        new_confidence,
        new_provenance,
    )
    
    if status == "replaced":
        logger.info("    - Existing fact replaced with higher-provenance fact")
    elif status == "merged":
        logger.info("    - Facts merged (same provenance)")
    elif status == "ignored":
        logger.info("    - New fact ignored (lower provenance)")
    
    return status


def generate_clarification_question(
    graph: KnowledgeGraph,
    subject_name: str,
    relation_type: str,
    conflicting_targets: List[str],
) -> str:
    """
    Generates a clarification question when conflicts are detected.
    
    Args:
        graph: The KnowledgeGraph instance.
        subject_name: The subject of the conflicting facts.
        relation_type: The type of relationship in conflict.
        conflicting_targets: List of conflicting target values.
        
    Returns:
        A natural language clarification question.
    """
    relation_display = relation_type.replace("_", " ").title()
    
    if len(conflicting_targets) == 1:
        return (
            f"I remember that {subject_name}'s {relation_display} is {conflicting_targets[0]}. "
            f"Is this still correct, or has it changed?"
        )
    else:
        targets_str = ", ".join(conflicting_targets[:-1]) + f", or {conflicting_targets[-1]}"
        return (
            f"I have conflicting information about {subject_name}'s {relation_display}. "
            f"Is it {targets_str}? Could you clarify which is correct?"
        )


def handle_conflict_with_user(
    graph: KnowledgeGraph,
    subject_name: str,
    relation_type: str,
    new_target_name: str,
    new_confidence: float = 0.8,
    new_provenance: str = "user",
) -> Dict[str, Any]:
    """
    Main entry point for conflict handling. Detects conflicts and either
    resolves them automatically or returns a clarification question.
    
    Args:
        graph: The KnowledgeGraph instance.
        subject_name: The subject of the new fact.
        relation_type: The type of relationship.
        new_target_name: The target of the new fact.
        new_confidence: Confidence of the new fact.
        new_provenance: Provenance of the new fact.
        
    Returns:
        A dictionary with:
        - "has_conflict": bool
        - "resolved": bool
        - "status": str
        - "clarification_question": str (if needed)
        - "conflicting_edges": List[RelationshipEdge]
    """
    conflicts = detect_conflicts(graph, subject_name, relation_type, new_target_name)
    
    if not conflicts:
        return {
            "has_conflict": False,
            "resolved": True,
            "status": "no_conflict",
            "clarification_question": None,
            "conflicting_edges": [],
        }
    
    # Try to resolve automatically
    resolved_statuses = []
    for conflict_edge in conflicts:
        status = resolve_conflict(graph, conflict_edge, new_confidence, new_provenance)
        resolved_statuses.append(status)
    
    # If all conflicts were resolved automatically, we're done
    if all(s in ("replaced", "merged", "ignored") for s in resolved_statuses):
        return {
            "has_conflict": True,
            "resolved": True,
            "status": resolved_statuses[0],
            "clarification_question": None,
            "conflicting_edges": conflicts,
        }
    
    # Otherwise, we need user clarification
    conflicting_targets = []
    for edge in conflicts:
        target_node = graph.get_node_by_id(edge.target)
        if target_node:
            conflicting_targets.append(target_node.name)
    conflicting_targets.append(new_target_name)
    
    clarification = generate_clarification_question(
        graph,
        subject_name,
        relation_type,
        conflicting_targets,
    )
    
    return {
        "has_conflict": True,
        "resolved": False,
        "status": "needs_clarification",
        "clarification_question": clarification,
        "conflicting_edges": conflicts,
    }

