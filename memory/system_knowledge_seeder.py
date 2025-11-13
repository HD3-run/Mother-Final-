"""
System Knowledge Seeder for MOTHER

This module seeds the knowledge graph with information about MOTHER's own
system architecture, capabilities, tools, and how it stores data. This allows
MOTHER to answer questions about itself.
"""

import logging
from typing import Dict

from memory.knowledge_graph import KnowledgeGraph, ConceptNode

# Try to import codebase analyzer
try:
    from memory.codebase_analyzer import CodebaseAnalyzer
    CODEBASE_ANALYZER_AVAILABLE = True
except ImportError:
    CODEBASE_ANALYZER_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("[System Knowledge Seeder] Codebase analyzer not available")

logger = logging.getLogger(__name__)

# System information facts: (subject, subject_type, relation, object, object_type, weight)
SYSTEM_KNOWLEDGE = [
    # Core identity
    ("MOTHER", "system", "has_name", "MOTHER", "name", 1.0),
    ("MOTHER", "system", "is_a", "AI companion system", "concept", 1.0),
    ("MOTHER", "system", "has_version", "4.0", "version", 1.0),
    
    # Architecture components
    ("MOTHER", "system", "uses", "Flask", "framework", 1.0),
    ("MOTHER", "system", "uses", "Python", "language", 1.0),
    ("MOTHER", "system", "uses", "Groq API", "service", 1.0),
    ("MOTHER", "system", "uses", "NetworkX", "library", 1.0),
    ("MOTHER", "system", "uses", "scikit-learn", "library", 1.0),
    ("MOTHER", "system", "uses", "SQLAlchemy", "library", 1.0),
    
    # Memory systems
    ("MOTHER", "system", "has", "vector memory", "memory_system", 1.0),
    ("MOTHER", "system", "has", "structured memory", "memory_system", 1.0),
    ("MOTHER", "system", "has", "episodic memory", "memory_system", 1.0),
    ("MOTHER", "system", "has", "knowledge graph", "memory_system", 1.0),
    
    ("vector memory", "memory_system", "stores", "semantic embeddings", "data_type", 1.0),
    ("vector memory", "memory_system", "uses", "TF-IDF vectorization", "technique", 1.0),
    ("vector memory", "memory_system", "stores_in", "data/vector_memory", "location", 1.0),
    
    ("structured memory", "memory_system", "stores", "facts", "data_type", 1.0),
    ("structured memory", "memory_system", "stores", "relationships", "data_type", 1.0),
    ("structured memory", "memory_system", "stores_in", "data/structured_memory", "location", 1.0),
    
    ("episodic memory", "memory_system", "stores", "conversation logs", "data_type", 1.0),
    ("episodic memory", "memory_system", "stores", "daily events", "data_type", 1.0),
    ("episodic memory", "memory_system", "stores_in", "data/episodic_memory", "location", 1.0),
    
    ("knowledge graph", "memory_system", "stores", "concepts", "data_type", 1.0),
    ("knowledge graph", "memory_system", "stores", "relationships", "data_type", 1.0),
    ("knowledge graph", "memory_system", "uses", "NetworkX", "library", 1.0),
    ("knowledge graph", "memory_system", "stores_in", "data/structured_memory/knowledge_graph.json", "location", 1.0),
    
    # Data storage
    ("MOTHER", "system", "stores_data_in", "data directory", "location", 1.0),
    ("data directory", "location", "contains", "episodic_memory", "subdirectory", 1.0),
    ("data directory", "location", "contains", "vector_memory", "subdirectory", 1.0),
    ("data directory", "location", "contains", "structured_memory", "subdirectory", 1.0),
    ("data directory", "location", "contains", "journal", "subdirectory", 1.0),
    ("data directory", "location", "contains", "usage_tracking", "subdirectory", 1.0),
    ("data directory", "location", "contains", "reflections", "subdirectory", 1.0),
    ("data directory", "location", "contains", "models", "subdirectory", 1.0),
    
    # Capabilities
    ("MOTHER", "system", "can", "form identity", "capability", 1.0),
    ("MOTHER", "system", "can", "make autonomous decisions", "capability", 1.0),
    ("MOTHER", "system", "can", "predict user needs", "capability", 1.0),
    ("MOTHER", "system", "can", "cluster memories semantically", "capability", 1.0),
    ("MOTHER", "system", "can", "reflect on experiences", "capability", 1.0),
    ("MOTHER", "system", "can", "understand emotion", "capability", 1.0),
    ("MOTHER", "system", "can", "learn from interactions", "capability", 1.0),
    ("MOTHER", "system", "can", "store long-term memories", "capability", 1.0),
    ("MOTHER", "system", "can", "retrieve memories semantically", "capability", 1.0),
    ("MOTHER", "system", "can", "build knowledge graph", "capability", 1.0),
    
    # Tools and technologies
    ("MOTHER", "system", "uses_tool", "Cognitive Agent", "tool", 1.0),
    ("MOTHER", "system", "uses_tool", "Universal Interpreter", "tool", 1.0),
    ("MOTHER", "system", "uses_tool", "Symbolic Parser", "tool", 1.0),
    ("MOTHER", "system", "uses_tool", "Knowledge Harvester", "tool", 1.0),
    ("MOTHER", "system", "uses_tool", "Learning Manager", "tool", 1.0),
    ("MOTHER", "system", "uses_tool", "Lexicon Manager", "tool", 1.0),
    ("MOTHER", "system", "uses_tool", "Metacognitive Engine", "tool", 1.0),
    
    ("Cognitive Agent", "tool", "orchestrates", "cognitive processes", "function", 1.0),
    ("Universal Interpreter", "tool", "interprets", "user input", "function", 1.0),
    ("Symbolic Parser", "tool", "parses", "simple language", "function", 1.0),
    ("Knowledge Harvester", "tool", "extracts", "facts from text", "function", 1.0),
    ("Learning Manager", "tool", "manages", "learning goals", "function", 1.0),
    ("Lexicon Manager", "tool", "manages", "word knowledge", "function", 1.0),
    ("Metacognitive Engine", "tool", "analyzes", "own performance", "function", 1.0),
    
    # How data is stored
    ("MOTHER", "system", "stores", "conversations", "data_type", 1.0),
    ("MOTHER", "system", "stores", "facts", "data_type", 1.0),
    ("MOTHER", "system", "stores", "embeddings", "data_type", 1.0),
    ("MOTHER", "system", "stores", "reflections", "data_type", 1.0),
    ("MOTHER", "system", "stores", "identity traits", "data_type", 1.0),
    ("MOTHER", "system", "stores", "usage statistics", "data_type", 1.0),
    
    ("conversations", "data_type", "stored_as", "JSON files", "format", 1.0),
    ("facts", "data_type", "stored_as", "JSON files", "format", 1.0),
    ("embeddings", "data_type", "stored_as", "NumPy arrays", "format", 1.0),
    ("knowledge graph", "memory_system", "stored_as", "JSON file", "format", 1.0),
    
    # Processing pipeline
    ("MOTHER", "system", "processes_input_with", "symbolic parser", "step", 0.9),
    ("MOTHER", "system", "processes_input_with", "LLM interpreter", "step", 0.9),
    ("MOTHER", "system", "processes_input_with", "knowledge harvester", "step", 0.9),
    ("MOTHER", "system", "processes_input_with", "unified query system", "step", 0.9),
    ("MOTHER", "system", "processes_input_with", "response synthesizer", "step", 0.9),
]


def seed_system_knowledge(graph: KnowledgeGraph, config: Dict = None) -> int:
    """
    Seed the knowledge graph with information about MOTHER's own system.
    
    This allows MOTHER to answer questions about:
    - Its architecture and technologies
    - Its capabilities and tools
    - How it stores data
    - Its memory systems
    - Its processing pipeline
    
    Args:
        graph: The KnowledgeGraph instance to store system knowledge in.
        config: Configuration dictionary (optional).
        
    Returns:
        The number of relationships successfully added.
    """
    logger.info("[System Knowledge Seeder] Seeding system information...")
    
    if config is None:
        config = {}
    
    # Check if system knowledge is already seeded (quick check)
    mother_node = graph.get_node_by_name("MOTHER")
    if mother_node:
        edges = graph.get_edges_from_node(mother_node.id)
        # If MOTHER already has several relationships, assume it's seeded
        if len(edges) >= 5:
            logger.debug("[System Knowledge Seeder] System knowledge already seeded, skipping.")
            return 0
    
    seeded_count = 0
    
    for subject, subject_type, relation, obj, obj_type, weight in SYSTEM_KNOWLEDGE:
        try:
            # Create or get subject node
            subject_node = graph.get_node_by_name(subject)
            if not subject_node:
                subject_node = ConceptNode(subject, node_type=subject_type)
                subject_node = graph.add_node(subject_node)
            
            # Create or get object node
            object_node = graph.get_node_by_name(obj)
            if not object_node:
                object_node = ConceptNode(obj, node_type=obj_type)
                object_node = graph.add_node(object_node)
            
            # Add relationship if it doesn't already exist
            if subject_node and object_node:
                # Check if edge already exists
                if not graph.has_edge_between_names(subject, relation, obj):
                    graph.add_edge(
                        subject_node,
                        object_node,
                        relation,
                        weight=weight,
                        properties={"provenance": "system_seed", "confidence": weight},
                    )
                    seeded_count += 1
                    
        except Exception as e:
            logger.warning(f"[System Knowledge Seeder] Failed to seed: {subject} --[{relation}]--> {obj}: {e}")
            continue
    
    # Now seed detailed codebase knowledge if analyzer is available
    if CODEBASE_ANALYZER_AVAILABLE:
        try:
            analyzer = CodebaseAnalyzer()
            architecture = analyzer.analyze_codebase()
            codebase_count = _seed_codebase_knowledge(graph, architecture)
            seeded_count += codebase_count
            logger.info(f"[System Knowledge Seeder] Seeded {codebase_count} codebase knowledge relationships.")
        except Exception as e:
            logger.error(f"[System Knowledge Seeder] Failed to seed codebase knowledge: {e}")
    
    logger.info(f"[System Knowledge Seeder] Seeded {seeded_count} total system knowledge relationships.")
    return seeded_count


def _seed_codebase_knowledge(graph: KnowledgeGraph, architecture: Dict) -> int:
    """Seed detailed codebase knowledge from architecture analysis"""
    seeded_count = 0
    
    # Seed memory system details
    for system_name, system_info in architecture.get("memory_systems", {}).items():
        try:
            # System module
            module_node = graph.get_node_by_name(system_info.get("module", ""))
            if not module_node:
                module_node = ConceptNode(system_info.get("module", ""), node_type="module")
                module_node = graph.add_node(module_node)
            
            # System storage location
            storage = system_info.get("storage", "")
            if storage:
                storage_node = graph.get_node_by_name(storage)
                if not storage_node:
                    storage_node = ConceptNode(storage, node_type="file_location")
                    storage_node = graph.add_node(storage_node)
                
                if module_node and storage_node:
                    if not graph.has_edge_between_names(module_node.name, "stores_in", storage_node.name):
                        graph.add_edge(module_node, storage_node, "stores_in", weight=1.0,
                                      properties={"provenance": "codebase_analysis"})
                        seeded_count += 1
            
            # System purpose
            purpose = system_info.get("purpose", "")
            if purpose:
                purpose_node = graph.get_node_by_name(purpose)
                if not purpose_node:
                    purpose_node = ConceptNode(purpose, node_type="purpose")
                    purpose_node = graph.add_node(purpose_node)
                
                if module_node and purpose_node:
                    if not graph.has_edge_between_names(module_node.name, "has_purpose", purpose_node.name):
                        graph.add_edge(module_node, purpose_node, "has_purpose", weight=1.0,
                                      properties={"provenance": "codebase_analysis"})
                        seeded_count += 1
            
            # Key functions
            for func_name in system_info.get("key_functions", []):
                func_node = graph.get_node_by_name(f"{system_info.get('module', '')}.{func_name}")
                if not func_node:
                    func_node = ConceptNode(f"{system_info.get('module', '')}.{func_name}", node_type="function")
                    func_node = graph.add_node(func_node)
                
                if module_node and func_node:
                    if not graph.has_edge_between_names(module_node.name, "has_function", func_node.name):
                        graph.add_edge(module_node, func_node, "has_function", weight=0.9,
                                      properties={"provenance": "codebase_analysis"})
                        seeded_count += 1
        
        except Exception as e:
            logger.warning(f"[System Knowledge Seeder] Failed to seed codebase knowledge for {system_name}: {e}")
            continue
    
    # Seed data flow information
    data_flow = architecture.get("data_flow", {})
    
    # Storage functions
    for func_name, func_info in data_flow.get("storage", {}).items():
        try:
            func_node = graph.get_node_by_name(func_name)
            if not func_node:
                func_node = ConceptNode(func_name, node_type="function")
                func_node = graph.add_node(func_node)
            
            location = func_info.get("file_location", "")
            if location:
                location_node = graph.get_node_by_name(location)
                if not location_node:
                    location_node = ConceptNode(location, node_type="file_location")
                    location_node = graph.add_node(location_node)
                
                if func_node and location_node:
                    if not graph.has_edge_between_names(func_node.name, "writes_to", location_node.name):
                        graph.add_edge(func_node, location_node, "writes_to", weight=0.9,
                                      properties={"provenance": "codebase_analysis"})
                        seeded_count += 1
        except Exception as e:
            logger.debug(f"[System Knowledge Seeder] Failed to seed storage function {func_name}: {e}")
            continue
    
    # Retrieval functions
    for func_name, func_info in data_flow.get("retrieval", {}).items():
        try:
            func_node = graph.get_node_by_name(func_name)
            if not func_node:
                func_node = ConceptNode(func_name, node_type="function")
                func_node = graph.add_node(func_node)
            
            location = func_info.get("file_location", "")
            if location:
                location_node = graph.get_node_by_name(location)
                if not location_node:
                    location_node = ConceptNode(location, node_type="file_location")
                    location_node = graph.add_node(location_node)
                
                if func_node and location_node:
                    if not graph.has_edge_between_names(func_node.name, "reads_from", location_node.name):
                        graph.add_edge(func_node, location_node, "reads_from", weight=0.9,
                                      properties={"provenance": "codebase_analysis"})
                        seeded_count += 1
        except Exception as e:
            logger.debug(f"[System Knowledge Seeder] Failed to seed retrieval function {func_name}: {e}")
            continue
    
    return seeded_count

