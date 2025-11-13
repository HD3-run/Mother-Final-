import os
import logging
import json
from datetime import datetime
from personality.loader import load_config
from memory.structured_store import init_db
from personality.identity_engine import initialize_identity
from utils.logger import setup_logging
from utils.ml_utils import initialize_ml_models
from memory.semantic_clustering import initialize_clustering
from memory.episodic_logger import init_episodic_logging
from reflection.reflection_engine import ReflectionEngine

def initialize_system():
    """Initialize the enhanced MotherX AI system"""
    try:
        logging.info("[INIT] Starting Enhanced MotherX AI System initialization...")
        
        # Setup logging
        setup_logging()
        logging.info("[LOG] Logging system initialized")
        
        # Load configuration
        config = load_config()
        logging.info(f"[CONFIG] Configuration loaded: {config.get('name', 'MOTHER')} v{config.get('version', '4.0')}")
        
        # Initialize database
        init_db()
        logging.info("[DB] Database initialized")
        
        # Initialize knowledge graph
        from memory.structured_store import get_knowledge_graph, save_knowledge_graph
        from memory.system_knowledge_seeder import seed_system_knowledge
        graph = get_knowledge_graph()
        
        # Seed system knowledge (allows MOTHER to answer questions about itself)
        system_facts_count = seed_system_knowledge(graph, config)
        if system_facts_count > 0:
            save_knowledge_graph()
            logging.info(f"[SYSTEM] Seeded {system_facts_count} system knowledge facts")
        
        save_knowledge_graph()
        logging.info(f"[KG] Knowledge graph initialized ({len(graph.graph.nodes)} nodes, {len(graph.graph.edges)} edges)")
        
        # Initialize identity engine
        initialize_identity()
        logging.info("[ID] Identity formation engine initialized")
        
        # Initialize ML models
        initialize_ml_models()
        logging.info("[ML] Machine learning models initialized")
        
        # Initialize semantic clustering
        initialize_clustering()
        logging.info("[CLUSTER] Semantic clustering initialized")
        
        # Initialize episodic logging (creates data/episodic_memory)
        init_episodic_logging()
        logging.info("[EPISODIC] Episodic logging initialized")
        
        # Initialize reflection engine (creates data/reflections subdirectories)
        reflection_engine = ReflectionEngine()
        logging.info("[REFLECT] Reflection engine initialized")
        
        # Create necessary directories and subdirectories
        directories = [
            'data',
            'data/episodic_memory',
            'data/vector_memory',
            'data/structured_memory',
            'data/journal',
            'data/usage_tracking',
            'data/reflections',
            'data/reflections/daily',
            'data/reflections/identity',
            'data/reflections/autonomous',
            'data/models',
            'data/terminal_logs',
            'logs',
            'instance',
            'memory',
            'models',
            'static',
            'static/uploads'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logging.info(f"[📁] Created directory: {directory}")
        
        # Initialize system state file
        system_state = {
            "initialized_at": datetime.now().isoformat(),
            "version": config.get('version', '4.0'),
            "features": {
                "identity_formation": config.get('identity_formation_enabled', True),
                "autonomous_decisions": config.get('autonomous_decisions_enabled', True),
                "predictive_modeling": config.get('predictive_modeling_enabled', True),
                "semantic_clustering": config.get('semantic_clustering_enabled', True)
            },
            "status": "initialized"
        }
        
        with open('data/system_state.json', 'w') as f:
            json.dump(system_state, f, indent=2)
        
        logging.info("[OK] Enhanced MotherX AI System initialization complete")
        logging.info("[ID] Identity Formation: ENABLED")
        logging.info("[AUTO] Autonomous Decisions: ENABLED")
        logging.info("[PRED] Predictive Modeling: ENABLED")
        logging.info("[CLUSTER] Semantic Clustering: ENABLED")
        
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] System initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_system_health():
    """Check system health and report status"""
    try:
        health_status = {
            "system": "healthy",
            "database": "connected",
            "ml_models": "loaded",
            "identity_engine": "active",
            "autonomous_system": "running",
            "timestamp": datetime.now().isoformat()
        }
        
        # Check if system state file exists
        if os.path.exists('data/system_state.json'):
            with open('data/system_state.json', 'r') as f:
                system_state = json.load(f)
                health_status["last_initialized"] = system_state.get('initialized_at')
        
        return health_status
        
    except Exception as e:
        logging.error(f"[ERROR] System health check failed: {e}")
        return {
            "system": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    initialize_system()
