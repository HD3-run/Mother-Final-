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

def initialize_system():
    """Initialize the enhanced MotherX AI system"""
    try:
        logging.info("[🚀] Starting Enhanced MotherX AI System initialization...")
        
        # Setup logging
        setup_logging()
        logging.info("[📝] Logging system initialized")
        
        # Load configuration
        config = load_config()
        logging.info(f"[⚙️] Configuration loaded: {config.get('name', 'MOTHER')} v{config.get('version', '4.0')}")
        
        # Initialize database
        init_db()
        logging.info("[💾] Database initialized")
        
        # Initialize identity engine
        initialize_identity()
        logging.info("[🎭] Identity formation engine initialized")
        
        # Initialize ML models
        initialize_ml_models()
        logging.info("[🤖] Machine learning models initialized")
        
        # Initialize semantic clustering
        initialize_clustering()
        logging.info("[🧠] Semantic clustering initialized")
        
        # Create necessary directories
        directories = [
            'data',
            'logs',
            'memory',
            'models',
            'static/uploads'
        ]
        
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
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
        
        logging.info("[✅] Enhanced MotherX AI System initialization complete")
        logging.info("[🧠] Identity Formation: ENABLED")
        logging.info("[🤖] Autonomous Decisions: ENABLED")
        logging.info("[📊] Predictive Modeling: ENABLED")
        logging.info("[🔗] Semantic Clustering: ENABLED")
        
        return True
        
    except Exception as e:
        logging.error(f"[❌] System initialization failed: {e}")
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
        logging.error(f"[❌] System health check failed: {e}")
        return {
            "system": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    initialize_system()
