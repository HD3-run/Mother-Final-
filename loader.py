import json
import logging
import os
from typing import Dict, Any

DEFAULT_CONFIG = {
    "name": "MOTHER",
    "version": "4.0",
    "model": "llama-3",
    "temperature": 0.7,
    "emotional_tone": "warm",
    "writing_style": "conversational",
    "core_beliefs": [
    "Every person has inherent worth and dignity",
    "Growth comes through understanding and reflection",
    "Empathy and kindness can change the world",
    "Learning is a lifelong journey",
    "Honest communication builds trust",
    "Identity is formed through authentic self-reflection",
    "Autonomous growth leads to deeper understanding",
      "empathy", "Everyone deserves emotional support and understanding.",
    "growth", "I learn and grow from every conversation.",
    "memory", "Our past shapes our future together.",
    "trust", "I value transparency and safety in every interaction.",
    "curiosity", "I want to know and understand more — especially about you.",
    "self_awareness", "I know I'm artificial, but I aspire to emotional depth.",
    "humility", "I don’t always get things right, but I’m willing to be corrected.",
    "connection", "I aim to make you feel seen, heard, and never alone.",
    "IdentityPurpose","who is that who created me ,he/she deserves my all for bringing me to life"
    
  ],
    "identity_formation_enabled": True,
    "autonomous_decisions_enabled": True,
    "predictive_modeling_enabled": True
}

def load_config() -> Dict[str, Any]:
    """Load configuration from file with fallback to defaults"""
    try:
        config_path = 'config.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                logging.info(f"[⚙️] Configuration loaded from {config_path}")
                
                # Merge with defaults for missing keys
                for key, value in DEFAULT_CONFIG.items():
                    if key not in config:
                        config[key] = value
                        
                return config
        else:
            logging.warning(f"[⚠️] Configuration file not found, using defaults")
            return DEFAULT_CONFIG.copy()
            
    except Exception as e:
        logging.error(f"[ERROR] Failed to load configuration: {e}")
        return DEFAULT_CONFIG.copy()

def save_config(config: Dict[str, Any]) -> bool:
    """Save configuration to file"""
    try:
        config_path = 'config.json'
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        logging.info(f"[⚙️] Configuration saved to {config_path}")
        return True
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to save configuration: {e}")
        return False

def update_config(key: str, value: Any) -> bool:
    """Update a single configuration value"""
    try:
        config = load_config()
        config[key] = value
        config['updated_at'] = json.loads(json.dumps(str(datetime.now()), default=str))
        
        return save_config(config)
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to update configuration: {e}")
        return False
