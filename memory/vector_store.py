import json
import logging
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

VECTOR_STORE_FILE = 'data/vector_store.json'
VECTORIZER_FILE = 'data/vectorizer.pkl'
EMBEDDINGS_FILE = 'data/embeddings.npy'

class VectorMemoryStore:
    """Enhanced vector memory store with semantic search"""
    
    def __init__(self):
        self.vectorizer = None
        self.embeddings = None
        self.memories = []
        self.load_store()
    
    def load_store(self):
        """Load the vector store from disk"""
        try:
            # Load memories
            if os.path.exists(VECTOR_STORE_FILE):
                with open(VECTOR_STORE_FILE, 'r') as f:
                    self.memories = json.load(f)
            
            # Load vectorizer
            if os.path.exists(VECTORIZER_FILE):
                with open(VECTORIZER_FILE, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            else:
                self.vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
            
            # Load embeddings
            if os.path.exists(EMBEDDINGS_FILE) and len(self.memories) > 0:
                self.embeddings = np.load(EMBEDDINGS_FILE)
            
            logging.info(f"[🧠] Vector store loaded with {len(self.memories)} memories")
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to load vector store: {e}")
            self.memories = []
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.embeddings = None
    
    def save_store(self):
        """Save the vector store to disk"""
        try:
            # Ensure data directory exists
            os.makedirs('data', exist_ok=True)
            
            # Save memories
            with open(VECTOR_STORE_FILE, 'w') as f:
                json.dump(self.memories, f, indent=2)
            
            # Save vectorizer
            with open(VECTORIZER_FILE, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(EMBEDDINGS_FILE, self.embeddings)
            
            logging.info(f"[💾] Vector store saved with {len(self.memories)} memories")
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to save vector store: {e}")
    
    def add_memory(self, user_input: str, ai_response: str, 
                  emotional_context: Dict = None, predicted_state: Dict = None):
        """Add a new memory to the vector store"""
        try:
            memory = {
                'id': len(self.memories),
                'user_input': user_input,
                'ai_response': ai_response,
                'timestamp': datetime.now().isoformat(),
                'emotional_context': emotional_context or {},
                'predicted_state': predicted_state or {},
                'content': f"{user_input} {ai_response}",  # Combined content for vectorization
                'metadata': {
                    'length': len(user_input) + len(ai_response),
                    'interaction_type': self._classify_interaction(user_input, ai_response)
                }
            }
            
            self.memories.append(memory)
            self._rebuild_embeddings()
            self.save_store()
            
            logging.info(f"[🧠] Added memory {memory['id']}")
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to add memory: {e}")
    
    def search_memory(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity"""
        try:
            if not self.memories or self.embeddings is None:
                return []
            
            # Vectorize query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.embeddings)[0]
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:limit]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    memory = self.memories[idx].copy()
                    memory['similarity'] = float(similarities[idx])
                    results.append(memory)
            
            logging.info(f"[🔍] Found {len(results)} memories for query: {query}")
            return results
            
        except Exception as e:
            logging.error(f"[ERROR] Memory search failed: {e}")
            return []
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent memories"""
        try:
            return self.memories[-limit:] if self.memories else []
        except Exception as e:
            logging.error(f"[ERROR] Failed to get recent memories: {e}")
            return []
    
    def get_memories_by_emotion(self, emotion: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get memories filtered by emotional context"""
        try:
            filtered_memories = []
            
            for memory in self.memories:
                emotional_context = memory.get('emotional_context', {})
                if emotional_context.get('dominant_emotion') == emotion:
                    filtered_memories.append(memory)
            
            return filtered_memories[-limit:] if filtered_memories else []
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to get memories by emotion: {e}")
            return []
    
    def _rebuild_embeddings(self):
        """Rebuild embeddings for all memories"""
        try:
            if not self.memories:
                return
            
            # Extract content for vectorization
            contents = [memory['content'] for memory in self.memories]
            
            # Fit and transform
            if len(contents) == 1:
                # First memory - fit the vectorizer
                self.embeddings = self.vectorizer.fit_transform(contents)
            else:
                # Update existing embeddings
                self.embeddings = self.vectorizer.fit_transform(contents)
            
            self.embeddings = self.embeddings.toarray()
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to rebuild embeddings: {e}")
    
    def _classify_interaction(self, user_input: str, ai_response: str) -> str:
        """Classify the type of interaction"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['?', 'what', 'how', 'why', 'when', 'where']):
            return 'question'
        elif any(word in user_lower for word in ['help', 'support', 'advice']):
            return 'support'
        elif any(word in user_lower for word in ['feel', 'emotion', 'sad', 'happy', 'angry']):
            return 'emotional'
        elif any(word in user_lower for word in ['tell', 'story', 'share']):
            return 'sharing'
        else:
            return 'general'
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        try:
            if not self.memories:
                return {'total_memories': 0}
            
            # Analyze interaction types
            interaction_types = {}
            emotions = {}
            
            for memory in self.memories:
                # Count interaction types
                interaction_type = memory.get('metadata', {}).get('interaction_type', 'unknown')
                interaction_types[interaction_type] = interaction_types.get(interaction_type, 0) + 1
                
                # Count emotions
                emotion = memory.get('emotional_context', {}).get('dominant_emotion', 'neutral')
                emotions[emotion] = emotions.get(emotion, 0) + 1
            
            return {
                'total_memories': len(self.memories),
                'interaction_types': interaction_types,
                'emotion_distribution': emotions,
                'average_length': np.mean([m.get('metadata', {}).get('length', 0) for m in self.memories]),
                'date_range': {
                    'earliest': self.memories[0]['timestamp'] if self.memories else None,
                    'latest': self.memories[-1]['timestamp'] if self.memories else None
                }
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to get memory statistics: {e}")
            return {'total_memories': 0}

# Global vector store instance
vector_store = VectorMemoryStore()

def add_memory(user_input: str, ai_response: str, 
              emotional_context: Dict = None, predicted_state: Dict = None):
    """Add memory to vector store"""
    vector_store.add_memory(user_input, ai_response, emotional_context, predicted_state)

def search_memory(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search memories"""
    return vector_store.search_memory(query, limit)

def get_recent_memories(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent memories"""
    return vector_store.get_recent_memories(limit)

def get_memory_statistics() -> Dict[str, Any]:
    """Get memory statistics"""
    return vector_store.get_memory_statistics()
