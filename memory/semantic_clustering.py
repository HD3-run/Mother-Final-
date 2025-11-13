import json
import logging
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from collections import Counter
import pickle

CLUSTERS_FILE = 'data/vector_memory/memory_clusters.json'
CLUSTER_MODEL_FILE = 'data/vector_memory/cluster_model.pkl'

class SemanticClusteringEngine:
    """Engine for semantic clustering of memories"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        self.cluster_model = None
        self.clusters = []
        self.load_clusters()
    
    def initialize_clustering(self):
        """Initialize the clustering system"""
        try:
            os.makedirs('data/vector_memory', exist_ok=True)
            
            if not os.path.exists(CLUSTERS_FILE):
                initial_clusters = {
                    'clusters': [],
                    'last_updated': datetime.now().isoformat(),
                    'total_memories_clustered': 0,
                    'cluster_statistics': {}
                }
                
                with open(CLUSTERS_FILE, 'w') as f:
                    json.dump(initial_clusters, f, indent=2)
            
            logging.info("[🧠] Semantic clustering initialized")
            return True
            
        except Exception as e:
            logging.error(f"[ERROR] Clustering initialization failed: {e}")
            return False
    
    def load_clusters(self):
        """Load existing clusters from storage"""
        try:
            if os.path.exists(CLUSTERS_FILE):
                with open(CLUSTERS_FILE, 'r') as f:
                    cluster_data = json.load(f)
                    self.clusters = cluster_data.get('clusters', [])
            
            if os.path.exists(CLUSTER_MODEL_FILE):
                with open(CLUSTER_MODEL_FILE, 'rb') as f:
                    self.cluster_model = pickle.load(f)
            
            logging.info(f"[🧠] Loaded {len(self.clusters)} memory clusters")
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to load clusters: {e}")
            self.clusters = []
    
    def save_clusters(self):
        """Save clusters to storage"""
        try:
            cluster_data = {
                'clusters': self.clusters,
                'last_updated': datetime.now().isoformat(),
                'total_memories_clustered': sum(len(cluster.get('memories', [])) for cluster in self.clusters),
                'cluster_statistics': self.calculate_cluster_statistics()
            }
            
            with open(CLUSTERS_FILE, 'w') as f:
                json.dump(cluster_data, f, indent=2)
            
            if self.cluster_model:
                with open(CLUSTER_MODEL_FILE, 'wb') as f:
                    pickle.dump(self.cluster_model, f)
            
            logging.info(f"[💾] Saved {len(self.clusters)} memory clusters")
            
        except Exception as e:
            logging.error(f"[ERROR] Failed to save clusters: {e}")
    
    def cluster_memories(self, memories: List[Dict] = None, min_cluster_size: int = 3) -> List[Dict]:
        """Cluster memories using semantic similarity"""
        try:
            if memories is None:
                # Load memories from vector store
                from memory.vector_store import vector_store
                memories = vector_store.memories
            
            if len(memories) < min_cluster_size:
                logging.info("[🧠] Not enough memories for clustering")
                return self.clusters
            
            # Extract content for clustering
            contents = []
            memory_data = []
            
            for memory in memories:
                content = f"{memory.get('user_input', '')} {memory.get('ai_response', '')}"
                contents.append(content)
                memory_data.append({
                    'id': memory.get('id'),
                    'timestamp': memory.get('timestamp'),
                    'emotional_context': memory.get('emotional_context', {}),
                    'content_preview': content[:100] + '...' if len(content) > 100 else content
                })
            
            # Vectorize content
            try:
                vectors = self.vectorizer.fit_transform(contents)
            except ValueError as e:
                logging.warning(f"[WARNING] Vectorization failed: {e}")
                return self.clusters
            
            # Determine optimal number of clusters
            n_clusters = min(max(2, len(memories) // 5), 10)  # 2 to 10 clusters
            
            # Perform clustering
            if len(memories) > 50:
                # Use DBSCAN for large datasets
                clustering = DBSCAN(eps=0.3, min_samples=min_cluster_size, metric='cosine')
                cluster_labels = clustering.fit_predict(vectors.toarray())
            else:
                # Use K-means for smaller datasets
                clustering = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clustering.fit_predict(vectors.toarray())
                self.cluster_model = clustering
            
            # Process clusters
            new_clusters = self.process_clusters(cluster_labels, memory_data, contents)
            
            # Merge with existing clusters or replace
            self.clusters = new_clusters
            self.save_clusters()
            
            logging.info(f"[🧠] Clustered {len(memories)} memories into {len(new_clusters)} clusters")
            return self.clusters
            
        except Exception as e:
            logging.error(f"[ERROR] Memory clustering failed: {e}")
            return self.clusters
    
    def process_clusters(self, cluster_labels: np.ndarray, memory_data: List[Dict], 
                        contents: List[str]) -> List[Dict]:
        """Process clustering results into structured clusters"""
        try:
            clusters = []
            unique_labels = set(int(l) for l in cluster_labels)  # Convert numpy ints to Python ints
            
            for label in unique_labels:
                if label == -1:  # Noise points in DBSCAN
                    continue
                
                # Get memories in this cluster
                cluster_indices = np.where(cluster_labels == label)[0]
                # Convert numpy indices to Python int for JSON serialization
                cluster_memories = [memory_data[int(i)] for i in cluster_indices]
                cluster_contents = [contents[int(i)] for i in cluster_indices]
                
                if len(cluster_memories) < 2:  # Skip small clusters
                    continue
                
                # Generate cluster summary
                cluster_summary = self.generate_cluster_summary(cluster_contents, cluster_memories)
                
                # Analyze emotional patterns
                emotional_patterns = self.analyze_cluster_emotions(cluster_memories)
                
                # Create cluster object
                cluster = {
                    'id': f"cluster_{int(label)}_{datetime.now().strftime('%Y%m%d')}",
                    'label': int(label),  # Convert numpy int to Python int
                    'size': len(cluster_memories),
                    'summary': cluster_summary,
                    'emotional_patterns': emotional_patterns,
                    'memories': cluster_memories,
                    'created_at': datetime.now().isoformat(),
                    'coherence_score': self.calculate_cluster_coherence(cluster_contents)
                }
                
                clusters.append(cluster)
            
            # Sort clusters by size and coherence
            clusters.sort(key=lambda x: (x['size'], x['coherence_score']), reverse=True)
            
            return clusters
            
        except Exception as e:
            logging.error(f"[ERROR] Cluster processing failed: {e}")
            return []
    
    def generate_cluster_summary(self, contents: List[str], memories: List[Dict]) -> Dict[str, Any]:
        """Generate summary for a cluster"""
        try:
            # Extract key themes using TF-IDF
            if len(contents) < 2:
                return {'themes': [], 'description': 'Single interaction cluster'}
            
            # Simple keyword extraction
            all_text = ' '.join(contents).lower()
            
            # Common conversation themes
            theme_keywords = {
                'work_career': ['work', 'job', 'career', 'office', 'meeting', 'project', 'business'],
                'relationships': ['family', 'friend', 'relationship', 'love', 'dating', 'partner'],
                'health_wellness': ['health', 'doctor', 'exercise', 'sick', 'medicine', 'wellness'],
                'emotions_feelings': ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited', 'worried'],
                'learning_growth': ['learn', 'study', 'education', 'growth', 'develop', 'improve'],
                'hobbies_interests': ['hobby', 'interest', 'fun', 'enjoy', 'play', 'music', 'book'],
                'goals_planning': ['goal', 'plan', 'future', 'dream', 'ambition', 'achieve'],
                'daily_life': ['day', 'morning', 'evening', 'routine', 'home', 'life']
            }
            
            detected_themes = []
            for theme, keywords in theme_keywords.items():
                if any(keyword in all_text for keyword in keywords):
                    detected_themes.append(theme.replace('_', ' ').title())
            
            # Time span analysis
            timestamps = [memory.get('timestamp') for memory in memories if memory.get('timestamp')]
            time_span = 'recent'
            if timestamps:
                timestamps.sort()
                earliest = datetime.fromisoformat(timestamps[0])
                latest = datetime.fromisoformat(timestamps[-1])
                days_span = (latest - earliest).days
                
                if days_span > 30:
                    time_span = 'extended'
                elif days_span > 7:
                    time_span = 'weekly'
                else:
                    time_span = 'recent'
            
            return {
                'themes': detected_themes[:3],  # Top 3 themes
                'description': f"Cluster of {len(memories)} interactions about {', '.join(detected_themes[:2]) if detected_themes else 'various topics'}",
                'time_span': time_span,
                'interaction_count': len(memories)
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Cluster summary generation failed: {e}")
            return {'themes': [], 'description': 'Mixed interaction cluster'}
    
    def analyze_cluster_emotions(self, memories: List[Dict]) -> Dict[str, Any]:
        """Analyze emotional patterns in a cluster"""
        try:
            emotions = []
            sentiment_scores = []
            
            for memory in memories:
                emotional_context = memory.get('emotional_context', {})
                emotion = emotional_context.get('dominant_emotion')
                if emotion:
                    emotions.append(emotion)
                
                # If sentiment is available (we'd need to add this to memory structure)
                # sentiment_scores.append(memory.get('sentiment', 0.5))
            
            if not emotions:
                return {'dominant_emotion': 'neutral', 'emotion_variety': 0}
            
            # Count emotion frequencies
            emotion_counts = Counter(emotions)
            dominant_emotion = emotion_counts.most_common(1)[0][0]
            emotion_variety = len(set(emotions))
            
            return {
                'dominant_emotion': dominant_emotion,
                'emotion_variety': emotion_variety,
                'emotion_distribution': dict(emotion_counts),
                'emotional_consistency': emotion_counts[dominant_emotion] / len(emotions)
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Emotion analysis failed: {e}")
            return {'dominant_emotion': 'neutral', 'emotion_variety': 0}
    
    def calculate_cluster_coherence(self, contents: List[str]) -> float:
        """Calculate coherence score for a cluster"""
        try:
            if len(contents) < 2:
                return 1.0
            
            # Use TF-IDF to measure content similarity
            vectors = self.vectorizer.fit_transform(contents)
            
            # Calculate pairwise similarities
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(vectors)
            
            # Calculate average similarity (excluding diagonal)
            n = len(contents)
            total_similarity = 0
            count = 0
            
            for i in range(n):
                for j in range(i + 1, n):
                    total_similarity += similarities[i][j]
                    count += 1
            
            return total_similarity / count if count > 0 else 0.0
            
        except Exception as e:
            logging.error(f"[ERROR] Coherence calculation failed: {e}")
            return 0.0
    
    def calculate_cluster_statistics(self) -> Dict[str, Any]:
        """Calculate overall cluster statistics"""
        try:
            if not self.clusters:
                return {}
            
            # Size distribution
            sizes = [cluster['size'] for cluster in self.clusters]
            
            # Theme analysis
            all_themes = []
            for cluster in self.clusters:
                themes = cluster.get('summary', {}).get('themes', [])
                all_themes.extend(themes)
            
            theme_counts = Counter(all_themes)
            
            # Emotional patterns
            dominant_emotions = [
                cluster.get('emotional_patterns', {}).get('dominant_emotion', 'neutral')
                for cluster in self.clusters
            ]
            emotion_counts = Counter(dominant_emotions)
            
            return {
                'total_clusters': len(self.clusters),
                'avg_cluster_size': float(np.mean(sizes)),  # Convert numpy float to Python float
                'size_distribution': {
                    'min': min(sizes),
                    'max': max(sizes),
                    'std': float(np.std(sizes))  # Convert numpy float to Python float
                },
                'top_themes': dict(theme_counts.most_common(5)),
                'emotion_distribution': dict(emotion_counts),
                'avg_coherence': float(np.mean([cluster.get('coherence_score', 0) for cluster in self.clusters]))  # Convert numpy float to Python float
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Statistics calculation failed: {e}")
            return {}
    
    def get_cluster_by_theme(self, theme: str) -> Optional[Dict]:
        """Get cluster that best matches a theme"""
        try:
            theme_lower = theme.lower()
            
            for cluster in self.clusters:
                themes = cluster.get('summary', {}).get('themes', [])
                if any(theme_lower in t.lower() for t in themes):
                    return cluster
            
            return None
            
        except Exception as e:
            logging.error(f"[ERROR] Theme cluster search failed: {e}")
            return None
    
    def get_clusters_by_emotion(self, emotion: str) -> List[Dict]:
        """Get clusters filtered by dominant emotion"""
        try:
            matching_clusters = []
            
            for cluster in self.clusters:
                dominant_emotion = cluster.get('emotional_patterns', {}).get('dominant_emotion')
                if dominant_emotion and dominant_emotion.lower() == emotion.lower():
                    matching_clusters.append(cluster)
            
            return matching_clusters
            
        except Exception as e:
            logging.error(f"[ERROR] Emotion cluster search failed: {e}")
            return []

# Global clustering engine
clustering_engine = SemanticClusteringEngine()

def initialize_clustering():
    """Initialize clustering system"""
    return clustering_engine.initialize_clustering()

def cluster_memories(memories: List[Dict] = None) -> List[Dict]:
    """Cluster memories"""
    return clustering_engine.cluster_memories(memories)

def get_memory_clusters() -> List[Dict]:
    """Get all memory clusters"""
    return clustering_engine.clusters

def get_cluster_statistics() -> Dict[str, Any]:
    """Get cluster statistics"""
    return clustering_engine.calculate_cluster_statistics()

def search_clusters_by_theme(theme: str) -> Optional[Dict]:
    """Search clusters by theme"""
    return clustering_engine.get_cluster_by_theme(theme)
