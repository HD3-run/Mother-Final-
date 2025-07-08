import logging
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
import json

ML_MODELS_DIR = 'data/models'
MODEL_METADATA_FILE = 'data/model_metadata.json'

class MLUtilities:
    """Machine learning utilities for the MotherX AI system"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.model_metadata = {}
        
        self.ensure_directories()
        self.load_model_metadata()
    
    def ensure_directories(self):
        """Ensure ML directories exist"""
        try:
            os.makedirs(ML_MODELS_DIR, exist_ok=True)
            logging.info("[🤖] ML directories initialized")
        except Exception as e:
            logging.error(f"[ERROR] ML directory creation failed: {e}")
    
    def load_model_metadata(self):
        """Load model metadata"""
        try:
            if os.path.exists(MODEL_METADATA_FILE):
                with open(MODEL_METADATA_FILE, 'r') as f:
                    self.model_metadata = json.load(f)
                logging.info(f"[🤖] Loaded metadata for {len(self.model_metadata)} models")
            else:
                self.model_metadata = {}
        except Exception as e:
            logging.error(f"[ERROR] Model metadata loading failed: {e}")
            self.model_metadata = {}
    
    def save_model_metadata(self):
        """Save model metadata"""
        try:
            with open(MODEL_METADATA_FILE, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
        except Exception as e:
            logging.error(f"[ERROR] Model metadata saving failed: {e}")
    
    def initialize_ml_models(self):
        """Initialize ML models for the system"""
        try:
            # Initialize user behavior classifier
            self._initialize_behavior_classifier()
            
            # Initialize mood predictor
            self._initialize_mood_predictor()
            
            # Initialize needs assessment model
            self._initialize_needs_model()
            
            # Initialize conversation quality predictor
            self._initialize_quality_predictor()
            
            logging.info("[🤖] ML models initialized")
            return True
            
        except Exception as e:
            logging.error(f"[ERROR] ML model initialization failed: {e}")
            return False
    
    def _initialize_behavior_classifier(self):
        """Initialize behavior pattern classifier"""
        try:
            model_name = 'behavior_classifier'
            model_path = os.path.join(ML_MODELS_DIR, f'{model_name}.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models[model_name] = model_data['model']
                    self.scalers[model_name] = model_data.get('scaler')
                    self.encoders[model_name] = model_data.get('encoder')
            else:
                # Create new model
                self.models[model_name] = RandomForestClassifier(
                    n_estimators=50,
                    random_state=42,
                    max_depth=10
                )
                self.scalers[model_name] = StandardScaler()
                self.encoders[model_name] = LabelEncoder()
                
                # Initialize metadata
                self.model_metadata[model_name] = {
                    'created_at': datetime.now().isoformat(),
                    'type': 'classifier',
                    'purpose': 'behavioral_pattern_recognition',
                    'last_trained': None,
                    'accuracy': 0.0,
                    'feature_count': 0
                }
            
        except Exception as e:
            logging.error(f"[ERROR] Behavior classifier initialization failed: {e}")
    
    def _initialize_mood_predictor(self):
        """Initialize mood prediction model"""
        try:
            model_name = 'mood_predictor'
            model_path = os.path.join(ML_MODELS_DIR, f'{model_name}.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models[model_name] = model_data['model']
                    self.scalers[model_name] = model_data.get('scaler')
                    self.encoders[model_name] = model_data.get('encoder')
            else:
                # Create new model
                self.models[model_name] = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    multi_class='ovr'
                )
                self.scalers[model_name] = StandardScaler()
                self.encoders[model_name] = LabelEncoder()
                
                # Initialize metadata
                self.model_metadata[model_name] = {
                    'created_at': datetime.now().isoformat(),
                    'type': 'classifier',
                    'purpose': 'mood_state_prediction',
                    'last_trained': None,
                    'accuracy': 0.0,
                    'feature_count': 0
                }
            
        except Exception as e:
            logging.error(f"[ERROR] Mood predictor initialization failed: {e}")
    
    def _initialize_needs_model(self):
        """Initialize user needs assessment model"""
        try:
            model_name = 'needs_assessor'
            model_path = os.path.join(ML_MODELS_DIR, f'{model_name}.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models[model_name] = model_data['model']
                    self.scalers[model_name] = model_data.get('scaler')
                    self.encoders[model_name] = model_data.get('encoder')
            else:
                # Create new model
                self.models[model_name] = RandomForestClassifier(
                    n_estimators=30,
                    random_state=42,
                    max_depth=8
                )
                self.scalers[model_name] = StandardScaler()
                self.encoders[model_name] = LabelEncoder()
                
                # Initialize metadata
                self.model_metadata[model_name] = {
                    'created_at': datetime.now().isoformat(),
                    'type': 'classifier',
                    'purpose': 'user_needs_assessment',
                    'last_trained': None,
                    'accuracy': 0.0,
                    'feature_count': 0
                }
            
        except Exception as e:
            logging.error(f"[ERROR] Needs model initialization failed: {e}")
    
    def _initialize_quality_predictor(self):
        """Initialize conversation quality predictor"""
        try:
            model_name = 'quality_predictor'
            model_path = os.path.join(ML_MODELS_DIR, f'{model_name}.pkl')
            
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.models[model_name] = model_data['model']
                    self.scalers[model_name] = model_data.get('scaler')
            else:
                # Create new model
                self.models[model_name] = LogisticRegression(
                    random_state=42,
                    max_iter=1000
                )
                self.scalers[model_name] = StandardScaler()
                
                # Initialize metadata
                self.model_metadata[model_name] = {
                    'created_at': datetime.now().isoformat(),
                    'type': 'regressor',
                    'purpose': 'conversation_quality_assessment',
                    'last_trained': None,
                    'accuracy': 0.0,
                    'feature_count': 0
                }
            
        except Exception as e:
            logging.error(f"[ERROR] Quality predictor initialization failed: {e}")
    
    def train_user_model(self, model_name: str, X: np.ndarray, y: np.ndarray,
                        feature_names: List[str] = None) -> Dict[str, Any]:
        """Train a specific user model"""
        try:
            if model_name not in self.models:
                logging.error(f"[ERROR] Model {model_name} not found")
                return {'success': False, 'error': 'Model not found'}
            
            if len(X) < 10:  # Minimum training samples
                logging.warning(f"[WARNING] Insufficient training data for {model_name}")
                return {'success': False, 'error': 'Insufficient training data'}
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
            )
            
            # Scale features
            if model_name in self.scalers and self.scalers[model_name]:
                X_train_scaled = self.scalers[model_name].fit_transform(X_train)
                X_test_scaled = self.scalers[model_name].transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Encode labels if necessary
            if model_name in self.encoders and self.encoders[model_name]:
                if hasattr(y_train[0], '__len__') and not isinstance(y_train[0], str):
                    # Already numeric
                    y_train_encoded = y_train
                    y_test_encoded = y_test
                else:
                    # Encode string labels
                    y_train_encoded = self.encoders[model_name].fit_transform(y_train)
                    y_test_encoded = self.encoders[model_name].transform(y_test)
            else:
                y_train_encoded = y_train
                y_test_encoded = y_test
            
            # Train model
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train_encoded)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = self._calculate_model_metrics(y_test_encoded, y_pred, model_name)
            
            # Save model
            self._save_model(model_name, model, feature_names)
            
            # Update metadata
            self.model_metadata[model_name].update({
                'last_trained': datetime.now().isoformat(),
                'accuracy': metrics.get('accuracy', 0.0),
                'feature_count': X.shape[1],
                'training_samples': len(X),
                'validation_metrics': metrics
            })
            
            self.save_model_metadata()
            
            logging.info(f"[🤖] Model {model_name} trained successfully. Accuracy: {metrics.get('accuracy', 0.0):.3f}")
            
            return {
                'success': True,
                'metrics': metrics,
                'training_samples': len(X),
                'model_info': self.model_metadata[model_name]
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Model training failed for {model_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict_with_model(self, model_name: str, features: np.ndarray,
                          return_probabilities: bool = False) -> Dict[str, Any]:
        """Make prediction with a trained model"""
        try:
            if model_name not in self.models:
                return {'success': False, 'error': 'Model not found'}
            
            model = self.models[model_name]
            
            # Check if model is trained
            if not hasattr(model, 'classes_') and not hasattr(model, 'coef_'):
                return {'success': False, 'error': 'Model not trained'}
            
            # Scale features
            if model_name in self.scalers and self.scalers[model_name]:
                features_scaled = self.scalers[model_name].transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Decode prediction if necessary
            if model_name in self.encoders and self.encoders[model_name]:
                try:
                    prediction = self.encoders[model_name].inverse_transform([prediction])[0]
                except (ValueError, AttributeError):
                    pass  # Keep numeric prediction
            
            result = {
                'success': True,
                'prediction': prediction,
                'model_info': self.model_metadata.get(model_name, {})
            }
            
            # Add probabilities if requested and available
            if return_probabilities and hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(features_scaled)[0]
                    if model_name in self.encoders and self.encoders[model_name]:
                        classes = self.encoders[model_name].classes_
                    else:
                        classes = model.classes_ if hasattr(model, 'classes_') else None
                    
                    if classes is not None:
                        prob_dict = dict(zip(classes, probabilities))
                        result['probabilities'] = prob_dict
                        result['confidence'] = max(probabilities)
                    else:
                        result['confidence'] = max(probabilities)
                except Exception as e:
                    logging.warning(f"[WARNING] Probability calculation failed: {e}")
            
            return result
            
        except Exception as e:
            logging.error(f"[ERROR] Prediction failed for {model_name}: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_model_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str) -> Dict[str, float]:
        """Calculate model performance metrics"""
        try:
            metrics = {}
            
            # Accuracy
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            # Classification metrics (if applicable)
            unique_labels = len(np.unique(y_true))
            if unique_labels > 1:
                if unique_labels == 2:
                    # Binary classification
                    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
                    metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
                else:
                    # Multi-class classification
                    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
                    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            return metrics
            
        except Exception as e:
            logging.error(f"[ERROR] Metrics calculation failed: {e}")
            return {'accuracy': 0.0}
    
    def _save_model(self, model_name: str, model, feature_names: List[str] = None):
        """Save trained model to disk"""
        try:
            model_data = {
                'model': model,
                'scaler': self.scalers.get(model_name),
                'encoder': self.encoders.get(model_name),
                'feature_names': feature_names,
                'saved_at': datetime.now().isoformat()
            }
            
            model_path = os.path.join(ML_MODELS_DIR, f'{model_name}.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logging.info(f"[💾] Model {model_name} saved successfully")
            
        except Exception as e:
            logging.error(f"[ERROR] Model saving failed for {model_name}: {e}")
    
    def cluster_data(self, data: np.ndarray, n_clusters: int = None,
                    method: str = 'kmeans') -> Dict[str, Any]:
        """Perform clustering on data"""
        try:
            if len(data) < 4:  # Minimum for clustering
                return {'success': False, 'error': 'Insufficient data for clustering'}
            
            # Determine optimal number of clusters if not provided
            if n_clusters is None:
                n_clusters = min(max(2, len(data) // 3), 8)
            
            # Scale data
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            # Perform clustering
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = clusterer.fit_predict(data_scaled)
                cluster_centers = clusterer.cluster_centers_
            else:
                return {'success': False, 'error': f'Clustering method {method} not supported'}
            
            # Calculate cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_mask = cluster_labels == i
                cluster_size = np.sum(cluster_mask)
                
                if cluster_size > 0:
                    cluster_stats[i] = {
                        'size': int(cluster_size),
                        'percentage': float(cluster_size / len(data) * 100),
                        'center': cluster_centers[i].tolist() if method == 'kmeans' else None
                    }
            
            # Calculate clustering quality metrics
            from sklearn.metrics import silhouette_score
            try:
                silhouette_avg = silhouette_score(data_scaled, cluster_labels)
            except:
                silhouette_avg = 0.0
            
            result = {
                'success': True,
                'labels': cluster_labels.tolist(),
                'n_clusters': n_clusters,
                'cluster_stats': cluster_stats,
                'silhouette_score': silhouette_avg,
                'method': method
            }
            
            logging.info(f"[🔗] Clustering completed: {n_clusters} clusters, silhouette score: {silhouette_avg:.3f}")
            
            return result
            
        except Exception as e:
            logging.error(f"[ERROR] Clustering failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def extract_feature_importance(self, model_name: str) -> Dict[str, Any]:
        """Extract feature importance from trained model"""
        try:
            if model_name not in self.models:
                return {'success': False, 'error': 'Model not found'}
            
            model = self.models[model_name]
            
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_).flatten()
            else:
                return {'success': False, 'error': 'Model does not support feature importance'}
            
            # Load feature names if available
            model_path = os.path.join(ML_MODELS_DIR, f'{model_name}.pkl')
            feature_names = None
            
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        model_data = pickle.load(f)
                        feature_names = model_data.get('feature_names')
                except:
                    pass
            
            # Create feature importance ranking
            if feature_names and len(feature_names) == len(importances):
                feature_importance = list(zip(feature_names, importances))
            else:
                feature_importance = list(zip([f'feature_{i}' for i in range(len(importances))], importances))
            
            # Sort by importance
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'success': True,
                'feature_importance': feature_importance,
                'top_features': feature_importance[:5],
                'model_type': type(model).__name__
            }
            
        except Exception as e:
            logging.error(f"[ERROR] Feature importance extraction failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        try:
            status = {
                'total_models': len(self.models),
                'models': {},
                'system_status': 'operational'
            }
            
            for model_name, model in self.models.items():
                model_info = self.model_metadata.get(model_name, {})
                
                # Check if model is trained
                is_trained = hasattr(model, 'classes_') or hasattr(model, 'coef_')
                
                status['models'][model_name] = {
                    'is_trained': is_trained,
                    'type': model_info.get('type', 'unknown'),
                    'purpose': model_info.get('purpose', 'unknown'),
                    'last_trained': model_info.get('last_trained'),
                    'accuracy': model_info.get('accuracy', 0.0),
                    'feature_count': model_info.get('feature_count', 0),
                    'training_samples': model_info.get('training_samples', 0)
                }
            
            # Calculate overall system health
            trained_models = sum(1 for info in status['models'].values() if info['is_trained'])
            if trained_models == 0:
                status['system_status'] = 'needs_training'
            elif trained_models < len(self.models) / 2:
                status['system_status'] = 'partially_trained'
            
            return status
            
        except Exception as e:
            logging.error(f"[ERROR] Model status check failed: {e}")
            return {'total_models': 0, 'models': {}, 'system_status': 'error'}
    
    def cleanup_old_models(self, days: int = 30):
        """Clean up old model files"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            cleaned_count = 0
            
            for filename in os.listdir(ML_MODELS_DIR):
                if filename.endswith('.pkl'):
                    file_path = os.path.join(ML_MODELS_DIR, filename)
                    
                    # Check file modification time
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if mod_time < cutoff_time:
                        # Archive old model
                        archive_dir = os.path.join(ML_MODELS_DIR, 'archived')
                        os.makedirs(archive_dir, exist_ok=True)
                        
                        archive_path = os.path.join(archive_dir, f"{filename}.{mod_time.strftime('%Y%m%d')}")
                        os.rename(file_path, archive_path)
                        
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logging.info(f"[🧹] Archived {cleaned_count} old model files")
            
            return cleaned_count
            
        except Exception as e:
            logging.error(f"[ERROR] Model cleanup failed: {e}")
            return 0

# Global ML utilities instance
ml_utils = MLUtilities()

def initialize_ml_models():
    """Initialize ML models"""
    return ml_utils.initialize_ml_models()

def train_user_model(model_name: str = 'behavior_classifier', retrain: bool = False) -> bool:
    """Train user behavior models"""
    try:
        # This would typically get real training data from the conversation history
        # For now, we'll create a minimal training setup
        
        if not retrain:
            # Check if model is already trained
            status = ml_utils.get_model_status()
            model_info = status.get('models', {}).get(model_name, {})
            if model_info.get('is_trained'):
                logging.info(f"[🤖] Model {model_name} already trained")
                return True
        
        # Generate minimal training data (in real implementation, this would come from actual data)
        from memory.episodic_logger import get_conversation_history
        history = get_conversation_history(days=30)
        
        if len(history) < 10:
            logging.warning("[WARNING] Insufficient conversation history for training")
            return False
        
        # Extract features and labels (simplified)
        features = []
        labels = []
        
        for interaction in history:
            # Simple feature extraction
            user_input = interaction.get('user_input', '')
            emotional_context = interaction.get('emotional_context', {})
            
            feature_vector = [
                len(user_input),
                user_input.count('?'),
                user_input.count('!'),
                emotional_context.get('emotional_intensity', 0.5),
                1 if emotional_context.get('dominant_emotion') == 'happy' else 0,
                1 if emotional_context.get('dominant_emotion') == 'sad' else 0
            ]
            
            features.append(feature_vector)
            
            # Simple label assignment
            if len(user_input) > 100:
                labels.append('verbose')
            elif len(user_input) < 20:
                labels.append('brief')
            else:
                labels.append('normal')
        
        if len(features) < 10:
            return False
        
        # Train the model
        X = np.array(features)
        y = np.array(labels)
        
        result = ml_utils.train_user_model(model_name, X, y, 
                                         ['input_length', 'questions', 'exclamations', 
                                          'emotional_intensity', 'is_happy', 'is_sad'])
        
        return result.get('success', False)
        
    except Exception as e:
        logging.error(f"[ERROR] User model training failed: {e}")
        return False

def get_model_predictions() -> Dict[str, Any]:
    """Get model predictions and status"""
    return ml_utils.get_model_status()

def predict_user_behavior(features: np.ndarray) -> Dict[str, Any]:
    """Predict user behavior"""
    return ml_utils.predict_with_model('behavior_classifier', features, return_probabilities=True)

def cluster_memories(data: np.ndarray) -> Dict[str, Any]:
    """Cluster memory data"""
    return ml_utils.cluster_data(data)
