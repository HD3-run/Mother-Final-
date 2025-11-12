import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

from memory.episodic_logger import get_conversation_history
from memory.structured_store import all_facts
from models import BehaviorPattern, PredictiveModel
from app import db

class PredictiveModelingEngine:
    """Engine for predictive user modeling and behavior analysis"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.behavior_patterns = {}
        self.prediction_cache = {}
        self.cache_timeout = 3600  # 1 hour cache
        
        # Feature extractors
        self.feature_extractors = {
            'temporal': self._extract_temporal_features,
            'linguistic': self._extract_linguistic_features,
            'emotional': self._extract_emotional_features,
            'behavioral': self._extract_behavioral_features
        }
        
        self.load_models()
    
    def load_models(self):
        """Load trained models from storage"""
        try:
            models_dir = 'data/models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Load mood prediction model
            mood_model_path = os.path.join(models_dir, 'mood_predictor.pkl')
            if os.path.exists(mood_model_path):
                with open(mood_model_path, 'rb') as f:
                    self.models['mood'] = pickle.load(f)
            
            # Load needs prediction model
            needs_model_path = os.path.join(models_dir, 'needs_predictor.pkl')
            if os.path.exists(needs_model_path):
                with open(needs_model_path, 'rb') as f:
                    self.models['needs'] = pickle.load(f)
            
            # Load behavior pattern model
            behavior_model_path = os.path.join(models_dir, 'behavior_predictor.pkl')
            if os.path.exists(behavior_model_path):
                with open(behavior_model_path, 'rb') as f:
                    self.models['behavior'] = pickle.load(f)
            
            # Load scalers
            scaler_path = os.path.join(models_dir, 'feature_scalers.pkl')
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scalers = pickle.load(f)
            
            logging.info(f"[🤖] Loaded {len(self.models)} predictive models")
            
        except Exception as e:
            logging.error(f"[ERROR] Model loading failed: {e}")
            self.models = {}
            self.scalers = {}
    
    def save_models(self):
        """Save trained models to storage"""
        try:
            models_dir = 'data/models'
            os.makedirs(models_dir, exist_ok=True)
            
            # Save individual models
            for model_name, model in self.models.items():
                model_path = os.path.join(models_dir, f'{model_name}_predictor.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
            
            # Save scalers
            if self.scalers:
                scaler_path = os.path.join(models_dir, 'feature_scalers.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(self.scalers, f)
            
            logging.info(f"[💾] Saved {len(self.models)} predictive models")
            
        except Exception as e:
            logging.error(f"[ERROR] Model saving failed: {e}")
    
    def predict_user_state(self, user_input: str, emotional_context: Dict) -> Dict[str, Any]:
        """Predict user's future state and needs"""
        try:
            # Check cache first
            cache_key = f"predict_{hash(user_input)}_{hash(str(emotional_context))}"
            if cache_key in self.prediction_cache:
                cached_result, timestamp = self.prediction_cache[cache_key]
                if datetime.now().timestamp() - timestamp < self.cache_timeout:
                    return cached_result
            
            # Get conversation history for context
            history = get_conversation_history(days=7)
            
            # Extract features
            features = self._extract_prediction_features(user_input, emotional_context, history)
            
            predictions = {}
            
            # Predict mood trajectory
            if 'mood' in self.models:
                mood_prediction = self._predict_mood(features)
                predictions['predicted_mood'] = mood_prediction
            
            # Predict user needs
            if 'needs' in self.models:
                needs_prediction = self._predict_needs(features, emotional_context)
                predictions['predicted_needs'] = needs_prediction
            
            # Predict behavioral patterns
            if 'behavior' in self.models:
                behavior_prediction = self._predict_behavior(features, history)
                predictions['behavioral_pattern'] = behavior_prediction
            
            # Generate interaction recommendations
            recommendations = self._generate_interaction_recommendations(predictions, emotional_context)
            predictions['interaction_recommendation'] = recommendations
            
            # Cache result
            self.prediction_cache[cache_key] = (predictions, datetime.now().timestamp())
            
            logging.info(f"[🔮] Generated predictions: {list(predictions.keys())}")
            return predictions
            
        except Exception as e:
            logging.error(f"[ERROR] User state prediction failed: {e}")
            return {
                'predicted_mood': 'stable',
                'predicted_needs': ['general_support'],
                'behavioral_pattern': 'normal',
                'interaction_recommendation': 'be_supportive'
            }
    
    def update_behavioral_patterns(self, user_input: str):
        """Update behavioral patterns based on user input"""
        try:
            # Extract behavioral features
            patterns = self._extract_behavioral_patterns(user_input)
            
            for pattern_type, pattern_data in patterns.items():
                # Check if pattern exists in database
                existing_pattern = BehaviorPattern.query.filter_by(
                    pattern_type=pattern_type
                ).first()
                
                if existing_pattern:
                    # Update existing pattern
                    existing_pattern.frequency += 1
                    existing_pattern.pattern_data = json.dumps(pattern_data)
                    existing_pattern.last_observed = datetime.utcnow()
                else:
                    # Create new pattern
                    new_pattern = BehaviorPattern(
                        pattern_type=pattern_type,
                        pattern_data=json.dumps(pattern_data),
                        frequency=1,
                        confidence=0.5,
                        last_observed=datetime.utcnow()
                    )
                    db.session.add(new_pattern)
            
            db.session.commit()
            logging.info(f"[📊] Updated {len(patterns)} behavioral patterns")
            
        except Exception as e:
            logging.error(f"[ERROR] Behavioral pattern update failed: {e}")
    
    def train_models(self, retrain: bool = False):
        """Train predictive models on historical data"""
        try:
            # Get training data
            history = get_conversation_history(days=30)  # Last 30 days
            
            if len(history) < 20:  # Need minimum data for training
                logging.warning("[WARNING] Insufficient data for model training")
                return False
            
            # Prepare training data
            X, y_mood, y_needs, y_behavior = self._prepare_training_data(history)
            
            if len(X) == 0:
                logging.warning("[WARNING] No features extracted for training")
                return False
            
            # Train mood prediction model
            if len(set(y_mood)) > 1:  # Need variation in target
                self._train_mood_model(X, y_mood)
            
            # Train needs prediction model
            if len(set(y_needs)) > 1:
                self._train_needs_model(X, y_needs)
            
            # Train behavior prediction model
            if len(set(y_behavior)) > 1:
                self._train_behavior_model(X, y_behavior)
            
            # Save trained models
            self.save_models()
            
            # Update model metadata in database
            self._update_model_metadata(len(history))
            
            logging.info("[🤖] Model training completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"[ERROR] Model training failed: {e}")
            return False
    
    def _extract_prediction_features(self, user_input: str, emotional_context: Dict, 
                                   history: List[Dict]) -> np.ndarray:
        """Extract features for prediction"""
        try:
            features = []
            
            # Temporal features
            temporal_features = self._extract_temporal_features(history)
            features.extend(temporal_features)
            
            # Linguistic features
            linguistic_features = self._extract_linguistic_features(user_input)
            features.extend(linguistic_features)
            
            # Emotional features
            emotional_features = self._extract_emotional_features(emotional_context)
            features.extend(emotional_features)
            
            # Behavioral features
            behavioral_features = self._extract_behavioral_features(history)
            features.extend(behavioral_features)
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logging.error(f"[ERROR] Feature extraction failed: {e}")
            return np.array([]).reshape(1, -1)
    
    def _extract_temporal_features(self, data) -> List[float]:
        """Extract time-based features"""
        try:
            features = []
            now = datetime.now()
            
            if isinstance(data, list) and data:  # History data
                # Time since last interaction
                last_interaction = datetime.fromisoformat(data[-1]['timestamp'])
                hours_since_last = (now - last_interaction).total_seconds() / 3600
                features.append(min(hours_since_last, 168))  # Cap at 1 week
                
                # Interaction frequency (last 24 hours)
                recent_interactions = [
                    d for d in data
                    if datetime.fromisoformat(d['timestamp']) > now - timedelta(hours=24)
                ]
                features.append(len(recent_interactions))
                
                # Day of week
                features.append(now.weekday())
                
                # Hour of day
                features.append(now.hour)
                
            else:  # Current interaction
                features.extend([0.0, 1.0, now.weekday(), now.hour])
            
            return features
            
        except Exception as e:
            logging.error(f"[ERROR] Temporal feature extraction failed: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_linguistic_features(self, text: str) -> List[float]:
        """Extract linguistic features from text"""
        try:
            if not text:
                return [0.0] * 8
            
            features = []
            
            # Text length
            features.append(min(len(text), 1000) / 1000.0)  # Normalized
            
            # Word count
            words = text.split()
            features.append(min(len(words), 200) / 200.0)  # Normalized
            
            # Average word length
            if words:
                avg_word_len = sum(len(word) for word in words) / len(words)
                features.append(min(avg_word_len, 15) / 15.0)
            else:
                features.append(0.0)
            
            # Question marks
            features.append(text.count('?') / max(len(text), 1))
            
            # Exclamation marks
            features.append(text.count('!') / max(len(text), 1))
            
            # Capital letters ratio
            if text:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
                features.append(caps_ratio)
            else:
                features.append(0.0)
            
            # Sentiment indicators
            positive_words = ['good', 'great', 'happy', 'love', 'like', 'amazing', 'wonderful']
            negative_words = ['bad', 'sad', 'hate', 'terrible', 'awful', 'horrible', 'angry']
            
            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            features.append(positive_count / max(len(words), 1))
            features.append(negative_count / max(len(words), 1))
            
            return features
            
        except Exception as e:
            logging.error(f"[ERROR] Linguistic feature extraction failed: {e}")
            return [0.0] * 8
    
    def _extract_emotional_features(self, emotional_context: Dict) -> List[float]:
        """Extract emotional features"""
        try:
            features = []
            
            # Emotional intensity
            features.append(emotional_context.get('emotional_intensity', 0.5))
            
            # Dominant emotion encoding
            emotion_mapping = {
                'happy': 1.0,
                'sad': -1.0,
                'angry': -0.8,
                'excited': 0.8,
                'worried': -0.6,
                'neutral': 0.0
            }
            
            emotion = emotional_context.get('dominant_emotion', 'neutral')
            features.append(emotion_mapping.get(emotion, 0.0))
            
            # Number of emotional indicators
            indicators = emotional_context.get('emotional_indicators', [])
            features.append(min(len(indicators), 10) / 10.0)
            
            return features
            
        except Exception as e:
            logging.error(f"[ERROR] Emotional feature extraction failed: {e}")
            return [0.0, 0.0, 0.0]
    
    def _extract_behavioral_features(self, data) -> List[float]:
        """Extract behavioral pattern features"""
        try:
            features = []
            
            if isinstance(data, list) and data:  # History data
                # Conversation length patterns
                lengths = [len(d.get('user_input', '')) for d in data[-10:]]
                features.append(np.mean(lengths) / 1000.0 if lengths else 0.0)
                features.append(np.std(lengths) / 1000.0 if len(lengths) > 1 else 0.0)
                
                # Response time patterns (if available)
                # This would require tracking response times
                features.append(0.5)  # Placeholder
                
                # Topic consistency
                # Simple heuristic: count repeated keywords
                all_text = ' '.join([d.get('user_input', '') for d in data[-5:]])
                words = all_text.lower().split()
                if words:
                    word_counts = Counter(words)
                    common_words = [word for word, count in word_counts.items() if count > 1]
                    consistency = len(common_words) / len(set(words))
                    features.append(consistency)
                else:
                    features.append(0.0)
                
            else:  # Current interaction features
                features.extend([0.0, 0.0, 0.5, 0.0])
            
            return features
            
        except Exception as e:
            logging.error(f"[ERROR] Behavioral feature extraction failed: {e}")
            return [0.0, 0.0, 0.0, 0.0]
    
    def _extract_behavioral_patterns(self, user_input: str) -> Dict[str, Any]:
        """Extract behavioral patterns from user input"""
        try:
            patterns = {}
            
            text = user_input.lower()
            
            # Communication patterns
            if len(user_input) > 200:
                patterns['verbose_communication'] = {'length': len(user_input)}
            elif len(user_input) < 20:
                patterns['brief_communication'] = {'length': len(user_input)}
            
            # Question patterns
            question_count = text.count('?')
            if question_count > 2:
                patterns['high_curiosity'] = {'questions': question_count}
            
            # Emotional expression patterns
            emotional_words = ['feel', 'emotion', 'happy', 'sad', 'angry', 'excited']
            if any(word in text for word in emotional_words):
                patterns['emotional_expression'] = {'present': True}
            
            # Help-seeking patterns
            help_words = ['help', 'support', 'advice', 'guidance']
            if any(word in text for word in help_words):
                patterns['help_seeking'] = {'present': True}
            
            return patterns
            
        except Exception as e:
            logging.error(f"[ERROR] Behavioral pattern extraction failed: {e}")
            return {}
    
    def _predict_mood(self, features: np.ndarray) -> str:
        """Predict user's mood trajectory"""
        try:
            if 'mood' not in self.models or features.shape[1] == 0:
                return 'stable'
            
            # Scale features
            if 'mood' in self.scalers:
                features_scaled = self.scalers['mood'].transform(features)
            else:
                features_scaled = features
            
            prediction = self.models['mood'].predict(features_scaled)[0]
            confidence = max(self.models['mood'].predict_proba(features_scaled)[0])
            
            if confidence > 0.7:
                return prediction
            else:
                return 'uncertain'
                
        except Exception as e:
            logging.error(f"[ERROR] Mood prediction failed: {e}")
            return 'stable'
    
    def _predict_needs(self, features: np.ndarray, emotional_context: Dict) -> List[str]:
        """Predict user's likely needs"""
        try:
            predicted_needs = []
            
            # Rule-based need prediction
            emotion = emotional_context.get('dominant_emotion', 'neutral')
            intensity = emotional_context.get('emotional_intensity', 0.5)
            
            if emotion in ['sad', 'worried', 'angry'] and intensity > 0.6:
                predicted_needs.append('emotional_support')
            
            if emotion == 'excited' and intensity > 0.7:
                predicted_needs.append('celebration_sharing')
            
            # Model-based prediction (if available)
            if 'needs' in self.models and features.shape[1] > 0:
                try:
                    if 'needs' in self.scalers:
                        features_scaled = self.scalers['needs'].transform(features)
                    else:
                        features_scaled = features
                    
                    model_prediction = self.models['needs'].predict(features_scaled)[0]
                    if model_prediction not in predicted_needs:
                        predicted_needs.append(model_prediction)
                        
                except Exception as e:
                    logging.warning(f"[WARNING] Model-based need prediction failed: {e}")
            
            # Default needs if none predicted
            if not predicted_needs:
                predicted_needs = ['general_support', 'conversation']
            
            return predicted_needs
            
        except Exception as e:
            logging.error(f"[ERROR] Needs prediction failed: {e}")
            return ['general_support']
    
    def _predict_behavior(self, features: np.ndarray, history: List[Dict]) -> str:
        """Predict behavioral patterns"""
        try:
            if not history:
                return 'new_user'
            
            # Analyze recent behavior patterns
            recent_interactions = history[-5:]
            
            # Check for patterns
            if len(recent_interactions) >= 3:
                lengths = [len(interaction.get('user_input', '')) for interaction in recent_interactions]
                avg_length = sum(lengths) / len(lengths)
                
                if avg_length > 200:
                    return 'detailed_communicator'
                elif avg_length < 30:
                    return 'brief_communicator'
                else:
                    return 'balanced_communicator'
            
            return 'establishing_pattern'
            
        except Exception as e:
            logging.error(f"[ERROR] Behavior prediction failed: {e}")
            return 'unknown_pattern'
    
    def _generate_interaction_recommendations(self, predictions: Dict, 
                                           emotional_context: Dict) -> str:
        """Generate interaction recommendations based on predictions"""
        try:
            mood = predictions.get('predicted_mood', 'stable')
            needs = predictions.get('predicted_needs', [])
            behavior = predictions.get('behavioral_pattern', 'normal')
            
            # Priority-based recommendation system
            if 'emotional_support' in needs:
                return 'provide_empathetic_support'
            elif mood in ['declining', 'negative']:
                return 'offer_gentle_encouragement'
            elif 'celebration_sharing' in needs:
                return 'share_enthusiasm'
            elif behavior == 'detailed_communicator':
                return 'engage_in_depth'
            elif behavior == 'brief_communicator':
                return 'keep_responses_concise'
            else:
                return 'maintain_supportive_conversation'
                
        except Exception as e:
            logging.error(f"[ERROR] Recommendation generation failed: {e}")
            return 'be_supportive'
    
    def _prepare_training_data(self, history: List[Dict]) -> Tuple[np.ndarray, List, List, List]:
        """Prepare training data from conversation history"""
        try:
            X = []
            y_mood = []
            y_needs = []
            y_behavior = []
            
            for i, interaction in enumerate(history):
                if i < 2:  # Need some history for features
                    continue
                
                # Get previous interactions for context
                context_history = history[:i]
                
                # Extract features
                user_input = interaction.get('user_input', '')
                emotional_context = interaction.get('emotional_context', {})
                
                features = self._extract_prediction_features(user_input, emotional_context, context_history)
                if features.shape[1] > 0:
                    X.append(features.flatten())
                    
                    # Create labels (simplified for example)
                    emotion = emotional_context.get('dominant_emotion', 'neutral')
                    
                    # Mood labels
                    if emotion in ['happy', 'excited']:
                        y_mood.append('positive')
                    elif emotion in ['sad', 'angry', 'worried']:
                        y_mood.append('negative')
                    else:
                        y_mood.append('stable')
                    
                    # Needs labels
                    if emotion in ['sad', 'worried']:
                        y_needs.append('emotional_support')
                    elif '?' in user_input:
                        y_needs.append('information')
                    else:
                        y_needs.append('conversation')
                    
                    # Behavior labels
                    if len(user_input) > 200:
                        y_behavior.append('detailed')
                    elif len(user_input) < 30:
                        y_behavior.append('brief')
                    else:
                        y_behavior.append('normal')
            
            return np.array(X), y_mood, y_needs, y_behavior
            
        except Exception as e:
            logging.error(f"[ERROR] Training data preparation failed: {e}")
            return np.array([]), [], [], []
    
    def _train_mood_model(self, X: np.ndarray, y: List[str]):
        """Train mood prediction model"""
        try:
            if len(set(y)) < 2:
                return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['mood'] = scaler
            
            # Train model
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_scaled, y)
            self.models['mood'] = model
            
            # Evaluate
            if len(X) > 10:
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                logging.info(f"[🎯] Mood model accuracy: {accuracy:.3f}")
            
        except Exception as e:
            logging.error(f"[ERROR] Mood model training failed: {e}")
    
    def _train_needs_model(self, X: np.ndarray, y: List[str]):
        """Train needs prediction model"""
        try:
            if len(set(y)) < 2:
                return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['needs'] = scaler
            
            # Train model
            model = LogisticRegression(random_state=42, max_iter=1000)
            model.fit(X_scaled, y)
            self.models['needs'] = model
            
            logging.info("[🎯] Needs prediction model trained")
            
        except Exception as e:
            logging.error(f"[ERROR] Needs model training failed: {e}")
    
    def _train_behavior_model(self, X: np.ndarray, y: List[str]):
        """Train behavior prediction model"""
        try:
            if len(set(y)) < 2:
                return
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            self.scalers['behavior'] = scaler
            
            # Train model
            model = RandomForestClassifier(n_estimators=30, random_state=42)
            model.fit(X_scaled, y)
            self.models['behavior'] = model
            
            logging.info("[🎯] Behavior prediction model trained")
            
        except Exception as e:
            logging.error(f"[ERROR] Behavior model training failed: {e}")
    
    def _update_model_metadata(self, training_size: int):
        """Update model metadata in database"""
        try:
            for model_name in self.models.keys():
                existing_model = PredictiveModel.query.filter_by(model_type=model_name).first()
                
                if existing_model:
                    existing_model.training_data_size = training_size
                    existing_model.last_trained = datetime.utcnow()
                else:
                    new_model = PredictiveModel(
                        model_type=model_name,
                        model_data="stored_in_pickle",
                        training_data_size=training_size,
                        accuracy_score=0.0,  # Would need proper evaluation
                        last_trained=datetime.utcnow()
                    )
                    db.session.add(new_model)
            
            db.session.commit()
            
        except Exception as e:
            logging.error(f"[ERROR] Model metadata update failed: {e}")
    
    def get_model_predictions(self) -> Dict[str, Any]:
        """Get current model prediction capabilities"""
        try:
            predictions = {
                'available_models': list(self.models.keys()),
                'model_status': {},
                'prediction_cache_size': len(self.prediction_cache)
            }
            
            # Get model metadata from database
            for model_name in self.models.keys():
                model_record = PredictiveModel.query.filter_by(model_type=model_name).first()
                if model_record:
                    predictions['model_status'][model_name] = {
                        'training_data_size': model_record.training_data_size,
                        'last_trained': model_record.last_trained.isoformat() if model_record.last_trained else None,
                        'accuracy_score': model_record.accuracy_score
                    }
            
            return predictions
            
        except Exception as e:
            logging.error(f"[ERROR] Model predictions retrieval failed: {e}")
            return {'available_models': [], 'model_status': {}}

# Global predictive modeling engine
predictive_engine = PredictiveModelingEngine()

def predict_user_state(user_input: str, emotional_context: Dict) -> Dict[str, Any]:
    """Predict user state and needs"""
    return predictive_engine.predict_user_state(user_input, emotional_context)

def update_behavioral_patterns(user_input: str):
    """Update behavioral patterns"""
    predictive_engine.update_behavioral_patterns(user_input)

def train_user_model(retrain: bool = False) -> bool:
    """Train predictive models"""
    return predictive_engine.train_models(retrain)

def get_model_predictions() -> Dict[str, Any]:
    """Get model prediction capabilities"""
    return predictive_engine.get_model_predictions()
