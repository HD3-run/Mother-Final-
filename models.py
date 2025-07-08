from app import db
from flask_login import UserMixin
from datetime import datetime
from pytz import timezone
import json

# Define the IST timezone
india_tz = timezone('Asia/Kolkata')
# Define the UTC timezone using pytz
utc_tz = timezone('UTC')

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    # Use a lambda function for default datetime to ensure it's evaluated at object creation
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(india_tz))

class IdentityState(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    personality_traits = db.Column(db.Text)  # JSON string
    core_values = db.Column(db.Text)  # JSON string
    beliefs = db.Column(db.Text)  # JSON string
    preferences = db.Column(db.Text)  # JSON string
    mood_state = db.Column(db.Text)  # JSON string
    coherence_score = db.Column(db.Float, default=0.0)
    # Corrected: Use pytz.timezone('UTC') and lambda
    last_updated = db.Column(db.DateTime, default=lambda: datetime.now(utc_tz))
    
    def get_traits(self):
        return json.loads(self.personality_traits or '{}')
    
    def set_traits(self, traits):
        self.personality_traits = json.dumps(traits)
    
    def get_values(self):
        return json.loads(self.core_values or '[]')
    
    def set_values(self, values):
        self.core_values = json.dumps(values)

class BehaviorPattern(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    pattern_type = db.Column(db.String(50), nullable=False)
    pattern_data = db.Column(db.Text)  # JSON string
    frequency = db.Column(db.Integer, default=1)
    confidence = db.Column(db.Float, default=0.0)
    # Corrected: Use pytz.timezone('UTC') and lambda
    last_observed = db.Column(db.DateTime, default=lambda: datetime.now(utc_tz))
    
class AutonomousAction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    action_type = db.Column(db.String(50), nullable=False)
    trigger_condition = db.Column(db.String(200))
    action_data = db.Column(db.Text)  # JSON string
    executed_at = db.Column(db.DateTime)
    success = db.Column(db.Boolean, default=True)
    # Corrected: Use pytz.timezone('UTC') and lambda
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(utc_tz))

class PredictiveModel(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(db.String(50), nullable=False)
    model_data = db.Column(db.Text)  # Serialized model
    accuracy_score = db.Column(db.Float, default=0.0)
    training_data_size = db.Column(db.Integer, default=0)
    # Corrected: Use pytz.timezone('UTC') and lambda
    last_trained = db.Column(db.DateTime, default=lambda: datetime.now(utc_tz))
