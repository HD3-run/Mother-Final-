import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the Base for SQLAlchemy models
class Base(DeclarativeBase):
    pass

# Initialize SQLAlchemy without an app, it will be initialized later
db = SQLAlchemy(model_class=Base)

# Setup logging (basic config, can be enhanced)
logging.basicConfig(level=logging.DEBUG)

def create_app():
    """
    Application factory function to create and configure the Flask app.
    """
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app.secret_key = os.environ.get("SESSION_SECRET", "mother-ai-secret-key-2025")
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    # Configure the database
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///motherx.db")
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }

    # Initialize the app with the SQLAlchemy extension
    db.init_app(app)

    # All app-specific initialization that needs the app context
    with app.app_context():
        # Import models to ensure tables are created.
        # It's crucial to import models *after* db.init_app(app)
        # and within the app context if they depend on `db`.
        import models  # noqa: F401 - F401 is ignored as models are imported for side effects (table creation)
        db.create_all() # Create database tables based on models

        # Initialize other components after database is ready
        # These imports are placed here to ensure the app context and db are ready
        # and to avoid potential circular import issues if these modules also import `app` or `db`.
        from memory.structured_store import init_db
        from personality.loader import load_config
        from utils.logger import setup_logging
        from startup import initialize_system
        from apscheduler.schedulers.background import BackgroundScheduler
        # Corrected import path based on the provided file structure: reflection/autonomous_decision.py
        from reflection.autonomous_decision import schedule_autonomous_tasks
        from routes import chat_api
        
        # Setup logging (can be called here if it depends on app config, or globally)
        setup_logging()
        
        # Initialize the system
        initialize_system()
        
        # Load configuration
        config = load_config()
        
        # Initialize database (if there's a separate init_db for structured_store)
        init_db()
        
        # Register blueprints
        app.register_blueprint(chat_api)
        
        # Initialize autonomous decision scheduler
        scheduler = BackgroundScheduler()
        schedule_autonomous_tasks(scheduler)
        scheduler.start()
        
        logging.info("[🚀] Enhanced MOTHER AI System initialized")
        logging.info(f"[🧠] Loaded configuration: {config.get('name', 'MOTHER')}")
        logging.info("[🤖] Autonomous decision system activated") 
        logging.info("[📊] Predictive modeling enabled")
        logging.info("[🎭] Identity formation engine started")

    return app

# This block ensures the app is created and run only when app.py is executed directly.
if __name__ == '__main__':
    app = create_app() # Create the app instance
    app.run(debug=True, host='0.0.0.0', port=5000)
