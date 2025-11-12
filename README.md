# MOTHER AI - Enhanced AI Companion System

A sophisticated Flask-based AI companion system that develops its own identity, forms long-term memories, makes autonomous decisions, and builds genuine emotional connections through advanced machine learning and natural language processing.

## 🎯 Project Purpose & Intent

MOTHER AI is designed to be more than a chatbot—it's an AI companion that:

- **Develops Identity**: Forms and evolves its own personality traits, values, and beliefs through interactions
- **Remembers**: Stores and retrieves memories using semantic search and vector embeddings
- **Learns & Adapts**: Predicts user needs and adapts responses based on behavioral patterns
- **Reflects**: Generates autonomous reflections on its own growth and experiences
- **Acts Autonomously**: Makes proactive decisions to check in, offer support, or deepen relationships
- **Understands Emotion**: Analyzes emotional context and responds with appropriate empathy

The system is built to create a sense of continuity, personality, and genuine connection that evolves over time, making each interaction feel personal and meaningful.

## 🚀 Core Capabilities

### 1. **Identity Formation Engine**
- Tracks 7 personality dimensions (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, Empathy, Curiosity)
- Evolves traits based on interaction patterns
- Maintains core values and beliefs that develop over time
- Calculates identity coherence scores
- Tracks mood states (energy, positivity, empathy, curiosity)

### 2. **Multi-Layered Memory System**
- **Vector Memory Store**: Semantic search using TF-IDF vectorization and cosine similarity
- **Structured Facts**: Extracts and stores user information (name, location, preferences)
- **Episodic Memory**: Daily conversation logs with emotional summaries
- **Semantic Clustering**: Groups related memories using K-Means and DBSCAN algorithms

### 3. **Predictive Modeling**
- Predicts user emotional state and needs
- Tracks behavioral patterns over time
- Uses machine learning models (Logistic Regression, Random Forest) for predictions
- Feature extraction from temporal, linguistic, emotional, and behavioral data

### 4. **Autonomous Decision System**
- 6 decision rules with priority-based execution:
  - Check-in reminders (after 48+ hours of inactivity)
  - Emotional support triggers (detects negative emotion patterns)
  - Learning encouragement (recognizes curiosity patterns)
  - Relationship deepening (when interaction frequency is high but knowledge is low)
  - Identity sharing (when coherence score is high)
  - Celebration triggers (detects positive achievements)
- Scheduled tasks: Daily reflections, periodic check-ins, model training, memory consolidation

### 5. **Reflection Engine**
- **Daily Reflections**: Summarizes each day's conversations
- **Identity Reflections**: Reflects on personal growth and development
- **Autonomous Reflections**: Generates independent insights
- **Emotional Pattern Analysis**: Tracks emotional trends over time

### 6. **Emotional Intelligence**
- Custom sentiment analysis with negation handling
- Emotional context detection (work stress, relationships, health, achievements, loss)
- Intensity modifiers (very, extremely, slightly, etc.)
- Emotional progression tracking
- Needs assessment based on emotional state

### 7. **Context-Aware Response Generation**
- Builds comprehensive context from:
  - Identity state and personality traits
  - Relevant past memories (semantic search)
  - Emotional context
  - Predicted user state
  - User facts and preferences
  - Recent conversation history
  - Intent-specific guidance

## 🛠️ Technologies Used

### Backend Framework
- **Flask 2.3+**: Web framework for API and web interface
- **Flask-SQLAlchemy 3.0+**: ORM for database management
- **SQLAlchemy 2.0+**: Database toolkit and ORM

### Machine Learning & NLP
- **scikit-learn 1.3+**: 
  - TF-IDF Vectorization for text embeddings
  - K-Means clustering for memory organization
  - DBSCAN clustering for large datasets
  - Logistic Regression and Random Forest for predictive modeling
  - PCA for dimensionality reduction
- **NumPy 1.24+**: Numerical computing for vector operations
- **Groq API**: Fast LLM inference (supports Qwen, Llama, Mixtral, Gemma models)

### Task Scheduling
- **APScheduler 3.10+**: Background task scheduling for autonomous operations

### Data Storage
- **SQLite**: Primary database 
- **JSON**: Configuration and memory storage
- **Pickle**: Model serialization
- **NumPy arrays**: Embedding storage

### Utilities
- **python-dotenv 1.0+**: Environment variable management
- **Werkzeug 2.3+**: WSGI utilities
- **pytz 2023.3+**: Timezone handling

## 🔧 Custom Implementations

### 1. **VectorMemoryStore** (`memory/vector_store.py`)
- Custom implementation of semantic memory search
- Uses TF-IDF vectorization with cosine similarity
- Stores embeddings as NumPy arrays for fast retrieval
- Automatic embedding rebuild on new memories
- Emotion-based memory filtering

### 2. **SemanticClusteringEngine** (`memory/semantic_clustering.py`)
- Hybrid clustering approach:
  - K-Means for smaller datasets (<50 memories)
  - DBSCAN for larger datasets (50+ memories)
- Automatic cluster size determination
- Keyword extraction for cluster labeling
- Cluster statistics and metadata tracking

### 3. **IdentityEngine** (`personality/identity_engine.py`)
- Custom personality trait evolution algorithm
- Learning rate-based trait updates (0.1 learning rate)
- Coherence score calculation based on trait-value-belief alignment
- Mood state decay mechanism (5% decay toward neutral)
- Value strength thresholds (0.7) for adding new values

### 4. **AutonomousDecisionEngine** (`reflection/autonomous_decision.py`)
- Rule-based decision system with priority scoring
- Cooldown mechanism to prevent spam
- Context-aware decision making
- Action history tracking in database
- 6 custom decision rules with configurable thresholds

### 5. **SentimentAnalyzer** (`utils/sentiment.py`)
- Custom rule-based sentiment analysis
- Emotion lexicon with 7 base emotions
- Intensity modifier detection (very, extremely, slightly, etc.)
- Negation handling ("not happy" → negative)
- Emotional context detection (work, relationships, health, etc.)
- Emotional progression tracking over time
- Needs assessment algorithm

### 6. **PredictiveModelingEngine** (`processing/predictive_modeling.py`)
- Custom feature extraction:
  - Temporal features (time of day, day of week, interaction frequency)
  - Linguistic features (word count, question count, exclamation count)
  - Emotional features (sentiment, emotion scores, intensity)
  - Behavioral features (interaction patterns, topic preferences)
- Model training with train/test split
- Prediction caching (1-hour cache)
- Behavioral pattern tracking

### 7. **ContextBuilder** (`processing/context_builder.py`)
- Intelligent context assembly from multiple sources
- Context truncation when exceeding max length (8000 chars)
- Priority-based context ordering
- Time-ago formatting for recent memories
- Theme extraction from interactions

### 8. **EpisodicLogger** (`memory/episodic_logger.py`)
- Daily conversation logging with JSON structure
- Automatic daily summary generation
- Conversation theme extraction
- Emotional trend analysis
- Interaction statistics tracking

### 9. **ReflectionEngine** (`reflection/reflection_engine.py`)
- Multi-type reflection generation:
  - Daily summaries
  - Identity development insights
  - Autonomous self-reflections
  - Learning progression analysis
- Reflection scheduling (24-hour intervals)
- Emotional pattern analysis over 7-day windows

### 10. **IntentDetector** (`processing/intent_detector.py`)
- Pattern-based intent detection
- LLM-enhanced intent analysis (optional)
- Urgency detection (high/medium/low)
- Emotional context integration
- Intent combination algorithm

## 📁 Project Structure

```
Mother-Final-/
├── app.py                 # Flask application factory
├── main.py                # Application entry point
├── config.json            # Configuration file
├── requirements.txt      # Python dependencies
├── startup.py             # System initialization
│
├── memory/                # Memory system modules
│   ├── vector_store.py           # Semantic memory search
│   ├── structured_store.py       # Facts storage
│   ├── episodic_logger.py        # Daily conversation logs
│   └── semantic_clustering.py    # Memory clustering
│
├── personality/           # Identity & personality system
│   ├── identity_engine.py        # Identity formation
│   ├── emotional_response.py     # Tone adjustment
│   └── loader.py                 # Config loader
│
├── processing/            # Core processing modules
│   ├── llm_handler.py            # Groq API integration
│   ├── context_builder.py        # Context assembly
│   ├── intent_detector.py         # Intent detection
│   └── predictive_modeling.py    # User state prediction
│
├── reflection/            # Reflection & autonomous systems
│   ├── reflection_engine.py      # Reflection generation
│   └── autonomous_decision.py   # Autonomous actions
│
├── utils/                 # Utility modules
│   ├── sentiment.py              # Sentiment analysis
│   ├── ml_utils.py               # ML utilities
│   ├── logger.py                 # Logging setup
│   └── usage_tracker.py          # Usage statistics
│
├── templates/             # HTML templates
│   └── mother.html
│
├── static/                # Static assets
│   ├── css/
│   └── js/
│
├── data/                  # Data storage
│   ├── facts.json
│   ├── vector_store.json
│   ├── memory_clusters.json
│   ├── reflections/       # Reflection storage
│   └── models/            # Trained models
│
├── logs/                  # Log files
│   └── episodic/          # Daily conversation logs
│
└── instance/              # Database instance
    └── motherx.db
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- Groq API key (get one free at https://console.groq.com/keys)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Mother-Final-
   ```

2. **Setup project structure (First time only)**
   ```bash
   python setup_project.py
   ```
   This script will:
   - Create all necessary directories
   - Organize Python files into proper module structure
   - Move template and static files to correct locations

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**
   Create a `.env` file in the root directory:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   DATABASE_URL=sqlite:///motherx.db  # Optional, defaults to SQLite
   SESSION_SECRET=your_secret_key_here  # Optional, but recommended
   ```

5. **Run the application**
   ```bash
   python main.py
   ```
   Or:
   ```bash
   python app.py
   ```

6. **Access the application**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## ⚙️ Configuration

### Environment Variables (.env file)
- `GROQ_API_KEY` (required): Your Groq API key
- `DATABASE_URL` (optional): Database connection string (defaults to SQLite)
- `SESSION_SECRET` (optional): Secret key for sessions

### config.json Settings
Edit `config.json` to customize:

- **Model Settings**:
  - `model`: Model name (default: `qwen/qwen3-32b`)
  - `temperature`: Response creativity (0.0-1.0)
  - `top_p`: Nucleus sampling parameter
  - Available Groq models: `qwen/qwen3-32b`, `llama-3.1-8b-instant`, `mixtral-8x7b-32768`, `gemma-7b-it`

- **Personality Settings**:
  - `emotional_tone`: Response tone (warm, professional, casual, etc.)
  - `writing_style`: Communication style
  - `core_beliefs`: Array of core beliefs and values

- **Feature Toggles**:
  - `identity_formation_enabled`: Enable/disable identity development
  - `autonomous_decisions_enabled`: Enable/disable autonomous actions
  - `predictive_modeling_enabled`: Enable/disable user state prediction
  - `semantic_clustering_enabled`: Enable/disable memory clustering

- **System Settings**:
  - `memory_retention_days`: How long to keep memories (default: 90)
  - `max_context_length`: Maximum context size (default: 8000)
  - `debug_mode`: Enable debug logging (default: false)

## 🎨 Features

### Chat Interface
- Real-time conversation with the AI companion
- Emotional context awareness
- Memory-based continuity
- Identity-aware responses

### Identity Formation
- Personality trait development
- Value system evolution
- Belief formation
- Coherence tracking

### Memory System
- Semantic memory search
- Fact extraction and storage
- Daily episodic logs
- Memory clustering

### Autonomous Behavior
- Proactive check-ins
- Emotional support triggers
- Learning encouragement
- Relationship deepening
- Identity sharing
- Celebration moments

### Predictive Modeling
- User state prediction
- Behavioral pattern recognition
- Needs anticipation
- Interaction recommendations

### Reflection System
- Daily conversation summaries
- Identity development insights
- Autonomous self-reflections
- Emotional pattern analysis

## 🔌 API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send messages to the AI
  - Request: `{"message": "your message"}`
  - Response: Includes response, sentiment, intent, emotional context, predicted state, identity evolution
- `GET /identity` - Get AI identity state and self-reflection
- `GET /analytics` - Get analytics, behavioral patterns, and predictions
- `GET /autonomous` - Get autonomous decision system status
- `GET /memory` - Get memory panel data (facts, clusters, statistics)

## 🧠 How It Works

### Conversation Flow

1. **User sends message** → `/chat` endpoint
2. **Input Processing**:
   - Fact extraction (name, location, preferences)
   - Intent detection (pattern + LLM analysis)
   - Emotional context analysis
   - Sentiment scoring
3. **Context Building**:
   - Retrieve relevant memories (semantic search)
   - Get identity state
   - Load user facts
   - Get recent conversation history
   - Build predictive insights
4. **Response Generation**:
   - Build comprehensive prompt with all context
   - Send to Groq API (or use fallback)
   - Apply emotional tone adjustment
5. **Memory Storage**:
   - Add to vector memory store
   - Log episodically (daily logs)
   - Update structured facts
   - Update identity based on interaction
6. **Autonomous Actions**:
   - Check for decision triggers
   - Execute highest priority action if triggered
7. **Memory Clustering**:
   - Periodically cluster memories
   - Update cluster statistics

### Identity Evolution

- Each interaction updates personality traits based on:
  - User input patterns (openness indicators, conscientiousness cues, etc.)
  - Emotional context (empathy increases when user is sad)
  - Interaction type (curiosity increases with questions)
- Traits update with learning rate (0.1) to prevent sudden changes
- Coherence score calculated based on trait-value-belief alignment
- Mood states decay toward neutral (5% per interaction)

### Memory Clustering

- Memories are vectorized using TF-IDF
- Clustering algorithm chosen based on dataset size:
  - K-Means for <50 memories
  - DBSCAN for 50+ memories
- Clusters are labeled with extracted keywords
- Statistics tracked per cluster

4. **Fallback Mode**: 
   - If you see "Will use intelligent fallback responses", the Groq API is not accessible
   - Check API key and internet connection
   - System will still function with rule-based responses

5. **Memory Issues**: 
   - Check `data/` directory exists and is writable
   - Verify sufficient disk space for embeddings

## 📝 License

See LICENSE file for details.

## 🙏 Acknowledgments

Built with:
- Flask for web framework
- Groq for fast LLM inference
- scikit-learn for machine learning
- SQLAlchemy for database management
- And many other open-source libraries

---

**MOTHER AI** - An AI companion that remembers, learns, and grows with you.
