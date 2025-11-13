# MOTHER AI - Advanced AI Companion System

MOTHER is a sophisticated AI companion system that develops its own identity, forms long-term memories, learns autonomously, and builds genuine connections through advanced cognitive architectures. MOTHER uses a multi-layered memory system, knowledge graph, and autonomous learning cycles to create a truly intelligent companion experience.

## 🎯 What MOTHER Is

MOTHER is an AI companion that:

- **Remembers Everything**: Stores information in multiple memory systems (knowledge graph, vector memory, episodic memory, structured facts)
- **Thinks Like a Human**: Uses a unified thinking system that queries all memory layers simultaneously before answering
- **Learns Autonomously**: Discovers new topics, studies them, and refines knowledge during idle time
- **Develops Identity**: Forms and evolves personality traits, values, and beliefs through interactions
- **Understands Context**: Uses semantic search, emotional analysis, and predictive modeling to understand you
- **Acts Autonomously**: Makes proactive decisions to check in, offer support, or deepen relationships
- **Reflects on Herself**: Generates autonomous reflections on her own growth and experiences

## 🏗️ System Architecture

MOTHER's architecture consists of several interconnected cognitive systems:

```
┌─────────────────────────────────────────────────────────────┐
│                    MOTHER AI System                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │   Cognitive  │  │  Metacognitive│  │   Universal   │   │
│  │    Agent     │◄─┤    Engine    │◄─┤  Interpreter  │   │
│  └──────┬───────┘  └──────────────┘  └──────────────┘   │
│         │                                                  │
│         ▼                                                  │
│  ┌────────────────────────────────────────────────────┐   │
│  │         Unified Thinking System (think())          │   │
│  │  Queries all memory layers simultaneously         │   │
│  └────────────────────────────────────────────────────┘   │
│         │                                                  │
│         ├──────────┬──────────┬──────────┬──────────┐   │
│         ▼          ▼          ▼          ▼          ▼   │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ │
│  │Knowledge │ │  Vector  │ │ Episodic │ │Structured│ │
│  │  Graph   │ │  Memory  │ │  Memory  │ │  Facts   │ │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │  Autonomous │  │  Reflection │  │   Identity  │   │
│  │   Learning  │  │   Engine    │  │   Engine    │   │
│  └──────────────┘  └──────────────┘  └──────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Core Capabilities

MOTHER's capabilities are built on both original systems and new cognitive architectures:

### 1. Identity Formation Engine

**Purpose**: Develops and evolves MOTHER's personality through interactions

**How it works**:
- Tracks 7 personality dimensions (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism, Empathy, Curiosity)
- Evolves traits based on interaction patterns
- Maintains core values and beliefs that develop over time
- Calculates identity coherence scores
- Tracks mood states (energy, positivity, empathy, curiosity)

**Implementation**: `personality/identity_engine.py`
- Learning rate-based trait updates (0.1 learning rate)
- Coherence score calculation based on trait-value-belief alignment
- Mood state decay mechanism (5% decay toward neutral)
- Value strength thresholds (0.7) for adding new values

### 2. Multi-Layered Memory System

**Purpose**: Stores and retrieves information across multiple memory layers

**How it works**:
- **Vector Memory Store**: Semantic search using TF-IDF vectorization and cosine similarity
- **Structured Facts**: Extracts and stores user information (name, location, preferences)
- **Episodic Memory**: Daily conversation logs with emotional summaries
- **Semantic Clustering**: Groups related memories using K-Means and DBSCAN algorithms
- **Knowledge Graph**: Stores concepts and relationships as semantic network (NEW)

**Implementation**:
- `memory/vector_store.py`: TF-IDF vectorization with cosine similarity
- `memory/structured_store.py`: Facts storage (backward compatible)
- `memory/episodic_logger.py`: Daily conversation logs
- `memory/semantic_clustering.py`: Hybrid clustering (K-Means/DBSCAN)
- `memory/knowledge_graph.py`: NetworkX-based semantic network (NEW)

### 3. Predictive Modeling

**Purpose**: Predicts user emotional state and needs

**How it works**:
- Predicts user emotional state and needs
- Tracks behavioral patterns over time
- Uses machine learning models (Logistic Regression, Random Forest) for predictions
- Feature extraction from temporal, linguistic, emotional, and behavioral data

**Implementation**: `processing/predictive_modeling.py`
- Custom feature extraction:
  - Temporal features (time of day, day of week, interaction frequency)
  - Linguistic features (word count, question count, exclamation count)
  - Emotional features (sentiment, emotion scores, intensity)
  - Behavioral features (interaction patterns, topic preferences)
- Model training with train/test split
- Prediction caching (1-hour cache)
- Behavioral pattern tracking

### 4. Autonomous Decision System

**Purpose**: Makes proactive decisions to enhance relationship

**How it works**:
- 6 decision rules with priority-based execution:
  - Check-in reminders (after 48+ hours of inactivity)
  - Emotional support triggers (detects negative emotion patterns)
  - Learning encouragement (recognizes curiosity patterns)
  - Relationship deepening (when interaction frequency is high but knowledge is low)
  - Identity sharing (when coherence score is high)
  - Celebration triggers (detects positive achievements)
- Scheduled tasks: Daily reflections, periodic check-ins, model training, memory consolidation

**Implementation**: `reflection/autonomous_decision.py`
- Rule-based decision system with priority scoring
- Cooldown mechanism to prevent spam
- Context-aware decision making
- Action history tracking in database
- 6 custom decision rules with configurable thresholds

### 5. Reflection Engine

**Purpose**: Generates autonomous reflections on experiences

**How it works**:
- **Daily Reflections**: Summarizes each day's conversations
- **Identity Reflections**: Reflects on personal growth and development
- **Autonomous Reflections**: Generates independent insights
- **Emotional Pattern Analysis**: Tracks emotional trends over time

**Implementation**: `reflection/reflection_engine.py`
- Multi-type reflection generation
- Reflection scheduling (24-hour intervals)
- Emotional pattern analysis over 7-day windows
- Learning progression analysis

### 6. Emotional Intelligence

**Purpose**: Analyzes emotional context and responds with appropriate empathy

**How it works**:
- Custom sentiment analysis with negation handling
- Emotional context detection (work stress, relationships, health, achievements, loss)
- Intensity modifiers (very, extremely, slightly, etc.)
- Emotional progression tracking
- Needs assessment based on emotional state

**Implementation**: `utils/sentiment.py`
- Custom rule-based sentiment analysis
- Emotion lexicon with 7 base emotions
- Intensity modifier detection (very, extremely, slightly, etc.)
- Negation handling ("not happy" → negative)
- Emotional context detection (work, relationships, health, etc.)
- Emotional progression tracking over time
- Needs assessment algorithm

### 7. Context-Aware Response Generation

**Purpose**: Builds comprehensive context for generating personalized responses

**How it works**:
- Builds comprehensive context from:
  - Identity state and personality traits
  - Relevant past memories (semantic search)
  - Emotional context
  - Predicted user state
  - User facts and preferences
  - Recent conversation history
  - Intent-specific guidance
  - System self-awareness (MOTHER's knowledge of her own architecture) (NEW)

**Implementation**: `processing/context_builder.py`
- Intelligent context assembly from multiple sources
- Context truncation when exceeding max length (8000 chars)
- Priority-based context ordering
- Time-ago formatting for recent memories
- Theme extraction from interactions
- Technical question detection and specialized prompts (NEW)

## 🧠 How MOTHER Works

### Conversation Flow

When you talk to MOTHER, here's what happens:

1. **Input Reception** (`routes.py` → `chat()`)
   - Your message arrives at the `/chat` endpoint
   - System logs the interaction and tracks usage

2. **Cognitive Processing** (if Cognitive Agent enabled)
   - **Cognitive Agent** (`processing/cognitive_agent.py`) processes the input:
     - **Symbolic Parsing**: First tries deterministic parsing (`symbolic_parser.py`)
     - **LLM Interpretation**: If parsing fails, uses `universal_interpreter.py` with Groq API
     - **Intent Detection**: Classifies as question, statement, command, etc.
     - **Entity Extraction**: Identifies people, places, concepts mentioned
     - **Lexicon Check**: Checks for unknown words, learns from WordNet if needed
   - If Cognitive Agent succeeds and returns clear answer, uses that response
   - If Cognitive Agent fails or returns unclear response, falls back to standard flow

3. **Special Query Handling** (`routes.py` → `handle_special_queries()`)
   - Checks for special queries (pet queries, date queries, memory queries)
   - Uses Unified Thinking System (`memory/unified_query.py` → `think()`) to query all memory layers
   - Returns answer if confidence >= 0.7

4. **Standard Flow** (if Cognitive Agent didn't handle it)
   - **Fact Extraction** (`routes.py` → `extract_facts()`): Extracts facts from statements (skips questions)
   - **Intent Detection** (`processing/intent_detector.py`): Classifies intent
   - **Emotional Analysis** (`utils/sentiment.py`): Analyzes sentiment and emotional context
   - **Predictive Modeling** (`processing/predictive_modeling.py`): Predicts user state
   - **Context Building** (`processing/context_builder.py`): Assembles comprehensive context
   - **LLM Response** (`processing/llm_handler.py`): Generates response using Groq API
   - **Tone Adjustment** (`personality/emotional_response.py`): Applies emotional tone

5. **For Questions** (when processed by Cognitive Agent or Standard Flow)
   - **Unified Thinking** (`memory/unified_query.py` → `think()`)
     - Queries Knowledge Graph: Searches for relationships like `user --[has_pet]--> petname`
     - Queries Vector Memory: Semantic search for similar conversations
     - Queries Episodic Memory: Checks conversation logs for mentions
     - Queries Structured Facts: Looks in `facts.json` for stored data
     - Queries Reflections: Checks self-reflections about the user
   - **Synthesis**: Combines results from all memory layers
   - **Answer Generation**: Returns comprehensive answer with confidence score
   - **LLM Fallback**: If confidence < 0.7, uses LLM with graph insights as context

6. **For Statements** (when processed by Cognitive Agent)
   - **Fact Extraction**: Extracts subject-verb-object relationships
   - **Conflict Detection** (`memory/conflict_resolver.py`): Checks for contradictions
   - **Knowledge Storage** (`memory/knowledge_harvester.py`):
     - Stores in Knowledge Graph: `user --[has_pet]--> petname` with properties
     - Stores in Vector Memory: Creates semantic embedding
     - Stores in Episodic Memory: Logs conversation with timestamp
     - Stores in Structured Facts: Updates `facts.json` (backward compatibility)
   - **Lexicon Learning**: If unknown words detected, learns from WordNet
   - **Graph Persistence**: Saves `knowledge_graph.json` to disk

7. **Memory Storage** (always happens)
   - **Vector Memory** (`memory/vector_store.py`): Stores semantic embedding
   - **Episodic Memory** (`memory/episodic_logger.py`): Logs conversation with timestamp
   - **Identity Update** (`personality/identity_engine.py`): Updates personality traits
   - **Behavioral Patterns** (`processing/predictive_modeling.py`): Updates behavioral patterns

8. **Autonomous Actions** (`reflection/autonomous_decision.py`)
   - Checks for decision triggers
   - Executes highest priority action if triggered

9. **Memory Clustering** (`memory/semantic_clustering.py`)
   - Periodically clusters memories
   - Updates cluster statistics

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

## 📚 Core Systems Explained

### 1. Cognitive Agent (`processing/cognitive_agent.py`)

**Purpose**: Orchestrates MOTHER's entire cognitive process

**How it works**:
- Receives user input and routes it through the cognitive pipeline
- Coordinates between symbolic parsing, LLM interpretation, and memory systems
- Handles intent processing (questions vs statements)
- Manages learning cycles and knowledge acquisition
- Synthesizes responses from multiple sources

**Key Methods**:
- `chat(user_input)`: Main entry point for processing user messages
- `_process_intent()`: Routes to appropriate handler based on intent
- `_process_statement_for_learning()`: Stores facts in knowledge graph
- `_answer_question_about()`: Retrieves information from knowledge graph
- `_get_all_knowledge_graph_contents()`: Enumerates all stored knowledge

**Integration**: Works with Universal Interpreter, Knowledge Harvester, Lexicon Manager, and Learning Manager

### 2. Universal Interpreter (`processing/universal_interpreter.py`)

**Purpose**: Provides LLM-based language understanding for complex inputs

**How it works**:
- Uses Groq API to interpret user input when symbolic parsing fails
- Extracts structured data: intent, entities, relations
- Handles context resolution (pronoun handling, coreference)
- Synthesizes natural language responses from structured facts
- Caches interpretations for performance

**Key Methods**:
- `interpret(text)`: Converts natural language to structured interpretation
- `resolve_context()`: Handles pronouns and references
- `synthesize()`: Converts structured facts back to natural language
- `decompose()`: Breaks complex sentences into atomic facts

**Integration**: Used by Cognitive Agent when symbolic parsing is insufficient

### 3. Knowledge Graph (`memory/knowledge_graph.py`)

**Purpose**: Stores concepts and relationships as a semantic network

**How it works**:
- Uses NetworkX to maintain a directed graph of concepts
- Nodes represent entities (people, places, concepts, functions, modules)
- Edges represent relationships (has_pet, play, uses, has_function, etc.)
- Stores properties on edges (confidence, provenance, timestamps)
- Supports graph traversal, path finding, and relationship queries

**Key Methods**:
- `add_node()`: Creates a new concept node
- `add_edge()`: Creates a relationship between nodes
- `get_node_by_name()`: Retrieves node by name
- `get_edges_from_node()`: Gets all relationships from a node
- `get_edges_to_node()`: Gets all relationships to a node
- `has_edge_between_names()`: Checks if relationship exists

**Storage**: Serialized as JSON in `data/structured_memory/knowledge_graph.json`

### 4. Unified Query System (`memory/unified_query.py`)

**Purpose**: MOTHER's "thinking" function that queries all memory layers

**How it works**:
- `think(query)` function queries all memory systems in parallel:
  - **Knowledge Graph**: Relationship queries (e.g., "user --[has_pet]--> ?")
  - **Vector Memory**: Semantic search using TF-IDF embeddings
  - **Episodic Memory**: Conversation log searches by date/topic
  - **Structured Facts**: Direct fact lookups
  - **Reflections**: Self-reflection analysis
- Synthesizes results with confidence scores
- Returns comprehensive answer with reasoning

**Key Functions**:
- `think(query)`: Main thinking function
- `_query_knowledge_graph()`: Searches knowledge graph
- `_query_vector_memory()`: Semantic search
- `_query_episodic_memory()`: Conversation log search
- `_synthesize_memories()`: Combines results from all sources

**Integration**: Called by Cognitive Agent when answering questions

### 5. Knowledge Harvester (`memory/knowledge_harvester.py`)

**Purpose**: Extracts and stores knowledge from user input and autonomous learning

**How it works**:
- Takes user statements and extracts facts
- Verifies facts using LLM (`fact_verifier.py`)
- Decomposes complex sentences into atomic facts (`fact_decomposer.py`)
- Stores facts in knowledge graph with confidence scores
- Handles conflict detection and resolution
- Supports autonomous learning cycles (discovery, study, refinement)

**Key Methods**:
- `harvest_from_sentence()`: Extracts and stores facts from text
- `discover_cycle()`: Autonomous discovery of new topics
- `study_cycle()`: Deep study of discovered topics
- `refinement_cycle()`: Refines existing knowledge

**Integration**: Used by Cognitive Agent and Cycle Manager

### 6. Lexicon Manager (`memory/lexicon_manager.py`)

**Purpose**: Manages MOTHER's vocabulary and word knowledge

**How it works**:
- Tracks which words MOTHER knows
- Integrates with WordNet for dictionary definitions
- Stores word knowledge in knowledge graph (definitions, synonyms, hypernyms, parts of speech)
- Automatically learns unknown words from WordNet
- Promotes words from "observed" to "trusted" status

**Key Methods**:
- `word_is_known()`: Checks if word exists in lexicon
- `learn_word_from_wordnet()`: Automatically learns word from WordNet
- `add_word()`: Manually adds word to lexicon

**Integration**: Used by Cognitive Agent to check for unknown words

### 7. Learning Manager (`memory/learning_manager.py`)

**Purpose**: Manages learning goals and recursive sub-goaling

**How it works**:
- Maintains a queue of learning goals (topics to learn)
- Implements recursive sub-goaling: breaks complex topics into prerequisites
- Prioritizes learning goals based on importance
- Integrates with Knowledge Harvester for actual learning

**Key Methods**:
- `add_learning_goal()`: Adds topic to learn
- `get_next_goal()`: Retrieves highest priority goal
- `break_down_goal()`: Recursively decomposes complex topics

**Integration**: Used by Cognitive Agent and Cycle Manager

### 8. Metacognitive Engine (`processing/metacognitive_engine.py`)

**Purpose**: Analyzes MOTHER's own performance and suggests improvements

**How it works**:
- Tracks interaction success/failure rates
- Analyzes performance logs
- Identifies patterns in failures
- Suggests optimizations to cognitive processes
- Monitors system health

**Key Methods**:
- `record_interaction()`: Logs interaction outcome
- `analyze_performance()`: Analyzes performance patterns
- `suggest_improvements()`: Generates improvement suggestions

**Integration**: Called after each interaction to track performance

### 9. Cycle Manager (`processing/cycle_manager.py`)

**Purpose**: Orchestrates autonomous learning cycles during idle time

**How it works**:
- Monitors user activity (tracks last interaction time)
- When idle for `idle_threshold_minutes`, starts learning cycles:
  - **Discovery Phase**: Finds new topics to learn
  - **Study Phase**: Deeply studies discovered topics
  - **Refinement Phase**: Refines existing knowledge
  - **Metacognitive Phase**: Analyzes own performance
- Pauses cycles when user becomes active
- Uses APScheduler for background task scheduling

**Key Methods**:
- `update_user_interaction()`: Tracks user activity
- `_manage_phases()`: Orchestrates learning phases
- `start_cycles()`: Begins autonomous learning
- `pause_cycles()`: Pauses when user active

**Integration**: Works with Knowledge Harvester and Learning Manager

### 10. Vector Memory Store (`memory/vector_store.py`)

**Purpose**: Provides semantic search over conversation history

**How it works**:
- Uses TF-IDF vectorization to create embeddings of memories
- Stores embeddings as NumPy arrays for fast retrieval
- Uses cosine similarity to find semantically similar memories
- Automatically rebuilds embeddings when new memories added
- Supports emotion-based filtering

**Key Methods**:
- `add_memory()`: Stores new memory with embedding
- `search_memory()`: Semantic search for similar memories
- `get_recent_memories()`: Retrieves chronologically recent memories

**Storage**: 
- `data/vector_memory/memories.json`: Memory text
- `data/vector_memory/embeddings.npy`: TF-IDF embeddings
- `data/vector_memory/vectorizer.pkl`: TF-IDF vectorizer

### 11. Episodic Memory (`memory/episodic_logger.py`)

**Purpose**: Maintains daily conversation logs with emotional context

**How it works**:
- Logs each conversation with timestamp, sentiment, emotional context
- Organizes logs by date (YYYY-MM-DD.json format)
- Generates daily summaries
- Tracks conversation themes and patterns
- Provides date-based querying

**Key Methods**:
- `log_event()`: Logs a conversation event
- `get_today_log()`: Gets today's conversations
- `get_log_for_date()`: Gets conversations for specific date
- `get_all_conversation_dates()`: Lists all dates with conversations

**Storage**: `data/episodic_memory/YYYY-MM-DD.json` files

### 12. Structured Memory (`memory/structured_store.py`)

**Purpose**: Stores structured facts (backward compatible with old system)

**How it works**:
- Maintains JSON-based fact storage (`facts.json`)
- Also stores facts in knowledge graph (hybrid approach)
- Provides backward-compatible API (`get_fact()`, `set_fact()`)
- Manages knowledge graph initialization and persistence

**Key Methods**:
- `get_fact(key)`: Retrieves structured fact
- `set_fact(key, value)`: Stores structured fact
- `get_knowledge_graph()`: Returns knowledge graph instance
- `save_knowledge_graph()`: Persists graph to disk

**Storage**: 
- `data/structured_memory/facts.json`: Structured facts
- `data/structured_memory/knowledge_graph.json`: Knowledge graph

### 13. Symbolic Parser (`processing/symbolic_parser.py`)

**Purpose**: Deterministic, rule-based parsing for simple language

**How it works**:
- Uses regex patterns to extract subject-verb-object relationships
- Handles simple sentence structures deterministically
- Fast and reliable for common patterns
- Falls back to LLM interpreter for complex sentences

**Key Methods**:
- `parse_fact(sentence)`: Extracts fact from sentence
- Pattern matching for common structures

**Integration**: Used by Cognitive Agent as first parsing attempt

### 14. Context Builder (`processing/context_builder.py`)

**Purpose**: Builds comprehensive context for LLM prompts

**How it works**:
- Assembles context from multiple sources:
  - Identity state and personality traits
  - Relevant memories (semantic search)
  - User facts and preferences
  - Recent conversation history
  - Emotional context
  - Predicted user state
  - System self-awareness (MOTHER's knowledge of her own codebase)
- Detects technical questions and adjusts prompt accordingly
- Truncates context if exceeds max length

**Key Methods**:
- `build_prompt()`: Main context building function
- `_build_system_context()`: Builds MOTHER's self-awareness context
- `_is_technical_question()`: Detects technical queries

**Integration**: Used by routes.py and Cognitive Agent for LLM prompts

### 15. Identity Engine (`personality/identity_engine.py`)

**Purpose**: Develops and evolves MOTHER's personality

**How it works**:
- Tracks 7 personality dimensions (Big Five + Empathy + Curiosity)
- Evolves traits based on interaction patterns
- Maintains core values and beliefs
- Calculates identity coherence scores
- Tracks mood states (energy, positivity, empathy, curiosity)

**Key Methods**:
- `update_identity()`: Updates personality based on interaction
- `get_identity_state()`: Returns current identity state
- `reflect_on_identity()`: Generates identity reflections

**Storage**: `data/system_state.json`

### 16. Reflection Engine (`reflection/reflection_engine.py`)

**Purpose**: Generates autonomous reflections on experiences

**How it works**:
- **Daily Reflections**: Summarizes each day's conversations
- **Identity Reflections**: Reflects on personal growth
- **Autonomous Reflections**: Generates independent insights
- **Emotional Pattern Analysis**: Tracks emotional trends

**Key Methods**:
- `generate_daily_reflection()`: Creates daily summary
- `generate_identity_reflection()`: Reflects on identity development
- `generate_autonomous_reflection()`: Creates independent reflection

**Storage**: `data/reflections/daily/`, `data/reflections/identity/`, `data/reflections/autonomous/`

### 17. Autonomous Decision Engine (`reflection/autonomous_decision.py`)

**Purpose**: Makes proactive decisions to enhance relationship

**How it works**:
- 6 decision rules with priority scoring:
  - Check-in reminders (after 48+ hours inactivity)
  - Emotional support triggers (negative emotion patterns)
  - Learning encouragement (curiosity patterns)
  - Relationship deepening (high frequency, low knowledge)
  - Identity sharing (high coherence score)
  - Celebration triggers (positive achievements)
- Cooldown mechanism prevents spam
- Tracks action history in database

**Key Methods**:
- `make_autonomous_decision()`: Evaluates and executes decisions
- `get_pending_actions()`: Returns pending autonomous actions

**Storage**: SQLite database (`instance/motherx.db`)

### 18. System Knowledge Seeder (`memory/system_knowledge_seeder.py`)

**Purpose**: Seeds knowledge graph with MOTHER's self-awareness

**How it works**:
- Analyzes MOTHER's codebase using `codebase_analyzer.py`
- Extracts information about modules, functions, data flow
- Stores system knowledge in knowledge graph:
  - MOTHER's technologies (Flask, Groq, NetworkX, etc.)
  - Memory systems (vector, episodic, structured, knowledge graph)
  - Processing modules (cognitive agent, interpreter, etc.)
  - Data storage locations and formats
- Enables MOTHER to answer questions about herself

**Key Methods**:
- `seed_system_knowledge()`: Main seeding function
- Integrates with Codebase Analyzer for dynamic analysis

**Integration**: Called during system initialization

### 19. Codebase Analyzer (`memory/codebase_analyzer.py`)

**Purpose**: Analyzes MOTHER's codebase to extract architectural knowledge

**How it works**:
- Traverses Python files in the project
- Extracts module information, function definitions, class structures
- Identifies data flow and relationships between modules
- Provides structured data for system knowledge seeding

**Key Methods**:
- `analyze_codebase()`: Main analysis function
- Returns structured representation of codebase architecture

**Integration**: Used by System Knowledge Seeder

## 🔧 Custom Implementations

These are the original custom implementations that form MOTHER's foundation:

### 1. VectorMemoryStore (`memory/vector_store.py`)

**Purpose**: Provides semantic search over conversation history

**How it works**:
- Custom implementation of semantic memory search
- Uses TF-IDF vectorization with cosine similarity
- Stores embeddings as NumPy arrays for fast retrieval
- Automatic embedding rebuild on new memories
- Emotion-based memory filtering

**Key Features**:
- Fast semantic search using cosine similarity
- Automatic embedding management
- Emotion-aware filtering

### 2. SemanticClusteringEngine (`memory/semantic_clustering.py`)

**Purpose**: Groups related memories for better organization

**How it works**:
- Hybrid clustering approach:
  - K-Means for smaller datasets (<50 memories)
  - DBSCAN for larger datasets (50+ memories)
- Automatic cluster size determination
- Keyword extraction for cluster labeling
- Cluster statistics and metadata tracking

**Key Features**:
- Adaptive algorithm selection based on dataset size
- Automatic keyword extraction
- Cluster statistics tracking

### 3. IdentityEngine (`personality/identity_engine.py`)

**Purpose**: Develops and evolves MOTHER's personality

**How it works**:
- Custom personality trait evolution algorithm
- Learning rate-based trait updates (0.1 learning rate)
- Coherence score calculation based on trait-value-belief alignment
- Mood state decay mechanism (5% decay toward neutral)
- Value strength thresholds (0.7) for adding new values

**Key Features**:
- 7 personality dimensions tracking
- Gradual trait evolution (prevents sudden changes)
- Coherence scoring for identity consistency

### 4. AutonomousDecisionEngine (`reflection/autonomous_decision.py`)

**Purpose**: Makes proactive decisions to enhance relationship

**How it works**:
- Rule-based decision system with priority scoring
- Cooldown mechanism to prevent spam
- Context-aware decision making
- Action history tracking in database
- 6 custom decision rules with configurable thresholds

**Key Features**:
- Priority-based decision execution
- Cooldown management
- Database-backed action history

### 5. SentimentAnalyzer (`utils/sentiment.py`)

**Purpose**: Analyzes emotional context in user input

**How it works**:
- Custom rule-based sentiment analysis
- Emotion lexicon with 7 base emotions
- Intensity modifier detection (very, extremely, slightly, etc.)
- Negation handling ("not happy" → negative)
- Emotional context detection (work, relationships, health, etc.)
- Emotional progression tracking over time
- Needs assessment algorithm

**Key Features**:
- Rule-based (fast, deterministic)
- Handles complex emotional expressions
- Context-aware emotion detection

### 6. PredictiveModelingEngine (`processing/predictive_modeling.py`)

**Purpose**: Predicts user emotional state and needs

**How it works**:
- Custom feature extraction:
  - Temporal features (time of day, day of week, interaction frequency)
  - Linguistic features (word count, question count, exclamation count)
  - Emotional features (sentiment, emotion scores, intensity)
  - Behavioral features (interaction patterns, topic preferences)
- Model training with train/test split
- Prediction caching (1-hour cache)
- Behavioral pattern tracking

**Key Features**:
- Multi-dimensional feature extraction
- ML models (Logistic Regression, Random Forest)
- Prediction caching for performance

### 7. ContextBuilder (`processing/context_builder.py`)

**Purpose**: Builds comprehensive context for LLM prompts

**How it works**:
- Intelligent context assembly from multiple sources
- Context truncation when exceeding max length (8000 chars)
- Priority-based context ordering
- Time-ago formatting for recent memories
- Theme extraction from interactions
- Technical question detection and specialized prompts

**Key Features**:
- Multi-source context assembly
- Smart truncation
- Priority-based ordering

### 8. EpisodicLogger (`memory/episodic_logger.py`)

**Purpose**: Maintains daily conversation logs

**How it works**:
- Daily conversation logging with JSON structure
- Automatic daily summary generation
- Conversation theme extraction
- Emotional trend analysis
- Interaction statistics tracking

**Key Features**:
- Date-based organization (YYYY-MM-DD.json)
- Automatic summarization
- Theme and trend analysis

### 9. ReflectionEngine (`reflection/reflection_engine.py`)

**Purpose**: Generates autonomous reflections on experiences

**How it works**:
- Multi-type reflection generation:
  - Daily summaries
  - Identity development insights
  - Autonomous self-reflections
  - Learning progression analysis
- Reflection scheduling (24-hour intervals)
- Emotional pattern analysis over 7-day windows

**Key Features**:
- Multiple reflection types
- Scheduled generation
- Pattern analysis

### 10. IntentDetector (`processing/intent_detector.py`)

**Purpose**: Classifies user intent from input

**How it works**:
- Pattern-based intent detection
- LLM-enhanced intent analysis (optional)
- Urgency detection (high/medium/low)
- Emotional context integration
- Intent combination algorithm

**Key Features**:
- Fast pattern matching
- Optional LLM enhancement
- Urgency and emotion awareness

---

## 📁 Project Structure

```
Mother-Final-/
├── app.py                      # Flask application factory
├── main.py                     # Application entry point
├── config.json                 # Configuration file
├── requirements.txt            # Python dependencies
├── startup.py                  # System initialization
├── routes.py                   # Flask routes and chat endpoint
│
├── memory/                     # Memory system modules
│   ├── knowledge_graph.py      # NetworkX-based knowledge graph
│   ├── unified_query.py        # Unified thinking system (think())
│   ├── knowledge_harvester.py  # Fact extraction and storage
│   ├── lexicon_manager.py      # Vocabulary and word knowledge
│   ├── learning_manager.py     # Learning goals and sub-goaling
│   ├── structured_store.py     # Structured facts + knowledge graph
│   ├── vector_store.py         # Semantic memory search (TF-IDF)
│   ├── episodic_logger.py      # Daily conversation logs
│   ├── semantic_clustering.py  # Memory clustering (K-Means/DBSCAN)
│   ├── conflict_resolver.py    # Conflict detection and resolution
│   ├── fact_verifier.py        # LLM-based fact verification
│   ├── fact_decomposer.py      # Complex sentence decomposition
│   ├── system_knowledge_seeder.py  # Seeds MOTHER's self-awareness
│   ├── codebase_analyzer.py    # Analyzes MOTHER's codebase
│   └── lexicon_seeder.py       # Seeds core vocabulary
│
├── processing/                 # Core processing modules
│   ├── cognitive_agent.py      # Main cognitive orchestrator
│   ├── universal_interpreter.py # LLM-based language understanding
│   ├── symbolic_parser.py      # Rule-based parsing
│   ├── context_builder.py      # Context assembly for LLM
│   ├── llm_handler.py          # Groq API integration
│   ├── intent_detector.py      # Intent classification
│   ├── predictive_modeling.py  # User state prediction
│   ├── metacognitive_engine.py # Self-analysis and optimization
│   └── cycle_manager.py        # Autonomous learning cycles
│
├── personality/                # Identity & personality system
│   ├── identity_engine.py      # Personality trait evolution
│   ├── emotional_response.py   # Tone adjustment
│   └── loader.py               # Config loader
│
├── reflection/                 # Reflection & autonomous systems
│   ├── reflection_engine.py   # Reflection generation
│   └── autonomous_decision.py # Autonomous decision making
│
├── utils/                      # Utility modules
│   ├── sentiment.py            # Sentiment analysis
│   ├── ml_utils.py             # ML utilities
│   ├── logger.py               # Logging setup
│   ├── usage_tracker.py        # Usage statistics
│   ├── terminal_logger.py      # Terminal output capture
│   ├── dictionary_utils.py     # WordNet integration
│   └── web_search.py           # Wikipedia/DuckDuckGo search
│
├── templates/                  # HTML templates
│   └── mother.html
│
├── static/                     # Static assets
│   ├── css/
│   └── js/
│
├── data/                       # Data storage
│   ├── episodic_memory/        # Daily logs (YYYY-MM-DD.json)
│   ├── vector_memory/          # TF-IDF embeddings (embeddings.npy, memories.json, vectorizer.pkl)
│   ├── structured_memory/      # Facts (facts.json) + knowledge_graph.json
│   ├── reflections/           # Reflection storage
│   │   ├── daily/             # Daily reflections
│   │   ├── identity/          # Identity reflections
│   │   └── autonomous/         # Autonomous reflections
│   ├── models/                # Trained ML models
│   ├── terminal_logs/         # Terminal output logs (timestamped .txt files)
│   ├── usage_tracking/        # Usage statistics (interaction_log.jsonl)
│   ├── journal/               # Journal entries
│   └── system_state.json      # System state
│
├── visualizations/            # Knowledge graph visualization
│   └── mother_knowledge_graph.html
│
└── instance/                  # Database instance
    └── motherx.db
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Groq API key (get one free at https://console.groq.com/keys)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Mother-Final-
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup environment variables**
   Create a `.env` file in the root directory:
   ```bash
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Initialize system** (optional, but recommended for first-time setup)
   ```bash
   python startup.py
   ```
   This will:
   - Create all necessary data directories (`data/`, `logs/`, `instance/`, etc.)
   - Initialize the knowledge graph
   - Seed system knowledge (MOTHER's self-awareness about her own architecture)
   - Initialize identity engine, ML models, semantic clustering, episodic logging
   - Create system state file
   
   **Note**: Lexicon seeding happens automatically when Cognitive Agent is first initialized (during first conversation), not in `startup.py`.

5. **Run the application**
   ```bash
   python main.py
   ```
   
   **What happens when you run `main.py`**:
   - Terminal logging is set up first (captures all output to `data/terminal_logs/`)
   - Flask app is created via `create_app()` factory function
   - System initialization runs (`startup.py` → `initialize_system()`):
     - Creates all data directories
     - Initializes knowledge graph
     - Seeds system knowledge (MOTHER's self-awareness)
     - Initializes identity engine, ML models, semantic clustering, episodic logging
   - Autonomous decision scheduler starts
   - Autonomous learning cycles start (if enabled)
   - Flask server starts on `http://localhost:5000`
   
   **Note**: All terminal output is automatically logged to timestamped `.txt` files in `data/terminal_logs/` while still appearing in your terminal.

6. **Access the application**
   Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

## ⚙️ Configuration

### config.json Settings

Edit `config.json` to customize MOTHER:

**Model Settings**:
- `model`: Groq model name (default: `qwen/qwen3-32b`)
- `temperature`: Response creativity (0.0-1.0)
- `max_tokens`: Maximum response length

**Feature Toggles**:
- `enable_cognitive_agent`: Enable cognitive agent (default: true)
- `enable_metacognitive_engine`: Enable metacognitive analysis (default: true)
- `enable_autonomous_learning`: Enable autonomous learning cycles (default: true)
- `show_full_thinking`: Show MOTHER's reasoning process (default: true)

**Autonomous Learning**:
- `idle_threshold_minutes`: Minutes of inactivity before learning starts (default: 5)
- `learning_phase_duration_minutes`: Duration of study phase (default: 30)
- `discovery_phase_duration_minutes`: Duration of discovery phase (default: 10)

**Personality Settings**:
- `emotional_tone`: Response tone (warm, professional, casual)
- `core_beliefs`: Array of core beliefs and values

## 🎨 Features

### Memory Systems
- **Knowledge Graph**: Stores concepts and relationships as semantic network
- **Vector Memory**: Semantic search using TF-IDF embeddings
- **Episodic Memory**: Daily conversation logs with emotional context
- **Structured Facts**: Backward-compatible fact storage

### Cognitive Capabilities
- **Unified Thinking**: Queries all memory layers simultaneously
- **Symbolic Parsing**: Fast, deterministic parsing for simple sentences
- **LLM Interpretation**: Complex language understanding via Groq API
- **Context Resolution**: Handles pronouns and references
- **Fact Extraction**: Extracts facts from statements
- **Question Answering**: Answers using all memory systems

### Learning Systems
- **Autonomous Learning**: Discovers and studies topics during idle time
- **WordNet Integration**: Automatically learns word definitions
- **Recursive Sub-goaling**: Breaks complex topics into prerequisites
- **Knowledge Refinement**: Improves existing knowledge over time

### Identity & Personality
- **Personality Evolution**: Develops traits through interactions
- **Value Formation**: Builds core values and beliefs
- **Mood Tracking**: Monitors energy, positivity, empathy, curiosity
- **Identity Coherence**: Calculates consistency of identity

### Autonomous Behavior
- **Proactive Check-ins**: Reaches out after inactivity
- **Emotional Support**: Detects and responds to negative emotions
- **Learning Encouragement**: Supports curiosity patterns
- **Relationship Deepening**: Actively learns more about you

### Reflection System
- **Daily Reflections**: Summarizes each day's conversations
- **Identity Reflections**: Reflects on personal growth
- **Autonomous Reflections**: Generates independent insights
- **Emotional Pattern Analysis**: Tracks emotional trends

## 🔌 API Endpoints

- `GET /` - Main chat interface
- `POST /chat` - Send messages to MOTHER
  - Request: `{"message": "your message"}`
  - Response: Includes response, sentiment, intent, emotional context, predicted state
- `GET /identity` - Get MOTHER's identity state and self-reflection
- `GET /analytics` - Get analytics, behavioral patterns, and predictions
- `GET /autonomous` - Get autonomous decision system status
- `GET /memory` - Get memory panel data (facts, clusters, statistics)

## 🛠️ Technologies Used

- **Flask**: Web framework
- **Groq API**: Fast LLM inference (Qwen, Llama, Mixtral, Gemma models)
- **NetworkX**: Knowledge graph implementation
- **scikit-learn**: TF-IDF vectorization, clustering, predictive modeling
- **NumPy**: Numerical computing for embeddings
- **NLTK/WordNet**: Dictionary and word knowledge
- **APScheduler**: Background task scheduling
- **SQLAlchemy**: Database ORM
- **SQLite**: Database storage
- **Pyvis**: Interactive graph visualization

## 📊 Knowledge Graph Visualization

To visualize MOTHER's knowledge graph:

1. **Generate visualization**:
   ```bash
   python visualize_knowledge_graph.py
   ```

2. **Open in browser**:
   ```
   visualizations/mother_knowledge_graph.html
   ```

The visualization shows:
- **Nodes**: Concepts, entities, functions, modules
- **Edges**: Relationships between nodes
- **Colors**: Different node types (user data, system knowledge, concepts)
- **Interactive**: Click and drag nodes, zoom, pan

## 📝 Data Storage

MOTHER stores data in the `data/` directory:

- `structured_memory/knowledge_graph.json`: Knowledge graph (NetworkX format)
- `structured_memory/facts.json`: Structured facts (backward compatible)
- `vector_memory/embeddings.npy`: TF-IDF embeddings
- `vector_memory/memories.json`: Memory text
- `episodic_memory/YYYY-MM-DD.json`: Daily conversation logs
- `reflections/`: Reflection storage (daily, identity, autonomous)
- `terminal_logs/`: Terminal output logs

## 🧪 Testing MOTHER

### Test Memory Storage
Tell MOTHER something: "I have a cat named something"
Then ask: "What are my pets?"
MOTHER should retrieve and answer using knowledge graph.

### Test Knowledge Graph
Ask: "Tell me about everything you have in your knowledge graph"
MOTHER will enumerate all stored nodes and relationships.

### Test Autonomous Learning
Leave MOTHER idle for 5+ minutes (configured in `idle_threshold_minutes`)
MOTHER will start autonomous learning cycles.

### Test Self-Awareness
Ask: "How are you made? What tools do you use?"
MOTHER will explain her own architecture using system knowledge.

## 🔄 How Systems Work Together

MOTHER's systems are designed to work seamlessly together:

1. **When you talk to MOTHER**:
   - Cognitive Agent (if enabled) orchestrates the entire process
   - If Cognitive Agent is disabled, Standard Flow handles everything
   - Both paths use the same memory systems and thinking process

2. **Memory Systems** (all active simultaneously):
   - **Knowledge Graph**: Stores relationships and concepts (NEW)
   - **Vector Memory**: Semantic search for similar conversations
   - **Episodic Memory**: Daily conversation logs
   - **Structured Facts**: Backward-compatible fact storage
   - All are queried simultaneously by `think()` function

3. **Learning Systems**:
   - **Immediate Learning**: When you tell MOTHER something, it's stored immediately
   - **Autonomous Learning**: During idle time, MOTHER discovers and studies new topics
   - **Word Learning**: Unknown words are learned from WordNet automatically

4. **Identity & Personality**:
   - Every interaction updates personality traits
   - Values and beliefs evolve over time
   - Mood states track emotional patterns

5. **Autonomous Behavior**:
   - Background scheduler runs autonomous tasks
   - Decision engine makes proactive choices
   - Reflection engine generates insights

## 📋 Key Points

- **All original systems are still active**: Vector Memory, Episodic Memory, Identity Engine, Predictive Modeling, etc. all work as before
- **New systems enhance old ones**: Knowledge Graph adds relationship storage, Cognitive Agent orchestrates everything, Unified Query queries all layers
- **Dual processing paths**: Cognitive Agent path (new) and Standard Flow path (original) both work
- **Unified memory query**: `think()` function queries all memory layers simultaneously
- **Autonomous learning**: Happens in background during idle time
- **Self-awareness**: MOTHER knows her own architecture and can answer questions about herself

## 🙏 Acknowledgments

Built with:
- Flask for web framework
- Groq for fast LLM inference
- NetworkX for knowledge graph
- scikit-learn for machine learning
- NLTK/WordNet for word knowledge
- Pyvis for graph visualization
- APScheduler for background tasks
- And many other open-source libraries

---

**MOTHER AI** - An AI companion that remembers, learns, and grows with you.
