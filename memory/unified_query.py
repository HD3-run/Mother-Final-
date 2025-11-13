"""
Unified Memory Query System - MOTHER's Thinking Process

This module implements a unified memory retrieval system that queries
all memory layers simultaneously, like human cognition accessing different
brain regions. It synthesizes results from structured memory, vector
memory, episodic memory, and reflections to provide comprehensive answers.
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from memory.structured_store import search_facts, get_fact, all_facts, get_knowledge_graph
from memory.vector_store import search_memory, get_recent_memories
from memory.episodic_logger import get_log_for_date, get_conversation_history, get_all_conversation_dates
from reflection.reflection_engine import reflection_engine


def think(query: str, query_type: str = "auto") -> Dict[str, Any]:
    """
    MOTHER's unified thinking process - queries ALL memory layers simultaneously
    
    This mimics human cognition where the brain accesses multiple memory systems
    (semantic, episodic, autobiographical) to answer questions.
    
    Args:
        query: The user's question or statement
        query_type: Type of query ("auto", "fact", "date", "memory", "reflection")
    
    Returns:
        {
            "answer": str,  # Synthesized answer from all sources
            "confidence": float,  # Confidence score (0-1)
            "sources": {
                "structured": List[Dict],  # Facts found
                "vector": List[Dict],       # Semantic matches
                "episodic": List[Dict],     # Chronological events
                "reflections": List[Dict]  # Self-reflections
            },
            "reasoning": str,  # How MOTHER arrived at answer
            "should_use_llm": bool  # Whether to fall back to LLM
        }
    """
    try:
        logging.info(f"[🧠] MOTHER thinking about: {query}")
        
        # Detect query type if auto
        if query_type == "auto":
            query_type = _detect_query_type(query)
        
        # Query ALL memory layers in parallel (like brain regions)
        structured_results = _query_structured_memory(query)
        graph_results = _query_knowledge_graph(query)
        vector_results = _query_vector_memory(query)
        episodic_results = _query_episodic_memory(query)
        reflection_results = _query_reflections(query)
        
        # Synthesize results (like brain combining information)
        synthesized = _synthesize_memories(
            query,
            query_type,
            structured_results,
            graph_results,
            vector_results,
            episodic_results,
            reflection_results
        )
        
        logging.info(f"[🧠] Thinking complete - Confidence: {synthesized['confidence']:.2f}")
        return synthesized
        
    except Exception as e:
        logging.error(f"[ERROR] Thinking process failed: {e}")
        return {
            "answer": None,
            "confidence": 0.0,
            "sources": {},
            "reasoning": f"Error in thinking process: {str(e)}",
            "should_use_llm": True
        }


def _detect_query_type(query: str) -> str:
    """Detect what type of query this is"""
    query_lower = query.lower()
    
    # Check if this is a STATEMENT (providing info) vs QUESTION (asking for info)
    # Statements: "my name is...", "I have...", "my cat's name is..."
    # Questions: "what is my name?", "do you remember...", "tell me about..."
    is_statement = any(pattern in query_lower for pattern in [
        'my name is', 'i am', 'i have', 'my cat', 'my dog', 'my pet',
        'i live in', 'i work as', 'i am from', 'my age is', 'i am',
        'my hobby is', 'i like', 'i enjoy'
    ])
    
    # If it's a statement, don't treat it as a query - let fact extraction handle it
    if is_statement:
        return "statement"  # Special type that won't trigger query logic
    
    # Date queries - check for date-related keywords first
    date_keywords = ['date', 'dates', 'when', 'july', 'august', 'yesterday', 'last week', 'memory', 'memories']
    if any(word in query_lower for word in date_keywords):
        # Specific date pattern (e.g., "july 8th 2025")
        if re.search(r'\d{1,2}(?:st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)', query_lower):
            return "date"
        # Date list queries (e.g., "what dates", "all dates", "tell me dates", "dates you have")
        if any(phrase in query_lower for phrase in [
            'date', 'dates', 'when did we talk', 'what dates', 'which dates', 
            'all dates', 'list dates', 'tell me dates', 'dates you have',
            'dates in memory', 'every date', 'dates that', 'dates we'
        ]):
            return "date"
    
    # Fact queries (pets, name, location, etc.) - only if it's a QUESTION
    question_words = ['what', 'tell me', 'do you', 'can you', 'do i have', 'what is', 'what are']
    if any(q_word in query_lower for q_word in question_words):
        if any(word in query_lower for word in ['pet', 'name', 'location', 'age', 'job', 'hobby', 'interest']):
            return "fact"
    
    # Memory/reflection queries
    if any(word in query_lower for word in ['remember', 'recall', 'what did we', 'talk about', 'discuss']):
        return "memory"
    
    # Reflection queries
    if any(word in query_lower for word in ['reflect', 'insight', 'think about', 'realize']):
        return "reflection"
    
    return "general"


def _query_structured_memory(query: str) -> List[Dict[str, Any]]:
    """Query structured memory (facts) - like semantic/declarative memory"""
    try:
        results = search_facts(query, limit=10)
        
        # Also check for direct fact keys
        query_lower = query.lower()
        direct_facts = []
        
        # Common fact keys to check
        fact_keys = ['name', 'location', 'age', 'job', 'pets', 'hobby', 'interest', 'mood']
        for key in fact_keys:
            if key in query_lower:
                fact_value = get_fact(key)
                if fact_value:
                    direct_facts.append({
                        'key': key,
                        'value': fact_value,
                        'confidence': 1.0,
                        'source': 'structured',
                        'type': 'direct_fact'
                    })
        
        # Combine search results with direct facts
        all_results = direct_facts + results
        for result in all_results:
            result['source'] = 'structured'
            result['type'] = 'fact'
        
        return all_results[:10]  # Limit to top 10
        
    except Exception as e:
        logging.error(f"[ERROR] Structured memory query failed: {e}")
        return []


def _query_knowledge_graph(query: str) -> List[Dict[str, Any]]:
    """Query knowledge graph (relationships) - like semantic network memory"""
    try:
        graph = get_knowledge_graph()
        results = []
        query_lower = query.lower()
        
        # Check for system-related queries (MOTHER, system, tools, abilities, etc.)
        system_keywords = ["mother", "system", "tool", "ability", "capability", "technology", 
                          "framework", "library", "store data", "memory", "architecture"]
        is_system_query = any(keyword in query_lower for keyword in system_keywords)
        
        if is_system_query:
            # Query MOTHER node directly
            mother_node = graph.get_node_by_name("MOTHER")
            if mother_node:
                edges = graph.get_edges_from_node(mother_node.id)
                for edge in edges:
                    target_node = graph.get_node_by_id(edge.target)
                    if target_node:
                        results.append({
                            'subject': 'MOTHER',
                            'relation': edge.type,
                            'object': target_node.name,
                            'confidence': edge.weight,
                            'source': 'graph',
                            'type': 'system_knowledge',
                            'properties': edge.properties,
                        })
                
                # DEEPLY TRAVERSE: Get relationships from related system nodes (modules, memory systems, tools)
                for edge in edges[:50]:  # Increased to 50 for comprehensive coverage
                    target_node = graph.get_node_by_id(edge.target)
                    if target_node:
                        # Get relationships from this target node (2 levels deep)
                        target_edges = graph.get_edges_from_node(target_node.id)
                        for target_edge in target_edges[:10]:  # Increased to 10 per node
                            final_target = graph.get_node_by_id(target_edge.target)
                            if final_target:
                                results.append({
                                    'subject': target_node.name,
                                    'relation': target_edge.type,
                                    'object': final_target.name,
                                    'confidence': target_edge.weight * 0.8,
                                    'source': 'graph',
                                    'type': 'system_knowledge',
                                })
                                
                                # Go 3 levels deep for modules and functions
                                if target_node.type == "module" or target_node.type == "function":
                                    deeper_edges = graph.get_edges_from_node(final_target.id)
                                    for deeper_edge in deeper_edges[:5]:
                                        deepest_target = graph.get_node_by_id(deeper_edge.target)
                                        if deepest_target:
                                            results.append({
                                                'subject': final_target.name,
                                                'relation': deeper_edge.type,
                                                'object': deepest_target.name,
                                                'confidence': deeper_edge.weight * 0.6,
                                                'source': 'graph',
                                                'type': 'system_knowledge',
                                            })
                
                # Also search for specific technical terms in node names
                all_nodes = graph.get_all_node_names()
                technical_terms = ["function", "module", "memory", "store", "retrieve", "file", "data"]
                for term in technical_terms:
                    if term in query_lower:
                        for node_name in all_nodes:
                            if term in node_name.lower():
                                node = graph.get_node_by_name(node_name)
                                if node:
                                    node_edges = graph.get_edges_from_node(node.id)
                                    for node_edge in node_edges[:5]:
                                        node_target = graph.get_node_by_id(node_edge.target)
                                        if node_target:
                                            results.append({
                                                'subject': node_name,
                                                'relation': node_edge.type,
                                                'object': node_target.name,
                                                'confidence': node_edge.weight * 0.7,
                                                'source': 'graph',
                                                'type': 'system_knowledge',
                                            })
        
        # Check for user-related queries - ENHANCED to find ANYTHING
        # Check multiple possible user node names
        user_node_names = ["user", "i", "you"]
        user_nodes = []
        for name in user_node_names:
            node = graph.get_node_by_name(name)
            if node:
                user_nodes.append(node)
        
        if user_nodes:
            # Get ALL relationships from ALL user nodes (comprehensive search)
            all_user_edges = []
            for user_node in user_nodes:
                edges = graph.get_edges_from_node(user_node.id)
                all_user_edges.extend(edges)
            
            # If query contains specific keywords, filter by relation type
            relation_keywords = {
                "pet": "has_pet",
                "name": "has_name",
                "location": "lives_in",
                "age": "has_age",
                "job": "works_as",
                "hobby": "has_hobby",
                "game": "play",
                "play": "play",
                "like": "like",
                "love": "love",
                "enjoy": "enjoy",
            }
            
            # Check if query has specific keywords
            has_specific_keyword = any(keyword in query_lower for keyword in relation_keywords.keys())
            
            if has_specific_keyword:
                # Filter by specific relation types
                for keyword, relation_type in relation_keywords.items():
                    if keyword in query_lower:
                        matching_edges = [e for e in all_user_edges if e.type == relation_type]
                        for edge in matching_edges:
                            target_node = graph.get_node_by_id(edge.target)
                            if target_node:
                                results.append({
                                    'subject': next((n.name for n in user_nodes if n.id == edge.source), 'user'),
                                    'relation': edge.type,
                                    'object': target_node.name,
                                    'confidence': edge.weight,
                                    'source': 'graph',
                                    'type': 'relationship',
                                    'properties': edge.properties,
                                })
            else:
                # NO SPECIFIC KEYWORD - Return ALL relationships from user nodes (comprehensive)
                # This allows MOTHER to find ANYTHING stored about the user
                for edge in all_user_edges:
                    # Skip system/internal facts unless specifically asked
                    if edge.type not in ["is_a", "has_part_of_speech", "has_synonym", "has_hypernym"]:
                        target_node = graph.get_node_by_id(edge.target)
                        if target_node:
                            results.append({
                                'subject': next((n.name for n in user_nodes if n.id == edge.source), 'user'),
                                'relation': edge.type,
                                'object': target_node.name,
                                'confidence': edge.weight,
                                'source': 'graph',
                                'type': 'relationship',
                                'properties': edge.properties,
                            })
        
        # Also search for nodes by name (for general queries about ANYTHING)
        # This searches ALL nodes, not just user-related ones
        all_nodes = graph.get_all_node_names()
        query_words = query_lower.split()
        
        for node_name in all_nodes:
            node_name_lower = node_name.lower()
            # Match if any query word appears in node name, or node name appears in query
            if any(word in node_name_lower for word in query_words if len(word) > 2) or \
               any(node_name_lower in word for word in query_words if len(word) > 2) or \
               query_lower in node_name_lower or node_name_lower in query_lower:
                node = graph.get_node_by_name(node_name)
                if node:
                    # Get relationships (both outgoing and incoming)
                    outgoing = graph.get_edges_from_node(node.id)
                    incoming = graph.get_edges_to_node(node.id)
                    
                    # Add outgoing relationships
                    for edge in outgoing[:10]:  # Increased limit
                        target_node = graph.get_node_by_id(edge.target)
                        if target_node:
                            results.append({
                                'subject': node_name,
                                'relation': edge.type,
                                'object': target_node.name,
                                'confidence': edge.weight,
                                'source': 'graph',
                                'type': 'relationship',
                            })
                    
                    # Add incoming relationships (what relates TO this node)
                    for edge in incoming[:10]:  # Increased limit
                        source_node = graph.get_node_by_id(edge.source)
                        if source_node:
                            results.append({
                                'subject': source_node.name,
                                'relation': edge.type,
                                'object': node_name,
                                'confidence': edge.weight,
                                'source': 'graph',
                                'type': 'relationship',
                            })
        
        # Sort by confidence
        results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        return results[:20]  # Increased limit for system queries
        
    except Exception as e:
        logging.error(f"[ERROR] Knowledge graph query failed: {e}")
        return []


def _query_vector_memory(query: str) -> List[Dict[str, Any]]:
    """Query vector memory (semantic search) - like episodic memory with semantic access"""
    try:
        results = search_memory(query, limit=10)
        
        # Add source metadata
        for result in results:
            result['source'] = 'vector'
            result['type'] = 'semantic_memory'
            # Rename 'similarity' to 'confidence' for consistency
            if 'similarity' in result:
                result['confidence'] = result.pop('similarity')
        
        return results
        
    except Exception as e:
        logging.error(f"[ERROR] Vector memory query failed: {e}")
        return []


def _query_episodic_memory(query: str) -> List[Dict[str, Any]]:
    """Query episodic memory (daily logs) - like autobiographical memory"""
    try:
        results = []
        
        # Check if query mentions a specific date
        date_match = _extract_date_from_query(query)
        if date_match:
            # Get log for specific date (returns list of events)
            events = get_log_for_date(date_match)
            if events:
                results.append({
                    'date': date_match,
                    'content': events,  # This is already a list of events
                    'confidence': 0.9,
                    'source': 'episodic',
                    'type': 'daily_log'
                })
        
        # Also search recent conversation history
        recent_history = get_conversation_history(days=7)
        query_lower = query.lower()
        
        # Find relevant conversations
        for interaction in recent_history[-20:]:  # Last 20 interactions
            user_input = interaction.get('user_input', '').lower()
            ai_response = interaction.get('ai_response', '').lower()
            
            # Simple keyword matching
            if any(word in user_input or word in ai_response for word in query_lower.split()[:3]):
                results.append({
                    'timestamp': interaction.get('timestamp'),
                    'user_input': interaction.get('user_input'),
                    'ai_response': interaction.get('ai_response'),
                    'confidence': 0.6,
                    'source': 'episodic',
                    'type': 'conversation'
                })
        
        return results[:10]  # Limit to top 10
        
    except Exception as e:
        logging.error(f"[ERROR] Episodic memory query failed: {e}")
        return []


def _query_reflections(query: str) -> List[Dict[str, Any]]:
    """Query reflection system - like metacognitive memory"""
    try:
        results = []
        query_lower = query.lower()
        
        # Check if query is about reflections or insights
        if any(word in query_lower for word in ['reflect', 'insight', 'think', 'realize', 'understand']):
            # Get recent reflections if available
            # This would require reflection engine to have a search method
            # For now, return empty - can be enhanced later
            pass
        
        return results
        
    except Exception as e:
        logging.error(f"[ERROR] Reflection query failed: {e}")
        return []


def _extract_date_from_query(query: str) -> Optional[str]:
    """Extract date from query if mentioned"""
    try:
        query_lower = query.lower()
        
        # Relative dates
        if 'yesterday' in query_lower:
            yesterday = datetime.now() - timedelta(days=1)
            return yesterday.strftime("%Y-%m-%d")
        elif 'last week' in query_lower:
            last_week = datetime.now() - timedelta(days=7)
            return last_week.strftime("%Y-%m-%d")
        elif 'last month' in query_lower:
            last_month = datetime.now() - timedelta(days=30)
            return last_month.strftime("%Y-%m-%d")
        
        # Specific date patterns
        # Pattern: "july 8th 2025" or "july 8, 2025"
        date_pattern = r"(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*)?(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s*,?\s*)?(\d{4})?"
        match = re.search(date_pattern, query_lower)
        if match:
            day = int(match.group(1))
            month_name = match.group(2).capitalize()
            year = int(match.group(3)) if match.group(3) else datetime.now().year
            
            try:
                month_num = datetime.strptime(month_name, "%B").month
                date_str = f"{year}-{month_num:02d}-{day:02d}"
                return date_str
            except ValueError:
                pass
        
        # Pattern: "7/8/2025" or "7-8-2025"
        date_pattern2 = r"(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})"
        match = re.search(date_pattern2, query_lower)
        if match:
            month = int(match.group(1))
            day = int(match.group(2))
            year = int(match.group(3))
            try:
                date_str = f"{year}-{month:02d}-{day:02d}"
                return date_str
            except ValueError:
                pass
        
        return None
        
    except Exception as e:
        logging.error(f"[ERROR] Date extraction failed: {e}")
        return None


def _synthesize_memories(query: str, query_type: str,
                        structured_results: List[Dict],
                        graph_results: List[Dict],
                        vector_results: List[Dict],
                        episodic_results: List[Dict],
                        reflection_results: List[Dict]) -> Dict[str, Any]:
    """
    Synthesize results from all memory layers - like brain combining information
    
    This is the core "thinking" process that:
    1. Ranks results by relevance and confidence
    2. Resolves conflicts between sources
    3. Generates a coherent answer
    4. Calculates overall confidence
    """
    try:
        # Collect all sources
        sources = {
            "structured": structured_results,
            "graph": graph_results,
            "vector": vector_results,
            "episodic": episodic_results,
            "reflections": reflection_results
        }
        
        # Handle different query types
        if query_type == "statement":
            # For statements, don't try to answer - let fact extraction and LLM handle it
            return {
                "answer": None,
                "confidence": 0.0,
                "sources": sources,
                "reasoning": "This is a statement providing information, not a query",
                "should_use_llm": True
            }
        elif query_type == "fact":
            return _synthesize_fact_query(query, sources)
        elif query_type == "date":
            return _synthesize_date_query(query, sources)
        elif query_type == "memory":
            return _synthesize_memory_query(query, sources)
        else:
            return _synthesize_general_query(query, sources)
            
    except Exception as e:
        logging.error(f"[ERROR] Memory synthesis failed: {e}")
        return {
            "answer": None,
            "confidence": 0.0,
            "sources": sources,
            "reasoning": f"Synthesis error: {str(e)}",
            "should_use_llm": True
        }


def _synthesize_fact_query(query: str, sources: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Synthesize fact-based queries (pets, name, location, etc.)"""
    structured = sources.get("structured", [])
    graph = sources.get("graph", [])
    vector = sources.get("vector", [])
    episodic = sources.get("episodic", [])
    
    # Prioritize knowledge graph for relationship queries
    if graph:
        best_relation = graph[0]
        subject = best_relation.get('subject', '')
        relation = best_relation.get('relation', '')
        obj = best_relation.get('object', '')
        
        # Format answer based on relation type
        if relation == "has_pet":
            # Collect all pets
            pets = [r.get('object') for r in graph if r.get('relation') == 'has_pet']
            if pets:
                if len(pets) == 1:
                    answer = f"Your pet's name is {pets[0]}."
                else:
                    pets_str = ', '.join(pets[:-1]) + f', and {pets[-1]}'
                    answer = f"Your pets' names are {pets_str}."
                return {
                    "answer": answer,
                    "confidence": best_relation.get('confidence', 0.9),
                    "sources": sources,
                    "reasoning": "Found in knowledge graph (relationship query)",
                    "should_use_llm": False
                }
        elif relation in ["has_name", "lives_in", "has_age", "works_as"]:
            relation_display = {
                "has_name": "name",
                "lives_in": "location",
                "has_age": "age",
                "works_as": "job",
            }.get(relation, relation)
            return {
                "answer": f"Your {relation_display} is {obj}.",
                "confidence": best_relation.get('confidence', 0.9),
                "sources": sources,
                "reasoning": "Found in knowledge graph (relationship query)",
                "should_use_llm": False
            }
    
    # Fall back to structured memory for facts (highest confidence)
    if structured:
        best_fact = structured[0]  # Already sorted by confidence
        fact_key = best_fact.get('key', '')
        fact_value = best_fact.get('value')
        
        # Format answer based on fact type
        if fact_key == 'pets' and isinstance(fact_value, dict):
            return _format_pet_answer(fact_value, sources)
        elif fact_key in ['name', 'location', 'age', 'job']:
            return {
                "answer": f"Your {fact_key} is {fact_value}.",
                "confidence": best_fact.get('confidence', 0.9),
                "sources": sources,
                "reasoning": f"Found in structured memory (fact database)",
                "should_use_llm": False
            }
    
    # Fall back to vector memory if structured memory doesn't have it
    if vector and vector[0].get('confidence', 0) > 0.5:
        return {
            "answer": None,  # Let LLM handle it with context
            "confidence": 0.4,
            "sources": sources,
            "reasoning": "Found in conversation history but not in structured facts",
            "should_use_llm": True
        }
    
    # No information found
    return {
        "answer": None,
        "confidence": 0.0,
        "sources": sources,
        "reasoning": "No relevant information found in any memory layer",
        "should_use_llm": True
    }


def _format_pet_answer(pet_data: Dict, sources: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Format answer about pets"""
    if not isinstance(pet_data, dict) or not pet_data.get('has_pets'):
        return {
            "answer": "I don't have any information about your pets yet.",
            "confidence": 0.5,
            "sources": sources,
            "reasoning": "No pet information in structured memory",
            "should_use_llm": False
        }
    
    names = pet_data.get('names', [])
    types = pet_data.get('types', [])
    count = pet_data.get('count', len(names) if names else 0)
    
    if names:
        if len(names) == 1:
            answer = f"You have {count} pet{'s' if count > 1 else ''}. {'Their' if count > 1 else 'Its'} name is {names[0]}."
        elif len(names) == 2:
            answer = f"You have {count} pets. Their names are {names[0]} and {names[1]}."
        else:
            names_str = ', '.join(names[:-1]) + f', and {names[-1]}'
            answer = f"You have {count} pets. Their names are {names_str}."
        
        if types:
            types_str = ', '.join(set(types))
            answer += f" They are {types_str}."
    elif types:
        answer = f"You have {count} pet{'s' if count > 1 else ''}. {'They are' if count > 1 else 'It is'} {', '.join(set(types))}."
    else:
        answer = f"You have {count} pet{'s' if count > 1 else ''}."
    
    return {
        "answer": answer,
        "confidence": 1.0,  # High confidence for structured facts
        "sources": sources,
        "reasoning": "Retrieved from structured memory (facts database)",
        "should_use_llm": False
    }


def _synthesize_date_query(query: str, sources: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Synthesize date-based queries"""
    episodic = sources.get("episodic", [])
    structured = sources.get("structured", [])
    vector = sources.get("vector", [])
    
    query_lower = query.lower()
    
    # Check if asking for list of dates (various phrasings)
    date_list_phrases = [
        'what dates', 'which dates', 'all dates', 'list dates', 
        'tell me dates', 'dates you have', 'dates in memory',
        'every date', 'dates that', 'dates we', 'dates did we',
        'what dates did we talk', 'dates you remember', 'dates in your memory'
    ]
    
    if any(phrase in query_lower for phrase in date_list_phrases):
        exclude_today = any(word in query_lower for word in ['apart from today', 'excluding today', 'other than today'])
        dates = get_all_conversation_dates(exclude_today=exclude_today)
        
        if dates:
            formatted_dates = []
            for date_str in dates:
                try:
                    date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                    formatted_dates.append(date_obj.strftime("%B %d, %Y"))
                except:
                    formatted_dates.append(date_str)
            
            if len(formatted_dates) == 1:
                answer = f"We talked on {formatted_dates[0]}."
            elif len(formatted_dates) <= 5:
                answer = f"We've talked on {len(formatted_dates)} date{'s' if len(formatted_dates) > 1 else ''}: {', '.join(formatted_dates)}."
            else:
                first_few = ', '.join(formatted_dates[:3])
                last_few = ', '.join(formatted_dates[-2:])
                answer = f"We've talked on {len(formatted_dates)} dates. Some include: {first_few}... and more recently {last_few}."
            
            return {
                "answer": answer,
                "confidence": 0.95,
                "sources": sources,
                "reasoning": "Retrieved from episodic memory (conversation index)",
                "should_use_llm": False
            }
    
    # Check if asking about specific date
    if episodic:
        daily_log = episodic[0]
        if daily_log.get('type') == 'daily_log':
            date_str = daily_log.get('date')
            # get_log_for_date returns list of events directly
            events = daily_log.get('content', [])
            
            if events and isinstance(events, list) and len(events) > 0:
                # Format conversation summary
                formatted = _format_date_conversations(events)
                return {
                    "answer": formatted,
                    "confidence": 0.9,
                    "sources": sources,
                    "reasoning": f"Retrieved from episodic memory for {date_str}",
                    "should_use_llm": False
                }
    
    # No specific date found
    return {
        "answer": None,
        "confidence": 0.0,
        "sources": sources,
        "reasoning": "No conversation found for that date",
        "should_use_llm": True
    }


def _format_date_conversations(events: List[Dict]) -> str:
    """Format conversation events for a specific date"""
    if not events:
        return "I don't have any recorded conversations for that date."
    
    formatted_parts = []
    for i, event in enumerate(events[:5], 1):  # Limit to 5 events
        user_input = event.get('user_input', '')[:100]
        ai_response = event.get('ai_response', '')[:150]
        
        if user_input and ai_response:
            formatted_parts.append(f"Exchange {i}: You said '{user_input}...' and I responded about {ai_response[:80]}...")
    
    if len(events) > 5:
        formatted_parts.append(f"... and {len(events) - 5} more exchanges.")
    
    return "Here's what we talked about:\n" + "\n".join(formatted_parts)


def _synthesize_memory_query(query: str, sources: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Synthesize memory/recall queries"""
    vector = sources.get("vector", [])
    episodic = sources.get("episodic", [])
    structured = sources.get("structured", [])
    
    # Combine vector and episodic results
    all_memories = []
    
    # Add vector memories (semantic matches)
    for mem in vector[:5]:
        all_memories.append({
            'type': 'semantic',
            'content': f"You said: '{mem.get('user_input', '')[:100]}...'",
            'confidence': mem.get('confidence', 0.5),
            'timestamp': mem.get('timestamp')
        })
    
    # Add episodic memories (chronological)
    for mem in episodic[:5]:
        if mem.get('type') == 'conversation':
            all_memories.append({
                'type': 'episodic',
                'content': f"On {mem.get('timestamp', 'unknown date')}, you said: '{mem.get('user_input', '')[:100]}...'",
                'confidence': mem.get('confidence', 0.6),
                'timestamp': mem.get('timestamp')
            })
    
    if all_memories:
        # Sort by confidence
        all_memories.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Format answer
        memory_summaries = [m['content'] for m in all_memories[:3]]
        answer = "I remember:\n" + "\n".join(f"- {summary}" for summary in memory_summaries)
        
        return {
            "answer": answer,
            "confidence": 0.7,
            "sources": sources,
            "reasoning": f"Found {len(all_memories)} relevant memories across vector and episodic memory",
            "should_use_llm": False
        }
    
    return {
        "answer": None,
        "confidence": 0.0,
        "sources": sources,
        "reasoning": "No relevant memories found",
        "should_use_llm": True
    }


def _synthesize_general_query(query: str, sources: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Synthesize general queries that don't fit specific categories"""
    structured = sources.get("structured", [])
    graph = sources.get("graph", [])
    vector = sources.get("vector", [])
    episodic = sources.get("episodic", [])
    
    # Prioritize graph results, especially system knowledge
    if graph:
        system_knowledge = [r for r in graph if r.get('type') == 'system_knowledge']
        if system_knowledge:
            # Found system knowledge - provide to LLM with high confidence
            return {
                "answer": None,  # Let LLM format it
                "confidence": 0.85,
                "sources": sources,
                "reasoning": f"Found {len(system_knowledge)} system knowledge facts in knowledge graph",
                "should_use_llm": True
            }
        elif graph[0].get('confidence', 0) > 0.7:
            # High confidence graph result
            return {
                "answer": None,
                "confidence": 0.75,
                "sources": sources,
                "reasoning": "Found relevant knowledge graph relationships",
                "should_use_llm": True
            }
    
    # Check if we have high-confidence structured facts
    if structured and structured[0].get('confidence', 0) > 0.8:
        return {
            "answer": None,  # Let LLM format it
            "confidence": 0.6,
            "sources": sources,
            "reasoning": "Found relevant facts, but query needs LLM interpretation",
            "should_use_llm": True
        }
    
    # Check if we have good semantic matches
    if vector and vector[0].get('confidence', 0) > 0.6:
        return {
            "answer": None,
            "confidence": 0.5,
            "sources": sources,
            "reasoning": "Found relevant conversations, providing context to LLM",
            "should_use_llm": True
        }
    
    # No strong matches
    return {
        "answer": None,
        "confidence": 0.0,
        "sources": sources,
        "reasoning": "No strong matches in memory, relying on LLM",
        "should_use_llm": True
    }

