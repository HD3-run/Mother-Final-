"""
Web Search Utilities for MOTHER

Provides Wikipedia and DuckDuckGo search capabilities for autonomous learning.
"""

import logging
import re
import time
from typing import Optional, Tuple
from urllib.parse import quote

import requests

try:
    import wikipedia
    wikipedia.set_user_agent("MOTHER-Agent/1.0")
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

logger = logging.getLogger(__name__)


def get_fact_from_wikipedia(topic: str) -> Optional[Tuple[str, str]]:
    """Retrieve a simple fact from a Wikipedia article.
    
    Args:
        topic: The topic to search for.
        
    Returns:
        A tuple of (topic, fact_sentence) or None if not found.
    """
    if not WIKIPEDIA_AVAILABLE:
        logger.debug("[Web Search] Wikipedia library not available")
        return None
    
    logger.info(f"[Web Search]: Searching Wikipedia for '{topic}'...")
    try:
        search_results = wikipedia.search(topic, results=1)
        if not search_results:
            return None
        
        page = wikipedia.page(search_results[0], auto_suggest=False, redirect=True)
        if not (page and page.summary):
            return None
        
        # Get first sentence
        first_sentence = page.summary.split(". ")[0].strip()
        if not first_sentence.endswith("."):
            first_sentence += "."
        
        logger.info(f"  [Web Search]: Found Wikipedia fact: '{first_sentence[:100]}...'")
        return topic, first_sentence
        
    except wikipedia.exceptions.DisambiguationError as e:
        # Try first option
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False, redirect=True)
            if page and page.summary:
                first_sentence = page.summary.split(". ")[0].strip()
                if not first_sentence.endswith("."):
                    first_sentence += "."
                return topic, first_sentence
        except Exception:
            pass
        return None
    except Exception as e:
        logger.debug(f"[Web Search] Wikipedia error for '{topic}': {e}")
        return None


def get_fact_from_duckduckgo(topic: str) -> Optional[Tuple[str, str]]:
    """Retrieve a definition from DuckDuckGo's API.
    
    Args:
        topic: The topic to search for.
        
    Returns:
        A tuple of (topic, fact_sentence) or None if not found.
    """
    logger.info(f"[Web Search]: Searching DuckDuckGo for '{topic}'...")
    try:
        url = f"https://api.duckduckgo.com/?q={quote(topic)}&format=json&no_html=1"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        definition = data.get("AbstractText") or data.get("Definition")
        if definition:
            first_sentence = definition.split(". ")[0].strip()
            if not first_sentence.endswith("."):
                first_sentence += "."
            
            logger.info(f"  [Web Search]: Found DuckDuckGo fact: '{first_sentence[:100]}...'")
            return topic, first_sentence
            
    except Exception as e:
        logger.debug(f"[Web Search] DuckDuckGo error for '{topic}': {e}")
        return None
    
    return None


def get_search_result_count(query: str) -> Optional[int]:
    """Scrape DuckDuckGo to get an approximate search result count for a query.
    
    This serves as a heuristic for determining the "popularity" of a topic.
    
    Args:
        query: The search term to look up.
        
    Returns:
        An integer of the approximate number of search results, or None if failed.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
        }
        url = f"https://duckduckgo.com/html/?q={quote(query)}"
        
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        
        match = re.search(r"([0-9,]+) results", response.text)
        if match:
            count_str = match.group(1).replace(",", "")
            return int(count_str)
    except Exception:
        pass
    return None


def find_new_topic(
    rejected_topics: set,
    max_attempts: int = 5,
    min_popularity: int = 10000,
) -> Optional[str]:
    """Find a new, focused, and unknown topic using heuristic-driven search.
    
    This method guides MOTHER's curiosity by selecting broad subjects,
    finding related topics on Wikipedia, and filtering for high-quality candidates.
    
    Args:
        rejected_topics: Set of topics that have been rejected before.
        max_attempts: Maximum number of attempts to find a topic.
        min_popularity: Minimum search result count for a topic to be considered.
        
    Returns:
        A string name of a suitable new topic, or None if none were found.
    """
    if not WIKIPEDIA_AVAILABLE:
        logger.debug("[Web Search] Wikipedia not available for topic discovery")
        return None
    
    core_subjects = [
        "Physics",
        "Chemistry",
        "Biology",
        "Mathematics",
        "Computer science",
        "History",
        "Geography",
        "Art",
        "Music",
        "Literature",
        "Philosophy",
        "Economics",
        "Psychology",
        "Sociology",
        "Astronomy",
        "Geology",
        "Common household items",
        "Types of animals",
        "Types of plants",
    ]
    
    import random
    
    for i in range(max_attempts):
        logger.info(f"[Discovery]: Searching for new topic (Attempt {i + 1}/{max_attempts})...")
        try:
            subject = random.choice(core_subjects)
            logger.info(f"  [Discovery]: Exploring core subject: '{subject}'")
            
            related_topics = wikipedia.search(subject, results=10)
            if not related_topics:
                continue
            
            topic = random.choice(related_topics)
            
            # Reject meta-pages
            reject_keywords = ["list of", "timeline of", "index of", "outline of"]
            if any(keyword in topic.lower() for keyword in reject_keywords):
                logger.debug(f"  [Discovery]: Rejecting meta-page: '{topic}'")
                continue
            
            # Check popularity
            search_popularity = get_search_result_count(topic)
            if search_popularity is not None and search_popularity < min_popularity:
                logger.debug(
                    f"  [Discovery]: Rejecting obscure topic '{topic}' (popularity: {search_popularity})"
                )
                continue
            
            # Clean topic name
            clean_topic = topic.strip().lower()
            if clean_topic not in rejected_topics:
                logger.info(f"  [Discovery]: Found new topic: '{topic}'")
                return clean_topic
                
        except Exception as e:
            logger.debug(f"  [Discovery]: Error during topic finding: {e}")
            time.sleep(1)
    
    logger.warning(f"[Discovery]: Could not find suitable topic after {max_attempts} attempts")
    return None

