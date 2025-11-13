"""
Lexicon Seeder for MOTHER

This module seeds the lexicon with foundational English vocabulary,
ensuring common words are recognized and not treated as unknown.
"""

import logging
from typing import Dict

from memory.lexicon_manager import LexiconManager
from memory.knowledge_graph import KnowledgeGraph, ConceptNode

logger = logging.getLogger(__name__)

# Core English vocabulary - parts of speech and common words
CORE_VOCABULARY: Dict[str, str] = {
    # Parts of speech
    "noun": "concept",
    "verb": "concept",
    "adjective": "concept",
    "adverb": "concept",
    "pronoun": "concept",
    "preposition": "concept",
    "conjunction": "concept",
    "determiner": "concept",
    "article": "concept",
    
    # Articles
    "a": "article",
    "an": "article",
    "the": "article",
    
    # Common determiners
    "this": "determiner",
    "that": "determiner",
    "these": "determiner",
    "those": "determiner",
    "all": "determiner",
    "any": "determiner",
    "each": "determiner",
    "every": "determiner",
    "some": "determiner",
    "no": "determiner",
    "more": "determiner",
    "most": "determiner",
    "many": "determiner",
    "few": "determiner",
    "little": "determiner",
    "much": "determiner",
    
    # Common verbs (be, have, do)
    "is": "verb",
    "are": "verb",
    "was": "verb",
    "were": "verb",
    "be": "verb",
    "being": "verb",
    "been": "verb",
    "am": "verb",
    "has": "verb",
    "have": "verb",
    "had": "verb",
    "having": "verb",
    "do": "verb",
    "does": "verb",
    "did": "verb",
    "done": "verb",
    
    # Common action verbs
    "say": "verb",
    "says": "verb",
    "said": "verb",
    "go": "verb",
    "goes": "verb",
    "went": "verb",
    "gone": "verb",
    "get": "verb",
    "gets": "verb",
    "got": "verb",
    "make": "verb",
    "makes": "verb",
    "made": "verb",
    "know": "verb",
    "knows": "verb",
    "knew": "verb",
    "known": "verb",
    "think": "verb",
    "thinks": "verb",
    "thought": "verb",
    "see": "verb",
    "sees": "verb",
    "saw": "verb",
    "seen": "verb",
    "come": "verb",
    "comes": "verb",
    "came": "verb",
    "take": "verb",
    "takes": "verb",
    "took": "verb",
    "taken": "verb",
    "give": "verb",
    "gives": "verb",
    "gave": "verb",
    "given": "verb",
    "find": "verb",
    "finds": "verb",
    "found": "verb",
    "use": "verb",
    "uses": "verb",
    "used": "verb",
    "work": "verb",
    "works": "verb",
    "worked": "verb",
    "call": "verb",
    "calls": "verb",
    "called": "verb",
    "try": "verb",
    "tries": "verb",
    "tried": "verb",
    "ask": "verb",
    "asks": "verb",
    "asked": "verb",
    "need": "verb",
    "needs": "verb",
    "needed": "verb",
    "feel": "verb",
    "feels": "verb",
    "felt": "verb",
    "become": "verb",
    "becomes": "verb",
    "became": "verb",
    "leave": "verb",
    "leaves": "verb",
    "left": "verb",
    "put": "verb",
    "puts": "verb",
    "mean": "verb",
    "means": "verb",
    "meant": "verb",
    "keep": "verb",
    "keeps": "verb",
    "kept": "verb",
    "let": "verb",
    "lets": "verb",
    "begin": "verb",
    "begins": "verb",
    "began": "verb",
    "begun": "verb",
    "seem": "verb",
    "seems": "verb",
    "seemed": "verb",
    "help": "verb",
    "helps": "verb",
    "helped": "verb",
    "show": "verb",
    "shows": "verb",
    "showed": "verb",
    "shown": "verb",
    "hear": "verb",
    "hears": "verb",
    "heard": "verb",
    "run": "verb",
    "runs": "verb",
    "ran": "verb",
    "move": "verb",
    "moves": "verb",
    "moved": "verb",
    "live": "verb",
    "lives": "verb",
    "lived": "verb",
    "believe": "verb",
    "believes": "verb",
    "believed": "verb",
    "bring": "verb",
    "brings": "verb",
    "brought": "verb",
    "happen": "verb",
    "happens": "verb",
    "happened": "verb",
    "write": "verb",
    "writes": "verb",
    "wrote": "verb",
    "written": "verb",
    "provide": "verb",
    "provides": "verb",
    "provided": "verb",
    "store": "verb",
    "stores": "verb",
    "stored": "verb",
    "sit": "verb",
    "sits": "verb",
    "sat": "verb",
    "stand": "verb",
    "stands": "verb",
    "stood": "verb",
    "lose": "verb",
    "loses": "verb",
    "lost": "verb",
    "pay": "verb",
    "pays": "verb",
    "paid": "verb",
    "meet": "verb",
    "meets": "verb",
    "met": "verb",
    "tell": "verb",
    "tells": "verb",
    "told": "verb",
    "can": "verb",
    "could": "verb",
    "will": "verb",
    "would": "verb",
    "should": "verb",
    "may": "verb",
    "might": "verb",
    "must": "verb",
    
    # Pronouns
    "i": "pronoun",
    "you": "pronoun",
    "he": "pronoun",
    "she": "pronoun",
    "it": "pronoun",
    "we": "pronoun",
    "they": "pronoun",
    "me": "pronoun",
    "him": "pronoun",
    "her": "pronoun",
    "us": "pronoun",
    "them": "pronoun",
    "my": "pronoun",
    "your": "pronoun",
    "his": "pronoun",
    "its": "pronoun",
    "our": "pronoun",
    "their": "pronoun",
    "mine": "pronoun",
    "yours": "pronoun",
    "hers": "pronoun",
    "ours": "pronoun",
    "theirs": "pronoun",
    "myself": "pronoun",
    "yourself": "pronoun",
    "himself": "pronoun",
    "herself": "pronoun",
    "itself": "pronoun",
    "ourselves": "pronoun",
    "yourselves": "pronoun",
    "themselves": "pronoun",
    "who": "pronoun",
    "whom": "pronoun",
    "whose": "pronoun",
    "which": "pronoun",
    "what": "pronoun",
    
    # Prepositions
    "of": "preposition",
    "in": "preposition",
    "to": "preposition",
    "for": "preposition",
    "with": "preposition",
    "on": "preposition",
    "at": "preposition",
    "from": "preposition",
    "by": "preposition",
    "about": "preposition",
    "as": "preposition",
    "into": "preposition",
    "like": "preposition",
    "through": "preposition",
    "after": "preposition",
    "over": "preposition",
    "between": "preposition",
    "out": "preposition",
    "against": "preposition",
    "during": "preposition",
    "without": "preposition",
    "before": "preposition",
    "under": "preposition",
    "around": "preposition",
    "among": "preposition",
    "since": "preposition",
    "within": "preposition",
    "toward": "preposition",
    "upon": "preposition",
    "above": "preposition",
    "across": "preposition",
    "below": "preposition",
    "beneath": "preposition",
    "beside": "preposition",
    "beyond": "preposition",
    "except": "preposition",
    "than": "preposition",
    "until": "preposition",
    
    # Conjunctions
    "and": "conjunction",
    "but": "conjunction",
    "or": "conjunction",
    "so": "conjunction",
    "if": "conjunction",
    "while": "conjunction",
    "because": "conjunction",
    "nor": "conjunction",
    "yet": "conjunction",
    "although": "conjunction",
    "though": "conjunction",
    "unless": "conjunction",
    "whether": "conjunction",
    "once": "conjunction",
    "when": "conjunction",
    "where": "conjunction",
    "why": "conjunction",
    "how": "conjunction",
    
    # Common nouns (for better understanding)
    "ability": "noun",
    "abilities": "noun",
    "system": "noun",
    "tool": "noun",
    "tools": "noun",
    "data": "noun",
    "information": "noun",
    "memory": "noun",
    "knowledge": "noun",
    "fact": "noun",
    "facts": "noun",
    "thing": "noun",
    "things": "noun",
    "way": "noun",
    "ways": "noun",
    "time": "noun",
    "person": "noun",
    "people": "noun",
    "place": "noun",
    "places": "noun",
    "word": "noun",
    "words": "noun",
}


def seed_core_vocabulary(lexicon: LexiconManager, graph: KnowledgeGraph) -> int:
    """
    Seed the lexicon with foundational English vocabulary.
    
    This function teaches MOTHER the basic building blocks of English
    grammar. It populates the knowledge graph with nodes for common parts
    of speech, articles, verbs, prepositions, and conjunctions.
    
    This foundational knowledge is essential for:
    - The SymbolicParser to function correctly
    - The "Unknown Word" reflex to correctly identify truly new words
    - Better understanding of common language patterns
    
    Args:
        lexicon: The LexiconManager instance to use.
        graph: The KnowledgeGraph instance to store words in.
        
    Returns:
        The number of words successfully seeded.
    """
    logger.info("[Lexicon Seeder] Seeding core vocabulary...")
    
    seeded_count = 0
    
    for word, part_of_speech in CORE_VOCABULARY.items():
        try:
            # Check if word already exists
            if lexicon.word_is_known(word):
                continue
            
            # Add word node
            word_node = ConceptNode(word.lower(), node_type="word")
            word_node = graph.add_node(word_node)
            
            # Add part of speech node
            pos_node = graph.get_node_by_name(part_of_speech)
            if not pos_node:
                pos_node = ConceptNode(part_of_speech, node_type="concept")
                pos_node = graph.add_node(pos_node)
            
            # Create relationship: word --[is_a]--> part_of_speech
            if word_node and pos_node:
                graph.add_edge(
                    word_node,
                    pos_node,
                    "is_a",
                    weight=0.95,
                    properties={"provenance": "seed", "confidence": 0.95},
                )
                seeded_count += 1
                
        except Exception as e:
            logger.warning(f"[Lexicon Seeder] Failed to seed word '{word}': {e}")
            continue
    
    logger.info(f"[Lexicon Seeder] Seeded {seeded_count} words into lexicon.")
    return seeded_count


def seed_common_tech_words(lexicon: LexiconManager, graph: KnowledgeGraph) -> int:
    """
    Seed additional common technology-related words that MOTHER might encounter.
    
    Args:
        lexicon: The LexiconManager instance to use.
        graph: The KnowledgeGraph instance to store words in.
        
    Returns:
        The number of words successfully seeded.
    """
    tech_words = {
        "technology": "noun",
        "computer": "noun",
        "software": "noun",
        "hardware": "noun",
        "program": "noun",
        "programming": "noun",
        "code": "noun",
        "algorithm": "noun",
        "database": "noun",
        "server": "noun",
        "client": "noun",
        "network": "noun",
        "internet": "noun",
        "website": "noun",
        "application": "noun",
        "app": "noun",
        "api": "noun",
        "function": "noun",
        "method": "noun",
        "class": "noun",
        "object": "noun",
        "variable": "noun",
        "string": "noun",
        "number": "noun",
        "boolean": "noun",
        "array": "noun",
        "list": "noun",
        "dictionary": "noun",
        "file": "noun",
        "directory": "noun",
        "folder": "noun",
        "path": "noun",
        "url": "noun",
        "http": "noun",
        "https": "noun",
        "json": "noun",
        "xml": "noun",
        "html": "noun",
        "css": "noun",
        "javascript": "noun",
        "python": "noun",
        "java": "noun",
        "c": "noun",
        "c++": "noun",
        "ai": "noun",
        "artificial": "adjective",
        "intelligence": "noun",
        "machine": "noun",
        "learning": "noun",
        "neural": "adjective",
        "network": "noun",
        "model": "noun",
        "training": "noun",
        "data": "noun",
        "storage": "noun",
        "memory": "noun",
        "cache": "noun",
        "buffer": "noun",
    }
    
    logger.info("[Lexicon Seeder] Seeding common technology words...")
    seeded_count = 0
    
    for word, part_of_speech in tech_words.items():
        try:
            if lexicon.word_is_known(word):
                continue
            
            word_node = ConceptNode(word.lower(), node_type="word")
            word_node = graph.add_node(word_node)
            
            pos_node = graph.get_node_by_name(part_of_speech)
            if not pos_node:
                pos_node = ConceptNode(part_of_speech, node_type="concept")
                pos_node = graph.add_node(pos_node)
            
            if word_node and pos_node:
                graph.add_edge(
                    word_node,
                    pos_node,
                    "is_a",
                    weight=0.9,
                    properties={"provenance": "seed", "confidence": 0.9},
                )
                seeded_count += 1
                
        except Exception as e:
            logger.warning(f"[Lexicon Seeder] Failed to seed tech word '{word}': {e}")
            continue
    
    logger.info(f"[Lexicon Seeder] Seeded {seeded_count} technology words.")
    return seeded_count

