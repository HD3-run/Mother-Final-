"""
Dictionary Utilities for MOTHER

This module provides WordNet/NLTK integration for linguistic analysis,
allowing MOTHER to automatically learn word definitions, parts of speech,
and related concepts when encountering unknown words.
"""

from __future__ import annotations

import logging
from typing import Final, Literal, TypedDict, NotRequired

try:
    import nltk
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    
    # Try to use WordNet, download if needed
    try:
        wn.synsets("test")
    except LookupError:
        logging.info("[Dictionary Utils] NLTK 'wordnet' corpus not found. Downloading now...")
        nltk.download("wordnet", quiet=True)
        from nltk.corpus import wordnet as wn
    
    lemmatizer = WordNetLemmatizer()
    NLTK_AVAILABLE = True
except ImportError:
    logging.warning("[Dictionary Utils] NLTK not available. Install with: pip install nltk")
    NLTK_AVAILABLE = False
    wn = None
    lemmatizer = None

logger = logging.getLogger(__name__)

# Part of speech mapping from NLTK tags to our types
POS_MAP: Final = {
    "NN": "noun",
    "NNS": "noun",
    "NNP": "noun",
    "NNPS": "noun",
    "VB": "verb",
    "VBD": "verb",
    "VBG": "verb",
    "VBN": "verb",
    "VBP": "verb",
    "VBZ": "verb",
    "JJ": "adjective",
    "JJR": "adjective",
    "JJS": "adjective",
    "RB": "adverb",
    "RBR": "adverb",
    "RBS": "adverb",
    "PRP": "pronoun",
    "PRP$": "pronoun",
    "DT": "article",
    "IN": "preposition",
    "CC": "conjunction",
    "CD": "number",
    "FW": "foreign_word",
    "LS": "list_item",
    "MD": "modal",
    "POS": "possessive",
    "RP": "particle",
    "TO": "to",
    "UH": "interjection",
    "WDT": "wh_determiner",
    "WP": "wh_pronoun",
    "WP$": "possessive_wh_pronoun",
    "WRB": "wh_adverb",
    "EX": "existential",
    "PDT": "predeterminer",
    "SYM": "symbol",
}


class WordInfo(TypedDict):
    """Information about a word from WordNet"""
    type: Literal["concept", "noun", "verb", "adjective", "adverb"]
    definitions: list[str]
    hypernyms_raw: list[str]  # More general concepts (e.g., 'dog' -> 'canine')
    related_words: list[str]
    synonyms: NotRequired[list[str]]


def get_word_info_from_wordnet(word: str) -> WordInfo:
    """Retrieve detailed linguistic information for a word from WordNet.
    
    This function queries WordNet for a given word and attempts to find
    its most likely part of speech, definitions, and hypernyms (more
    general concepts, e.g., 'dog' -> 'canine'). It also extracts related
    words from the definitions and lemmas.
    
    It prioritizes nouns, then verbs, then adjectives when selecting the
    primary "synset" (sense of the word) to analyze.
    
    Args:
        word: The single word to look up.
        
    Returns:
        A WordInfo TypedDict containing the extracted type, definitions,
        hypernyms, and related words. Returns a default 'concept' type
        if the word is not found or NLTK is not available.
    """
    if not NLTK_AVAILABLE or not wn:
        logger.warning(f"[Dictionary Utils] WordNet not available for '{word}'")
        return WordInfo({
            "type": "concept",
            "definitions": [],
            "hypernyms_raw": [],
            "related_words": [],
        })
    
    word_info = WordInfo({
        "type": "concept",
        "definitions": [],
        "hypernyms_raw": [],
        "related_words": [],
        "synonyms": [],
    })
    
    try:
        synsets = wn.synsets(word.lower())
        
        if not synsets:
            logger.debug(f"[Dictionary Utils] No WordNet synsets found for '{word}'")
            return word_info
        
        # Prioritize: noun > verb > adjective > adverb > other
        best_synset = None
        
        for ss in synsets:
            if ss.pos() == "n":
                best_synset = ss
                break
        
        if not best_synset:
            for ss in synsets:
                if ss.pos() == "v":
                    best_synset = ss
                    break
        
        if not best_synset:
            for ss in synsets:
                if ss.pos() in ("a", "s"):  # adjective or satellite adjective
                    best_synset = ss
                    break
        
        if not best_synset:
            for ss in synsets:
                if ss.pos() == "r":  # adverb
                    best_synset = ss
                    break
        
        if not best_synset:
            best_synset = synsets[0]  # Fallback to first synset
        
        if best_synset:
            # Map NLTK POS to our types
            nltk_pos = best_synset.pos()
            if nltk_pos == "n":
                word_info["type"] = "noun"
            elif nltk_pos == "v":
                word_info["type"] = "verb"
            elif nltk_pos in ("a", "s"):
                word_info["type"] = "adjective"
            elif nltk_pos == "r":
                word_info["type"] = "adverb"
            
            # Get definition
            definition = best_synset.definition()
            if definition:
                word_info["definitions"].append(definition)
            
            # Get examples (can be useful as additional context)
            for example in best_synset.examples():
                if example and example not in word_info["definitions"]:
                    word_info["definitions"].append(example)
            
            # Get hypernyms (more general concepts)
            for hypernym_synset in best_synset.hypernyms():
                for lemma in hypernym_synset.lemmas():
                    hypernym_name = lemma.name().replace("_", " ")
                    if hypernym_name not in word_info["hypernyms_raw"]:
                        word_info["hypernyms_raw"].append(hypernym_name)
            
            # Get synonyms (lemmas from the same synset)
            for lemma in best_synset.lemmas():
                lemma_name = lemma.name().replace("_", " ")
                if lemma_name != word.lower() and lemma_name not in word_info.get("synonyms", []):
                    if "synonyms" not in word_info:
                        word_info["synonyms"] = []
                    word_info["synonyms"].append(lemma_name)
            
            # Extract related words from definitions
            for definition in word_info["definitions"]:
                tokens = definition.lower().split()
                for token_raw in tokens:
                    token = token_raw.strip(".,;?!\"'()[]{}")
                    if (
                        token.isalpha()
                        and len(token) > 2
                        and token != word.lower()
                        and token not in word_info["related_words"]
                    ):
                        word_info["related_words"].append(token)
            
            logger.debug(f"[Dictionary Utils] Retrieved WordNet info for '{word}': type={word_info['type']}, definitions={len(word_info['definitions'])}")
            
    except Exception as e:
        logger.error(f"[Dictionary Utils] Error getting WordNet info for '{word}': {e}")
    
    return word_info


def get_pos_tag_simple(word: str) -> str:
    """Determine the part of speech for a word using a fallback strategy.
    
    This function first attempts to use NLTK's fast `pos_tag` function.
    If the required NLTK resource is not downloaded, it gracefully falls
    back to querying WordNet for the word's primary part of speech.
    
    If both methods fail, it returns the generic type 'concept'.
    
    Args:
        word: The single word to tag.
        
    Returns:
        A string representing the determined part of speech (e.g., 'noun',
        'verb', 'concept').
    """
    if not NLTK_AVAILABLE:
        logger.warning(f"[Dictionary Utils] NLTK not available for POS tagging '{word}'")
        return "concept"
    
    try:
        # Try NLTK's pos_tag first (faster)
        try:
            tagged_word = nltk.pos_tag([word])
            if tagged_word:
                return POS_MAP.get(tagged_word[0][1], "concept")
        except LookupError:
            logger.debug(f"[Dictionary Utils] NLTK pos_tagger resource missing for '{word}'. Falling back to WordNet.")
            # Fall through to WordNet fallback
        
        # Fallback to WordNet
        if wn:
            synsets = wn.synsets(word.lower())
            if synsets:
                best_synset = None
                for ss in synsets:
                    if ss.pos() == "n":
                        best_synset = ss
                        break
                    if ss.pos() == "v" and not best_synset:
                        best_synset = ss
                    elif ss.pos() in ("a", "s") and not best_synset:
                        best_synset = ss
                
                if not best_synset and synsets:
                    best_synset = synsets[0]
                
                if best_synset:
                    nltk_pos = best_synset.pos()
                    if nltk_pos == "n":
                        return "noun"
                    if nltk_pos == "v":
                        return "verb"
                    if nltk_pos in ("a", "s"):
                        return "adjective"
                    if nltk_pos == "r":
                        return "adverb"
        
    except Exception as e:
        logger.error(f"[Dictionary Utils] Error in get_pos_tag_simple for '{word}': {e}")
    
    return "concept"


def lemmatize_word(word: str, pos: str | None = None) -> str:
    """Reduce a word to its base or dictionary form (lemma).
    
    Uses the WordNetLemmatizer to convert a word to its root form.
    For example, 'running' becomes 'run', and 'cats' becomes 'cat'.
    Providing the part of speech (POS) can improve accuracy.
    
    Args:
        word: The word to lemmatize.
        pos: An optional part-of-speech tag (e.g., 'n', 'v', 'a', 'r').
        
    Returns:
        The lemmatized form of the word as a string, or the original word
        if lemmatization is not available.
    """
    if not NLTK_AVAILABLE or not lemmatizer:
        logger.warning(f"[Dictionary Utils] Lemmatizer not available for '{word}'")
        return word
    
    try:
        if pos:
            wn_pos = None
            if pos.startswith("n"):
                wn_pos = wn.NOUN
            elif pos.startswith("v"):
                wn_pos = wn.VERB
            elif pos.startswith("a"):
                wn_pos = wn.ADJ
            elif pos.startswith("r"):
                wn_pos = wn.ADV
            
            if wn_pos:
                return lemmatizer.lemmatize(word, wn_pos)
        
        return lemmatizer.lemmatize(word)
        
    except Exception as e:
        logger.error(f"[Dictionary Utils] Error lemmatizing '{word}': {e}")
        return word

