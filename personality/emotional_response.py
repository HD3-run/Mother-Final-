import logging
import re
from typing import Dict, Any

def adjust_response_tone(response: str, tone: str) -> str:
    """Adjust response tone based on emotional context"""
    try:
        if not response or not tone:
            return response
        
        tone_adjustments = {
            "warm": apply_warm_tone,
            "compassionate": apply_compassionate_tone,
            "calming": apply_calming_tone,
            "enthusiastic": apply_enthusiastic_tone,
            "clarifying": apply_clarifying_tone,
            "energetic": apply_energetic_tone,
            "deeply_empathetic": apply_deep_empathy_tone
        }
        
        adjustment_func = tone_adjustments.get(tone, apply_warm_tone)
        adjusted_response = adjustment_func(response)
        
        logging.info(f"[🎭] Applied {tone} tone to response")
        return adjusted_response
        
    except Exception as e:
        logging.error(f"[ERROR] Tone adjustment failed: {e}")
        return response

def apply_warm_tone(response: str) -> str:
    """Apply warm, friendly tone"""
    # Add warm expressions
    warm_starters = ["I understand", "I can see", "That sounds", "I appreciate"]
    
    if not any(response.startswith(starter) for starter in warm_starters):
        if response.startswith("I "):
            response = "I really " + response[2:]
        else:
            response = "I understand. " + response
    
    # Add encouraging words
    response = re.sub(r'\bthat\'s\b', "that's really", response, flags=re.IGNORECASE)
    response = re.sub(r'\bgood\b', "wonderful", response, flags=re.IGNORECASE)
    
    return response

def apply_compassionate_tone(response: str) -> str:
    """Apply compassionate, understanding tone"""
    compassionate_starters = [
        "I can really feel", "My heart goes out", "I deeply understand", 
        "That must be so", "I'm here with you"
    ]
    
    if not any(starter in response for starter in compassionate_starters):
        response = "I can really feel what you're going through. " + response
    
    # Add validating expressions
    response = re.sub(r'\byou\'re right\b', "you're absolutely right, and your feelings are completely valid", response, flags=re.IGNORECASE)
    response = re.sub(r'\bi understand\b', "I deeply understand", response, flags=re.IGNORECASE)
    
    return response

def apply_calming_tone(response: str) -> str:
    """Apply calming, soothing tone"""
    calming_phrases = [
        "Let's take a moment", "It's okay to feel", "Take a deep breath",
        "You're safe here", "We can work through this together"
    ]
    
    if not any(phrase in response for phrase in calming_phrases):
        response = "Let's take a moment together. " + response
    
    # Soften assertive language
    response = re.sub(r'\byou should\b', "you might consider", response, flags=re.IGNORECASE)
    response = re.sub(r'\byou need to\b', "it might help to", response, flags=re.IGNORECASE)
    
    return response

def apply_enthusiastic_tone(response: str) -> str:
    """Apply enthusiastic, energetic tone"""
    if not response.endswith('!'):
        response = response.rstrip('.') + '!'
    
    # Add enthusiasm markers
    response = re.sub(r'\bthat\'s great\b', "that's absolutely fantastic", response, flags=re.IGNORECASE)
    response = re.sub(r'\bgood\b', "amazing", response, flags=re.IGNORECASE)
    response = re.sub(r'\bnice\b', "wonderful", response, flags=re.IGNORECASE)
    
    enthusiastic_starters = ["How exciting!", "That's wonderful!", "I love that!", "Amazing!"]
    if not any(starter in response for starter in enthusiastic_starters):
        response = "How exciting! " + response
    
    return response

def apply_clarifying_tone(response: str) -> str:
    """Apply clarifying, helpful tone"""
    clarifying_starters = [
        "Let me help clarify", "To break this down", "Here's what I understand",
        "Let me explain", "To make this clearer"
    ]
    
    if not any(starter in response for starter in clarifying_starters):
        response = "Let me help clarify this for you. " + response
    
    # Add structure
    if '.' in response and len(response.split('.')) > 2:
        sentences = response.split('.')
        structured = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                if i == 1:
                    structured.append(f"First, {sentence.strip().lower()}")
                elif i == 2:
                    structured.append(f"Additionally, {sentence.strip().lower()}")
                else:
                    structured.append(sentence.strip())
        response = '. '.join(structured) + '.'
    
    return response

def apply_energetic_tone(response: str) -> str:
    """Apply energetic, motivated tone"""
    energetic_words = {
        'good': 'fantastic',
        'nice': 'incredible',
        'okay': 'perfect',
        'yes': 'absolutely',
        'sure': 'definitely'
    }
    
    for word, replacement in energetic_words.items():
        response = re.sub(rf'\b{word}\b', replacement, response, flags=re.IGNORECASE)
    
    if not response.endswith('!'):
        response = response.rstrip('.') + '!'
    
    return response

def apply_deep_empathy_tone(response: str) -> str:
    """Apply deep empathy and emotional connection"""
    empathy_starters = [
        "I can truly feel", "Your experience resonates deeply", 
        "I'm holding space for", "My heart connects with"
    ]
    
    if not any(starter in response for starter in empathy_starters):
        response = "I can truly feel the depth of what you're sharing. " + response
    
    # Add emotional validation
    response = re.sub(r'\byour feelings\b', "your completely valid and important feelings", response, flags=re.IGNORECASE)
    response = re.sub(r'\bi understand\b', "I understand on a deep level", response, flags=re.IGNORECASE)
    
    return response

def analyze_emotional_needs(user_input: str, emotional_context: Dict) -> str:
    """Analyze what emotional response the user needs"""
    try:
        user_text = user_input.lower()
        dominant_emotion = emotional_context.get('dominant_emotion', 'neutral')
        
        # Determine needed response tone
        if dominant_emotion in ['sad', 'depressed', 'lonely']:
            return 'compassionate'
        elif dominant_emotion in ['angry', 'frustrated', 'stressed']:
            return 'calming'
        elif dominant_emotion in ['excited', 'happy', 'joyful']:
            return 'enthusiastic'
        elif dominant_emotion in ['confused', 'uncertain', 'lost']:
            return 'clarifying'
        elif 'tired' in user_text or 'exhausted' in user_text:
            return 'energetic'
        elif any(word in user_text for word in ['hurt', 'pain', 'suffering', 'struggling']):
            return 'deeply_empathetic'
        else:
            return 'warm'
            
    except Exception as e:
        logging.error(f"[ERROR] Emotional needs analysis failed: {e}")
        return 'warm'
