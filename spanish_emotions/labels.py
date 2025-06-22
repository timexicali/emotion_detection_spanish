"""
Spanish emotion labels for emotion detection.
"""

# Spanish emotion labels
EMOTION_LABELS = [
    "alegría",      # joy
    "tristeza",     # sadness
    "enojo",        # anger
    "miedo",        # fear
    "sorpresa",     # surprise
    "desagrado",    # disgust
    "neutral"       # neutral
]

# English to Spanish emotion mapping for dataset conversion
EMOTION_MAPPING = {
    'joy': 'alegría',
    'happiness': 'alegría',
    'happy': 'alegría',
    'sadness': 'tristeza', 
    'sad': 'tristeza',
    'sorrow': 'tristeza',
    'anger': 'enojo',
    'angry': 'enojo',
    'rage': 'enojo',
    'fear': 'miedo',
    'afraid': 'miedo',
    'scared': 'miedo',
    'surprise': 'sorpresa',
    'surprised': 'sorpresa',
    'amazed': 'sorpresa',
    'disgust': 'desagrado',
    'disgusted': 'desagrado',
    'repulsion': 'desagrado',
    'neutral': 'neutral'
}

def get_spanish_emotion(english_emotion: str) -> str:
    """
    Convert English emotion label to Spanish.
    
    Args:
        english_emotion: English emotion label
        
    Returns:
        Spanish emotion label or 'neutral' if not found
    """
    return EMOTION_MAPPING.get(english_emotion.lower(), 'neutral')

def is_valid_emotion(emotion: str) -> bool:
    """
    Check if an emotion label is valid.
    
    Args:
        emotion: Emotion label to check
        
    Returns:
        True if valid, False otherwise
    """
    return emotion in EMOTION_LABELS 