"""
Spanish Emotion Detection Library

A lightweight, commercial-use-ready library for detecting emotions in Spanish text.
Uses transformer models (BETO/RoBERTuito) for multi-label emotion classification.
"""

__version__ = "0.1.0"
__author__ = "Daniel Gomez Morales"
__email__ = "daniel.gm78@emotionwise.ai"

# Import main classes and functions
from .detector import EmotionDetector, detect_emotions
from .labels import EMOTION_LABELS, get_spanish_emotion, is_valid_emotion
from .model_loader import (
    load_spanish_model, 
    load_tokenizer_and_model,
    list_available_spanish_models,
    get_device
)
from .preprocessing import (
    normalize_spanish_text,
    preprocess_spanish_text,
    create_spanish_sample_dataset,
    clean_spanish_dataset
)
from .utils import (
    encode_emotions,
    decode_emotions,
    prepare_dataset_for_training,
    validate_dataset,
    split_dataset,
    calculate_emotion_statistics
)

# Export main classes and functions
__all__ = [
    # Main detector class
    "EmotionDetector",
    "detect_emotions",
    
    # Labels and validation
    "EMOTION_LABELS",
    "get_spanish_emotion",
    "is_valid_emotion",
    
    # Model loading
    "load_spanish_model",
    "load_tokenizer_and_model",
    "list_available_spanish_models",
    "get_device",
    
    # Text preprocessing
    "normalize_spanish_text",
    "preprocess_spanish_text",
    "create_spanish_sample_dataset",
    "clean_spanish_dataset",
    
    # Utilities
    "encode_emotions",
    "decode_emotions",
    "prepare_dataset_for_training",
    "validate_dataset",
    "split_dataset",
    "calculate_emotion_statistics"
]

def get_version():
    """Get the library version."""
    return __version__

def get_supported_models():
    """Get list of supported Spanish models."""
    return list_available_spanish_models()

def get_supported_emotions():
    """Get list of supported Spanish emotions."""
    return EMOTION_LABELS.copy()

# Library info
def info():
    """Print library information."""
    print(f"Spanish Emotion Detection Library v{__version__}")
    print(f"Author: {__author__}")
    print(f"Email: {__email__}")
    print(f"Supported emotions: {', '.join(EMOTION_LABELS)}")
    print(f"Supported models: {', '.join(list_available_spanish_models().keys())}")
    print("Documentation: https://github.com/yourusername/emotion_detection_spanish") 