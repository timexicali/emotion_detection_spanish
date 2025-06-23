"""
Model loading utilities for Spanish emotion detection.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional, Tuple
from .labels import EMOTION_LABELS

# Recommended Spanish transformer models
SPANISH_MODELS = {
    'beto': 'dccuchile/bert-base-spanish-wwm-uncased',
    'robertuito': 'pysentimiento/robertuito-base-uncased',
    'roberta-bne': 'BSC-TeMU/roberta-base-bne',
    'beto-cased': 'dccuchile/bert-base-spanish-wwm-cased'
}

def get_device() -> str:
    """
    Get the best available device for inference.
    
    Returns:
        Device name ('cuda' or 'cpu')
    """
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_tokenizer_and_model(model_name_or_path: str, 
                            device: Optional[str] = None,
                            num_labels: Optional[int] = None) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    """
    Load tokenizer and model for Spanish emotion detection.
    
    Args:
        model_name_or_path: Hugging Face model name or local path
        device: Device to load model on ('cuda' or 'cpu')
        num_labels: Number of emotion labels (defaults to len(EMOTION_LABELS))
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    if device is None:
        device = get_device()
    
    if num_labels is None:
        num_labels = len(EMOTION_LABELS)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Load model
    if os.path.exists(model_name_or_path) and os.path.isdir(model_name_or_path):
        # Local fine-tuned model - read config to get actual number of labels
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name_or_path)
        actual_num_labels = getattr(config, 'num_labels', num_labels)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        )
    else:
        # Hugging Face model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            problem_type="multi_label_classification"
        )
    
    # Move model to device
    model.to(device)
    model.eval()
    
    return tokenizer, model, device

def load_spanish_model(model_key: str = 'beto',
                      device: Optional[str] = None) -> Tuple[AutoTokenizer, AutoModelForSequenceClassification, str]:
    """
    Load a pre-configured Spanish model.
    
    Args:
        model_key: Key for predefined Spanish models ('beto', 'robertuito', etc.)
        device: Device to load model on
        
    Returns:
        Tuple of (tokenizer, model, device)
    """
    if model_key not in SPANISH_MODELS:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(SPANISH_MODELS.keys())}")
    
    model_name = SPANISH_MODELS[model_key]
    return load_tokenizer_and_model(model_name, device)

def validate_model_for_spanish(tokenizer: AutoTokenizer) -> bool:
    """
    Validate that a model is suitable for Spanish text.
    
    Args:
        tokenizer: Model tokenizer to validate
        
    Returns:
        True if model appears suitable for Spanish
    """
    # Test Spanish-specific characters and words
    test_spanish = "¡Hola! ¿Cómo estás? Niño, corazón, año"
    
    try:
        tokens = tokenizer.tokenize(test_spanish)
        # If Spanish accents and characters are heavily fragmented, might not be ideal
        # This is a basic heuristic
        if len(tokens) > 20:  # Too many tokens for simple Spanish text
            return False
        return True
    except Exception:
        return False

def get_model_info(model_name_or_path: str) -> dict:
    """
    Get information about a model.
    
    Args:
        model_name_or_path: Model name or path
        
    Returns:
        Dictionary with model information
    """
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name_or_path)
        
        return {
            'model_type': config.model_type,
            'vocab_size': getattr(config, 'vocab_size', 'unknown'),
            'hidden_size': getattr(config, 'hidden_size', 'unknown'),
            'num_attention_heads': getattr(config, 'num_attention_heads', 'unknown'),
            'num_hidden_layers': getattr(config, 'num_hidden_layers', 'unknown'),
            'max_position_embeddings': getattr(config, 'max_position_embeddings', 'unknown')
        }
    except Exception as e:
        return {'error': str(e)}

def list_available_spanish_models() -> dict:
    """
    List all available pre-configured Spanish models.
    
    Returns:
        Dictionary mapping model keys to model names
    """
    return SPANISH_MODELS.copy() 