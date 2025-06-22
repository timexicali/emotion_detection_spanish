"""
Text preprocessing utilities for Spanish emotion detection.
"""

import re
import emoji
from unidecode import unidecode
from typing import List, Dict, Any
import pandas as pd

def preprocess_spanish_text(text: str, 
                           normalize_accents: bool = False,
                           remove_emojis: bool = False,
                           lowercase: bool = True,
                           remove_urls: bool = True,
                           remove_mentions: bool = True) -> str:
    """
    Preprocess Spanish text for emotion detection.
    
    Args:
        text: Input text to preprocess
        normalize_accents: Whether to remove accents (default: False to preserve Spanish)
        remove_emojis: Whether to remove emojis (default: False, they carry emotion info)
        lowercase: Whether to convert to lowercase
        remove_urls: Whether to remove URLs
        remove_mentions: Whether to remove @mentions
        
    Returns:
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove URLs
    if remove_urls:
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove mentions and hashtags (keep the text part)
    if remove_mentions:
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
    
    # Handle emojis
    if remove_emojis:
        text = emoji.demojize(text, language='es')
        text = re.sub(r':[a-zA-Z_]+:', '', text)
    else:
        # Convert emojis to text in Spanish
        text = emoji.demojize(text, language='es')
    
    # Remove excessive punctuation
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)
    text = re.sub(r'[.]{2,}', '...', text)
    
    # Normalize accents if requested (generally not recommended for Spanish)
    if normalize_accents:
        text = unidecode(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text.strip())
    
    return text

def normalize_spanish_text(text: str) -> str:
    """
    Standard normalization for Spanish text.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    return preprocess_spanish_text(
        text,
        normalize_accents=False,  # Preserve Spanish accents
        remove_emojis=False,      # Keep emotional content from emojis
        lowercase=True,
        remove_urls=True,
        remove_mentions=True
    )

def clean_spanish_dataset(df: pd.DataFrame, 
                         text_column: str = 'text',
                         emotion_column: str = 'emotions') -> pd.DataFrame:
    """
    Clean a Spanish emotion dataset.
    
    Args:
        df: DataFrame with text and emotion columns
        text_column: Name of the text column
        emotion_column: Name of the emotion column
        
    Returns:
        Cleaned DataFrame
    """
    # Create a copy
    df_clean = df.copy()
    
    # Preprocess texts
    df_clean[text_column] = df_clean[text_column].apply(normalize_spanish_text)
    
    # Remove empty texts
    df_clean = df_clean[df_clean[text_column].str.strip() != '']
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    
    return df_clean

def create_spanish_sample_dataset() -> pd.DataFrame:
    """
    Create a sample Spanish emotion dataset for testing and demonstration.
    
    Returns:
        DataFrame with sample Spanish text and emotion labels
    """
    sample_data = [
        {"text": "¡Estoy muy feliz por esta noticia!", "emotions": ["alegría"]},
        {"text": "Me siento muy triste por lo que pasó", "emotions": ["tristeza"]},
        {"text": "¡Qué asco me da esta situación!", "emotions": ["desagrado"]},
        {"text": "Tengo mucho miedo de lo que pueda pasar", "emotions": ["miedo"]},
        {"text": "¡No puedo creer lo que acaba de pasar!", "emotions": ["sorpresa"]},
        {"text": "Estoy muy enojado con esta decisión", "emotions": ["enojo"]},
        {"text": "El clima está normal hoy", "emotions": ["neutral"]},
        {"text": "¡Qué alegría me da verte! Aunque tengo un poco de miedo", "emotions": ["alegría", "miedo"]},
        {"text": "Esta película me da asco y tristeza", "emotions": ["desagrado", "tristeza"]},
        {"text": "¡Increíble! Me siento muy feliz", "emotions": ["sorpresa", "alegría"]},
        {"text": "Tengo tanto miedo que no puedo dormir", "emotions": ["miedo"]},
        {"text": "¡Qué sorpresa tan desagradable!", "emotions": ["sorpresa", "desagrado"]},
        {"text": "Me da mucha tristeza ver esto", "emotions": ["tristeza"]},
        {"text": "¡Estoy furioso con esta situación!", "emotions": ["enojo"]},
        {"text": "No siento nada especial sobre esto", "emotions": ["neutral"]},
        {"text": "¡Qué felicidad tan grande siento!", "emotions": ["alegría"]},
        {"text": "Esto me causa mucho disgusto", "emotions": ["desagrado"]},
        {"text": "¡Vaya sorpresa! No me lo esperaba", "emotions": ["sorpresa"]},
        {"text": "El miedo me paraliza completamente", "emotions": ["miedo"]},
        {"text": "Siento una mezcla de alegría y sorpresa", "emotions": ["alegría", "sorpresa"]},
    ]
    
    return pd.DataFrame(sample_data) 