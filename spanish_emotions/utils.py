"""
Utility functions for Spanish emotion detection.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import os
from .labels import EMOTION_LABELS

def encode_emotions(emotions: List[str]) -> List[int]:
    """
    Convert Spanish emotion labels to binary encoding.
    
    Args:
        emotions: List of Spanish emotion labels
        
    Returns:
        Binary encoding for each emotion label
    """
    encoding = [0] * len(EMOTION_LABELS)
    for emotion in emotions:
        if emotion in EMOTION_LABELS:
            idx = EMOTION_LABELS.index(emotion)
            encoding[idx] = 1
    return encoding

def decode_emotions(encoding: List[float], threshold: float = 0.5) -> List[str]:
    """
    Convert binary encoding back to Spanish emotion labels.
    
    Args:
        encoding: Binary/probability encoding
        threshold: Threshold for positive prediction
        
    Returns:
        List of predicted Spanish emotion labels
    """
    emotions = []
    for i, score in enumerate(encoding):
        if score >= threshold:
            emotions.append(EMOTION_LABELS[i])
    return emotions if emotions else ["neutral"]

def prepare_dataset_for_training(df: pd.DataFrame,
                                text_column: str = 'text',
                                emotion_column: str = 'emotions') -> pd.DataFrame:
    """
    Prepare a Spanish emotion dataset for training.
    
    Args:
        df: DataFrame with text and emotion columns
        text_column: Name of the text column
        emotion_column: Name of the emotion column
        
    Returns:
        DataFrame ready for training
    """
    from .preprocessing import clean_spanish_dataset
    
    # Clean the dataset
    df_clean = clean_spanish_dataset(df, text_column, emotion_column)
    
    # Parse emotions if they're strings
    if isinstance(df_clean[emotion_column].iloc[0], str):
        df_clean[emotion_column] = df_clean[emotion_column].apply(
            lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [x]
        )
    
    # Add binary encodings
    df_clean['emotion_encodings'] = df_clean[emotion_column].apply(encode_emotions)
    
    return df_clean

def save_predictions(predictions: List[Dict], 
                    output_path: str,
                    include_metadata: bool = True) -> None:
    """
    Save emotion predictions to a JSON file.
    
    Args:
        predictions: List of prediction dictionaries
        output_path: Path to save the predictions
        include_metadata: Whether to include metadata
    """
    output_data = {
        'predictions': predictions
    }
    
    if include_metadata:
        output_data['metadata'] = {
            'num_predictions': len(predictions),
            'emotion_labels': EMOTION_LABELS,
            'model_info': 'Spanish emotion detection'
        }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

def load_predictions(input_path: str) -> List[Dict]:
    """
    Load emotion predictions from a JSON file.
    
    Args:
        input_path: Path to the predictions file
        
    Returns:
        List of prediction dictionaries
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('predictions', data)

def convert_dataset_format(input_path: str, 
                          output_path: str,
                          input_format: str = 'csv',
                          output_format: str = 'json') -> None:
    """
    Convert between different dataset formats.
    
    Args:
        input_path: Path to input file
        output_path: Path to output file
        input_format: Input format ('csv' or 'json')
        output_format: Output format ('csv' or 'json')
    """
    # Load data
    if input_format == 'csv':
        df = pd.read_csv(input_path)
    elif input_format == 'json':
        df = pd.read_json(input_path)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    
    # Save data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if output_format == 'csv':
        df.to_csv(output_path, index=False)
    elif output_format == 'json':
        df.to_json(output_path, orient='records', force_ascii=False, indent=2)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

def validate_dataset(df: pd.DataFrame,
                    text_column: str = 'text',
                    emotion_column: str = 'emotions',
                    verbose: bool = True) -> Dict[str, Any]:
    """
    Validate a Spanish emotion dataset.
    
    Args:
        df: DataFrame to validate
        text_column: Name of the text column
        emotion_column: Name of the emotion column
        verbose: Whether to print validation details
        
    Returns:
        Dictionary with validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    # Check required columns
    if text_column not in df.columns:
        results['valid'] = False
        results['errors'].append(f"Missing text column: {text_column}")
    
    if emotion_column not in df.columns:
        results['valid'] = False
        results['errors'].append(f"Missing emotion column: {emotion_column}")
    
    if not results['valid']:
        return results
    
    # Check for empty texts
    empty_texts = df[df[text_column].isna() | (df[text_column].str.strip() == '')].shape[0]
    if empty_texts > 0:
        results['warnings'].append(f"Found {empty_texts} empty texts")
    
    # Check emotions
    invalid_emotions = []
    emotion_counts = {}
    
    for idx, emotions in enumerate(df[emotion_column]):
        if isinstance(emotions, str):
            try:
                emotions = eval(emotions)
            except:
                results['errors'].append(f"Invalid emotion format at row {idx}: {emotions}")
                continue
        
        if not isinstance(emotions, list):
            emotions = [emotions]
        
        for emotion in emotions:
            if emotion not in EMOTION_LABELS:
                invalid_emotions.append((idx, emotion))
            else:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    if invalid_emotions:
        results['valid'] = False
        results['errors'].append(f"Invalid emotions found: {invalid_emotions[:5]}...")  # Show first 5
    
    # Statistics
    results['stats'] = {
        'total_samples': len(df),
        'empty_texts': empty_texts,
        'unique_emotions': len(emotion_counts),
        'emotion_distribution': emotion_counts,
        'avg_text_length': df[text_column].str.len().mean() if not df[text_column].empty else 0
    }
    
    if verbose:
        print("Dataset Validation Results:")
        print(f"✅ Valid: {results['valid']}")
        print(f"📊 Total samples: {results['stats']['total_samples']}")
        print(f"📝 Average text length: {results['stats']['avg_text_length']:.1f} characters")
        print(f"😊 Emotions found: {list(emotion_counts.keys())}")
        
        if results['errors']:
            print("❌ Errors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if results['warnings']:
            print("⚠️  Warnings:")
            for warning in results['warnings']:
                print(f"  - {warning}")
    
    return results

def split_dataset(df: pd.DataFrame,
                 train_size: float = 0.8,
                 val_size: float = 0.1,
                 test_size: float = 0.1,
                 random_state: int = 42) -> tuple:
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df: DataFrame to split
        train_size: Proportion for training set
        val_size: Proportion for validation set
        test_size: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    # Validate split sizes
    if abs(train_size + val_size + test_size - 1.0) > 1e-6:
        raise ValueError("Split sizes must sum to 1.0")
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_size,
        random_state=random_state,
        stratify=df['emotions'].apply(str)  # Stratify by emotion combinations
    )
    
    # Second split: val vs test
    val_ratio = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio,
        random_state=random_state,
        stratify=temp_df['emotions'].apply(str)
    )
    
    return train_df, val_df, test_df

def calculate_emotion_statistics(df: pd.DataFrame,
                                emotion_column: str = 'emotions') -> Dict[str, Any]:
    """
    Calculate statistics for emotion distribution in dataset.
    
    Args:
        df: DataFrame with emotion data
        emotion_column: Name of the emotion column
        
    Returns:
        Dictionary with emotion statistics
    """
    stats = {
        'total_samples': len(df),
        'emotion_counts': {},
        'multi_label_samples': 0,
        'average_emotions_per_sample': 0,
        'emotion_co_occurrence': {}
    }
    
    all_emotions = []
    multi_label_count = 0
    
    for emotions in df[emotion_column]:
        if isinstance(emotions, str):
            try:
                emotions = eval(emotions)
            except:
                continue
        
        if not isinstance(emotions, list):
            emotions = [emotions]
        
        if len(emotions) > 1:
            multi_label_count += 1
        
        all_emotions.extend(emotions)
        
        # Track emotion counts
        for emotion in emotions:
            if emotion in EMOTION_LABELS:
                stats['emotion_counts'][emotion] = stats['emotion_counts'].get(emotion, 0) + 1
    
    stats['multi_label_samples'] = multi_label_count
    stats['average_emotions_per_sample'] = len(all_emotions) / len(df) if len(df) > 0 else 0
    
    return stats 