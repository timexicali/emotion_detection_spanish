"""
Main emotion detection class for Spanish text.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Union
import numpy as np
import os

from .model_loader import load_tokenizer_and_model, load_spanish_model
from .preprocessing import normalize_spanish_text
from .labels import EMOTION_LABELS, is_valid_emotion
from .utils import encode_emotions, decode_emotions

class EmotionDetector:
    """
    Spanish emotion detection using transformer models.
    
    This is the main class for detecting emotions in Spanish text.
    It supports both pre-trained and fine-tuned models.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_key: str = 'beto',
                 device: Optional[str] = None,
                 max_length: int = 512):
        """
        Initialize the Spanish emotion detector.
        
        Args:
            model_path: Path to fine-tuned model (if available)
            model_key: Key for pre-configured Spanish models ('beto', 'robertuito', etc.)
            device: Device to use ('cuda' or 'cpu'), auto-detected if None
            max_length: Maximum sequence length for tokenization
        """
        self.model_path = model_path
        self.model_key = model_key
        self.max_length = max_length
        
        # Load model and tokenizer
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.tokenizer, self.model, self.device = load_tokenizer_and_model(
                model_path, device
            )
            self._is_fine_tuned = True
        else:
            print(f"Loading base Spanish model: {model_key}")
            self.tokenizer, self.model, self.device = load_spanish_model(
                model_key, device
            )
            self._is_fine_tuned = False
            print("⚠️  Using base model. Fine-tune for better emotion detection.")
    
    def detect(self, 
               text: str, 
               threshold: float = 0.5,
               return_scores: bool = False,
               normalize_text: bool = True) -> Union[List[str], Dict[str, float]]:
        """
        Detect emotions in Spanish text.
        
        Args:
            text: Spanish text to analyze
            threshold: Confidence threshold for emotion prediction (0.0-1.0)
            return_scores: Whether to return confidence scores
            normalize_text: Whether to apply text normalization
            
        Returns:
            List of emotion labels or dict with confidence scores
        """
        # Normalize text if requested
        if normalize_text:
            processed_text = normalize_spanish_text(text)
        else:
            processed_text = text.strip()
        
        if not processed_text:
            return {"neutral": 1.0} if return_scores else ["neutral"]
        
        # Tokenize
        inputs = self.tokenizer(
            processed_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length
        ).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
            # Apply sigmoid for multi-label classification
            probabilities = torch.sigmoid(logits).cpu().numpy()[0]
        
        if return_scores:
            return {
                emotion: float(score) 
                for emotion, score in zip(EMOTION_LABELS, probabilities)
            }
        else:
            # Return emotions above threshold
            emotions = decode_emotions(probabilities, threshold)
            return emotions
    
    def detect_batch(self, 
                    texts: List[str], 
                    threshold: float = 0.5,
                    batch_size: int = 16,
                    normalize_text: bool = True) -> List[List[str]]:
        """
        Detect emotions for a batch of Spanish texts.
        
        Args:
            texts: List of Spanish texts to analyze
            threshold: Confidence threshold for emotion predictions
            batch_size: Batch size for processing
            normalize_text: Whether to apply text normalization
            
        Returns:
            List of emotion predictions for each text
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_results = []
            
            for text in batch_texts:
                emotions = self.detect(text, threshold, return_scores=False, 
                                     normalize_text=normalize_text)
                batch_results.append(emotions)
            
            results.extend(batch_results)
        
        return results
    
    def get_top_emotions(self, 
                        text: str, 
                        top_k: int = 3,
                        normalize_text: bool = True) -> List[tuple]:
        """
        Get top-k emotions with their confidence scores for Spanish text.
        
        Args:
            text: Spanish text to analyze
            top_k: Number of top emotions to return
            normalize_text: Whether to apply text normalization
            
        Returns:
            List of (emotion, score) tuples sorted by confidence
        """
        scores = self.detect(text, return_scores=True, normalize_text=normalize_text)
        
        # Sort by confidence score
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_emotions[:top_k]
    
    def analyze_emotion_distribution(self, 
                                   text: str,
                                   normalize_text: bool = True) -> Dict[str, float]:
        """
        Get the full emotion distribution for Spanish text.
        
        Args:
            text: Spanish text to analyze
            normalize_text: Whether to apply text normalization
            
        Returns:
            Dictionary with all emotion scores
        """
        return self.detect(text, return_scores=True, normalize_text=normalize_text)
    
    def is_fine_tuned(self) -> bool:
        """
        Check if the model is fine-tuned for emotion detection.
        
        Returns:
            True if using a fine-tuned model, False if using base model
        """
        return self._is_fine_tuned
    
    def get_supported_emotions(self) -> List[str]:
        """
        Get the list of supported Spanish emotions.
        
        Returns:
            List of Spanish emotion labels
        """
        return EMOTION_LABELS.copy()
    
    def validate_emotions(self, emotions: List[str]) -> bool:
        """
        Validate that emotion labels are supported.
        
        Args:
            emotions: List of emotion labels to validate
            
        Returns:
            True if all emotions are valid
        """
        return all(is_valid_emotion(emotion) for emotion in emotions)
    
    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_path': self.model_path or f"spanish_model:{self.model_key}",
            'device': self.device,
            'max_length': str(self.max_length),
            'fine_tuned': str(self._is_fine_tuned),
            'num_emotions': str(len(EMOTION_LABELS)),
            'emotions': ', '.join(EMOTION_LABELS)
        }

# Convenience function for quick emotion detection
def detect_emotions(text: str, 
                   model_key: str = 'beto',
                   threshold: float = 0.5,
                   normalize_text: bool = True) -> List[str]:
    """
    Quick emotion detection function for Spanish text.
    
    Args:
        text: Spanish text to analyze
        model_key: Spanish model to use ('beto', 'robertuito', etc.)
        threshold: Confidence threshold
        normalize_text: Whether to normalize text
        
    Returns:
        List of detected emotions
    """
    detector = EmotionDetector(model_key=model_key)
    return detector.detect(text, threshold, normalize_text=normalize_text) 