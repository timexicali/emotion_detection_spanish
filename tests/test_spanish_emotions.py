"""
Tests for Spanish emotion detection library.
"""

import pytest
import sys
import os

# Add the parent directory to the path so we can import spanish_emotions
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import spanish_emotions
from spanish_emotions import (
    EmotionDetector, 
    EMOTION_LABELS,
    normalize_spanish_text,
    encode_emotions,
    decode_emotions,
    create_spanish_sample_dataset,
    validate_dataset,
    get_spanish_emotion,
    is_valid_emotion
)

class TestLabels:
    """Test emotion labels and validation."""
    
    def test_emotion_labels(self):
        """Test that emotion labels are properly defined."""
        expected_emotions = ["alegría", "tristeza", "enojo", "miedo", "sorpresa", "desagrado", "neutral"]
        assert EMOTION_LABELS == expected_emotions
    
    def test_is_valid_emotion(self):
        """Test emotion validation."""
        assert is_valid_emotion("alegría") == True
        assert is_valid_emotion("invalid_emotion") == False
        assert is_valid_emotion("happiness") == False  # English emotion
    
    def test_get_spanish_emotion(self):
        """Test English to Spanish emotion mapping."""
        assert get_spanish_emotion("joy") == "alegría"
        assert get_spanish_emotion("sadness") == "tristeza"
        assert get_spanish_emotion("invalid") == "neutral"

class TestPreprocessing:
    """Test text preprocessing functionality."""
    
    def test_normalize_spanish_text_basic(self):
        """Test basic Spanish text normalization."""
        text = "¡Hola! ¿Cómo estás?    "
        processed = normalize_spanish_text(text)
        assert processed == "¡hola! ¿cómo estás?"
    
    def test_normalize_spanish_text_urls(self):
        """Test URL removal."""
        text = "Visit https://example.com for more info"
        processed = normalize_spanish_text(text)
        assert "https://example.com" not in processed
    
    def test_normalize_spanish_text_mentions(self):
        """Test mention removal."""
        text = "Hola @usuario ¿cómo estás?"
        processed = normalize_spanish_text(text)
        assert "@usuario" not in processed
    
    def test_create_spanish_sample_dataset(self):
        """Test sample dataset creation."""
        df = create_spanish_sample_dataset()
        assert len(df) > 0
        assert 'text' in df.columns
        assert 'emotions' in df.columns
        
        # Check that all emotions in dataset are valid
        for emotions in df['emotions']:
            for emotion in emotions:
                assert emotion in EMOTION_LABELS

class TestUtils:
    """Test utility functions."""
    
    def test_encode_emotions(self):
        """Test emotion encoding."""
        emotions = ["alegría", "miedo"]
        encoded = encode_emotions(emotions)
        expected = [1, 0, 0, 1, 0, 0, 0]  # alegría at index 0, miedo at index 3
        assert encoded == expected
    
    def test_decode_emotions(self):
        """Test emotion decoding."""
        encoding = [0.8, 0.1, 0.2, 0.7, 0.3, 0.1, 0.2]
        decoded = decode_emotions(encoding, threshold=0.5)
        expected = ["alegría", "miedo"]  # Only scores >= 0.5
        assert decoded == expected
    
    def test_decode_emotions_neutral_fallback(self):
        """Test neutral fallback when no emotions exceed threshold."""
        encoding = [0.1, 0.2, 0.1, 0.3, 0.2, 0.1, 0.4]
        decoded = decode_emotions(encoding, threshold=0.5)
        assert decoded == ["neutral"]
    
    def test_validate_dataset(self):
        """Test dataset validation."""
        df = create_spanish_sample_dataset()
        results = validate_dataset(df, verbose=False)
        assert results['valid'] == True
        assert 'stats' in results
        assert results['stats']['total_samples'] > 0

class TestEmotionDetector:
    """Test emotion detection functionality."""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        # This will use the base model since no trained model exists yet
        return EmotionDetector(model_key='beto')
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly."""
        assert detector is not None
        assert detector.model is not None
        assert detector.tokenizer is not None
        assert detector.device in ['cuda', 'cpu']
    
    def test_detect_emotions_basic(self, detector):
        """Test basic emotion detection."""
        text = "¡Estoy muy feliz por esta noticia!"
        emotions = detector.detect(text)
        assert isinstance(emotions, list)
        assert len(emotions) > 0
        # All returned emotions should be valid
        for emotion in emotions:
            assert emotion in EMOTION_LABELS
    
    def test_detect_emotions_with_scores(self, detector):
        """Test emotion detection with confidence scores."""
        text = "¡Estoy muy feliz por esta noticia!"
        scores = detector.detect(text, return_scores=True)
        assert isinstance(scores, dict)
        assert len(scores) == len(EMOTION_LABELS)
        # All scores should be between 0 and 1
        for emotion, score in scores.items():
            assert 0 <= score <= 1
            assert emotion in EMOTION_LABELS
    
    def test_detect_empty_text(self, detector):
        """Test detection with empty text."""
        emotions = detector.detect("")
        assert emotions == ["neutral"]
    
    def test_detect_batch(self, detector):
        """Test batch detection."""
        texts = [
            "¡Estoy muy feliz!",
            "Me siento triste",
            "Tengo miedo"
        ]
        batch_results = detector.detect_batch(texts)
        assert len(batch_results) == len(texts)
        for emotions in batch_results:
            assert isinstance(emotions, list)
            for emotion in emotions:
                assert emotion in EMOTION_LABELS
    
    def test_get_top_emotions(self, detector):
        """Test getting top emotions with scores."""
        text = "¡Estoy muy feliz por esta noticia!"
        top_emotions = detector.get_top_emotions(text, top_k=3)
        assert len(top_emotions) <= 3
        
        # Check that results are sorted by confidence (descending)
        scores = [score for _, score in top_emotions]
        assert scores == sorted(scores, reverse=True)
        
        # Check that all emotions are valid
        for emotion, score in top_emotions:
            assert emotion in EMOTION_LABELS
            assert 0 <= score <= 1
    
    def test_get_supported_emotions(self, detector):
        """Test getting supported emotions."""
        emotions = detector.get_supported_emotions()
        assert emotions == EMOTION_LABELS
    
    def test_validate_emotions(self, detector):
        """Test emotion validation."""
        valid_emotions = ["alegría", "tristeza"]
        invalid_emotions = ["alegría", "invalid_emotion"]
        
        assert detector.validate_emotions(valid_emotions) == True
        assert detector.validate_emotions(invalid_emotions) == False
    
    def test_get_model_info(self, detector):
        """Test getting model information."""
        info = detector.get_model_info()
        assert isinstance(info, dict)
        assert 'device' in info
        assert 'emotions' in info
        assert 'fine_tuned' in info

class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_detect_emotions_function(self):
        """Test the convenience detect_emotions function."""
        from spanish_emotions import detect_emotions
        
        text = "¡Estoy muy feliz!"
        emotions = detect_emotions(text, model_key='beto', threshold=0.3)
        assert isinstance(emotions, list)
        for emotion in emotions:
            assert emotion in EMOTION_LABELS

class TestLibraryInfo:
    """Test library information functions."""
    
    def test_get_version(self):
        """Test version retrieval."""
        version = spanish_emotions.get_version()
        assert isinstance(version, str)
        assert len(version) > 0
    
    def test_get_supported_models(self):
        """Test supported models retrieval."""
        models = spanish_emotions.get_supported_models()
        assert isinstance(models, dict)
        assert 'beto' in models
    
    def test_get_supported_emotions_function(self):
        """Test supported emotions retrieval."""
        emotions = spanish_emotions.get_supported_emotions()
        assert emotions == EMOTION_LABELS

def test_integration():
    """Test integration between different components."""
    # Create sample data
    df = create_spanish_sample_dataset()
    
    # Test that we can normalize all texts
    processed_texts = df['text'].apply(normalize_spanish_text).tolist()
    assert len(processed_texts) == len(df)
    
    # Test encoding all emotions
    encoded_emotions = df['emotions'].apply(encode_emotions).tolist()
    assert len(encoded_emotions) == len(df)
    
    # Each encoding should have the correct length
    for encoding in encoded_emotions:
        assert len(encoding) == len(EMOTION_LABELS)

if __name__ == "__main__":
    pytest.main([__file__]) 