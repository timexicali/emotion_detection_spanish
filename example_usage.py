#!/usr/bin/env python3
"""
Example usage of the Spanish Emotion Detection Library
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(__file__))

import spanish_emotions
from spanish_emotions import (
    EmotionDetector, 
    EMOTION_LABELS, 
    normalize_spanish_text,
    create_spanish_sample_dataset,
    detect_emotions
)

def main():
    print("🇪🇸 Spanish Emotion Detection Library - Example Usage")
    print("=" * 60)
    
    # 1. Show library information
    print("\n📖 Library Information:")
    spanish_emotions.info()
    
    # 2. Show available emotions
    print(f"\n📊 Available Emotions ({len(EMOTION_LABELS)}):")
    for i, emotion in enumerate(EMOTION_LABELS, 1):
        print(f"  {i}. {emotion}")
    
    # 3. Show available models
    print(f"\n🤖 Available Spanish Models:")
    models = spanish_emotions.get_supported_models()
    for key, model_name in models.items():
        print(f"  {key}: {model_name}")
    
    # 4. Demonstrate text preprocessing
    print("\n🔧 Text Preprocessing:")
    sample_texts = [
        "¡Hola! @usuario ¿Cómo estás? 😊 https://example.com",
        "ME SIENTO MUY TRISTE!!!! 😢😢😢",
        "No puedo creer esto... #increíble #sorpresa"
    ]
    
    for text in sample_texts:
        processed = normalize_spanish_text(text)
        print(f"  Original: {text}")
        print(f"  Processed: {processed}")
        print()
    
    print("🤖 Initializing Spanish Emotion Detector...")
    try:
        # Initialize detector with BETO model
        detector = EmotionDetector(model_key='beto')
        print("✅ Detector initialized successfully!")
        print(f"📱 Using device: {detector.device}")
        print(f"🔬 Fine-tuned: {detector.is_fine_tuned()}")
        
        # 5. Single text predictions
        print("\n🔍 Single Text Predictions:")
        test_texts = [
            "¡Estoy muy feliz por esta noticia!",
            "Me siento muy triste por lo que pasó",
            "Tengo mucho miedo de lo que pueda pasar",
            "¡Qué asco me da esta situación!",
            "¡No puedo creer lo que acaba de pasar!",
            "Estoy muy enojado con esta decisión"
        ]
        
        for text in test_texts:
            emotions = detector.detect(text, threshold=0.3)
            print(f"  Text: '{text}'")
            print(f"  Emotions: {emotions}")
            print()
        
        # 6. Predictions with confidence scores
        print("\n📈 Predictions with Confidence Scores:")
        example_text = "¡Qué alegría me da verte! Aunque tengo un poco de miedo"
        scores = detector.detect(example_text, return_scores=True)
        
        print(f"  Text: '{example_text}'")
        print("  Confidence Scores:")
        for emotion, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            print(f"    {emotion}: {score:.3f}")
        print()
        
        # 7. Top emotions
        print("\n🏆 Top 3 Emotions:")
        top_emotions = detector.get_top_emotions(example_text, top_k=3)
        for i, (emotion, score) in enumerate(top_emotions, 1):
            print(f"  {i}. {emotion}: {score:.3f}")
        print()
        
        # 8. Batch predictions
        print("\n⚡ Batch Predictions:")
        batch_texts = [
            "¡Increíble noticia!",
            "Me siento deprimido",
            "Tengo pánico",
            "Qué aburrido esto",
            "¡Qué sorpresa tan agradable!"
        ]
        
        batch_results = detector.detect_batch(batch_texts, threshold=0.4)
        for text, emotions in zip(batch_texts, batch_results):
            print(f"  '{text}' → {emotions}")
        print()
        
        # 9. Emotion distribution analysis
        print("\n📊 Emotion Distribution Analysis:")
        analysis_text = "¡Estoy súper emocionado pero también un poco nervioso!"
        distribution = detector.analyze_emotion_distribution(analysis_text)
        print(f"  Text: '{analysis_text}'")
        print("  Full Distribution:")
        for emotion, score in distribution.items():
            bar_length = int(score * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"    {emotion:>10}: {bar} {score:.3f}")
        print()
        
    except Exception as e:
        print(f"❌ Error initializing detector: {e}")
        print("💡 Note: This might happen if the model can't be downloaded or loaded.")
        print("   The library will still work once you have internet access.")
    
    # 10. Convenience function demo
    print("\n🚀 Convenience Function Demo:")
    try:
        emotions = detect_emotions("¡Esto es fantástico!", model_key='beto', threshold=0.4)
        print(f"  Quick detection: 'Esto es fantástico!' → {emotions}")
    except Exception as e:
        print(f"  Error with convenience function: {e}")
    
    # 11. Show sample dataset
    print("\n📋 Sample Training Dataset:")
    sample_df = create_spanish_sample_dataset()
    print(f"  Dataset size: {len(sample_df)} samples")
    print("  First few examples:")
    for idx, row in sample_df.head(3).iterrows():
        print(f"    Text: '{row['text']}'")
        print(f"    Emotions: {row['emotions']}")
        print()
    
    # 12. CLI examples
    print("\n💻 CLI Usage Examples:")
    print("  # Basic emotion detection")
    print("  python -m spanish_emotions detect \"¡Estoy muy feliz!\"")
    print("  ")
    print("  # With confidence scores")
    print("  python -m spanish_emotions detect \"Me siento triste\" --scores")
    print("  ")
    print("  # From file with JSON output")
    print("  python -m spanish_emotions detect --file texts.txt --json")
    print("  ")
    print("  # From stdin")
    print("  echo \"Tengo miedo\" | python -m spanish_emotions detect")
    print("  ")
    print("  # Create sample dataset")
    print("  python -m spanish_emotions sample sample_data.csv")
    print("  ")
    print("  # Validate dataset")
    print("  python -m spanish_emotions validate my_dataset.csv")
    print("  ")
    print("  # Show library info")
    print("  python -m spanish_emotions info")
    print()
    
    # 13. Python API examples
    print("\n🐍 Python API Examples:")
    print("  # Basic usage")
    print("  from spanish_emotions import EmotionDetector")
    print("  detector = EmotionDetector(model_key='beto')")
    print("  emotions = detector.detect('¡Estoy feliz!')")
    print("  ")
    print("  # Quick detection")
    print("  from spanish_emotions import detect_emotions")
    print("  emotions = detect_emotions('Text in Spanish')")
    print("  ")
    print("  # Batch processing")
    print("  texts = ['Text 1', 'Text 2', 'Text 3']")
    print("  results = detector.detect_batch(texts)")
    print("  ")
    print("  # With confidence scores")
    print("  scores = detector.detect('Text', return_scores=True)")
    print("  top_emotions = detector.get_top_emotions('Text', top_k=3)")
    print()
    
    print("✨ Example completed!")
    print("📚 This is a pure Python library - no FastAPI or server components.")
    print("🔧 Use it as a dependency in your own Python projects.")
    print("📖 Check the README.md for more detailed usage instructions.")

if __name__ == "__main__":
    main() 