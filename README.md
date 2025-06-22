# Spanish Emotion Detection Library 🇪🇸 🔥

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A lightweight, **commercial-use-ready** Python library for detecting emotions in Spanish text using transformer models (BETO/RoBERTuito). Perfect for sentiment analysis, content moderation, and emotion-aware applications.

## ✨ Features

- 🎯 **Multi-label emotion detection** in Spanish text
- 🤖 **Transformer-based** using BETO, RoBERTuito, or other Spanish models  
- 🏭 **Commercial-friendly** MIT license
- 📦 **Pure Python library** - no server dependencies
- 🚀 **Easy integration** into existing Python projects
- 📊 **7 emotion categories**: alegría, tristeza, enojo, miedo, sorpresa, desagrado, neutral
- 🔧 **Flexible API** with batch processing support
- 💻 **CLI interface** for command-line usage

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/emotion_detection_spanish.git
cd emotion_detection_spanish

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from spanish_emotions import EmotionDetector

# Initialize detector with BETO model
detector = EmotionDetector(model_key='beto')

# Detect emotions in Spanish text
emotions = detector.detect("¡Estoy muy feliz por esta noticia!")
print(emotions)  # ['alegría']

# Get confidence scores
scores = detector.detect("¡Estoy muy feliz!", return_scores=True)
print(scores)  # {'alegría': 0.89, 'tristeza': 0.02, ...}

# Batch processing
texts = ["¡Estoy feliz!", "Me siento triste", "Tengo miedo"]
results = detector.detect_batch(texts)
print(results)  # [['alegría'], ['tristeza'], ['miedo']]
```

### Quick Detection Function
```python
from spanish_emotions import detect_emotions

# One-line emotion detection
emotions = detect_emotions("¡Qué alegría me da verte!")
print(emotions)  # ['alegría']
```

## 📋 Available Emotions

| Español | English | Description |  
|---------|---------|-------------|
| alegría | joy | Happiness, contentment |
| tristeza | sadness | Sadness, sorrow |
| enojo | anger | Anger, irritation |
| miedo | fear | Fear, anxiety |
| sorpresa | surprise | Surprise, amazement |
| desagrado | disgust | Disgust, dislike |
| neutral | neutral | Neutral emotional state |

## 🤖 Supported Models

The library supports multiple Spanish transformer models:

| Model Key | Model Name | Description | Recommended Use |
|-----------|------------|-------------|-----------------|
| `beto` | `dccuchile/bert-base-spanish-wwm-uncased` | BETO - Spanish BERT | General purpose, good balance |
| `robertuito` | `pysentimiento/robertuito-base-uncased` | RoBERTuito - Spanish RoBERTa | Social media, informal text |
| `roberta-bne` | `BSC-TeMU/roberta-base-bne` | RoBERTa trained on BNE corpus | Formal text, news |
| `beto-cased` | `dccuchile/bert-base-spanish-wwm-cased` | Cased BETO model | Case-sensitive applications |

```python
# Use different models
detector_beto = EmotionDetector(model_key='beto')
detector_robertuito = EmotionDetector(model_key='robertuito')

# Use custom model path
detector_custom = EmotionDetector(model_path='path/to/your/fine-tuned-model')
```

## 💻 Command Line Interface

### Basic Detection
```bash
# Detect emotions in text
python -m spanish_emotions detect "¡Estoy muy feliz por esta noticia!"

# With confidence scores
python -m spanish_emotions detect "Me siento triste" --scores

# Use different model
python -m spanish_emotions detect "¡Increíble!" --model-key robertuito

# Set custom threshold
python -m spanish_emotions detect "¡Estoy feliz!" --threshold 0.3
```

### File Processing
```bash
# Process file (one text per line)
python -m spanish_emotions detect --file texts.txt

# Output as JSON
python -m spanish_emotions detect --file texts.txt --json

# From stdin
echo "Tengo mucho miedo" | python -m spanish_emotions detect
```

### Dataset Operations
```bash
# Create sample dataset
python -m spanish_emotions sample sample_data.csv

# Validate dataset format
python -m spanish_emotions validate my_dataset.csv

# Show library information
python -m spanish_emotions info
```

## 🔧 Advanced Usage

### Batch Processing
```python
detector = EmotionDetector(model_key='beto')

# Process multiple texts efficiently
texts = [
    "¡Qué día tan maravilloso!",
    "Me siento muy preocupado por esto",
    "No sé qué pensar de esta situación"
]

# Batch detection with custom threshold
results = detector.detect_batch(texts, threshold=0.4, batch_size=16)
for text, emotions in zip(texts, results):
    print(f"'{text}' → {emotions}")
```

### Top Emotions Analysis
```python
# Get top emotions with confidence scores
text = "¡Estoy súper emocionado pero también nervioso!"
top_emotions = detector.get_top_emotions(text, top_k=3)

for rank, (emotion, confidence) in enumerate(top_emotions, 1):
    print(f"{rank}. {emotion}: {confidence:.3f}")
```

### Full Emotion Distribution
```python
# Analyze complete emotion distribution
distribution = detector.analyze_emotion_distribution("¡Qué sorpresa tan agradable!")

for emotion, score in distribution.items():
    print(f"{emotion}: {score:.3f}")
```

### Text Preprocessing
```python
from spanish_emotions import normalize_spanish_text

# Normalize Spanish text
text = "¡¡¡HOLA!!! @usuario ¿Cómo estás? 😊 https://example.com"
normalized = normalize_spanish_text(text)
print(normalized)  # "¡hola! ¿cómo estás? :cara_feliz_con_ojos_sonrientes:"
```

## 📊 Working with Datasets

### Sample Dataset
```python
from spanish_emotions import create_spanish_sample_dataset

# Create sample dataset for testing
df = create_spanish_sample_dataset()
print(f"Sample dataset: {len(df)} samples")
print(df.head())
```

### Dataset Validation
```python
from spanish_emotions import validate_dataset
import pandas as pd

# Load and validate your dataset
df = pd.read_csv('my_emotions_dataset.csv')
results = validate_dataset(df, text_column='text', emotion_column='emotions')

if results['valid']:
    print("✅ Dataset is valid!")
    print(f"📊 Statistics: {results['stats']}")
else:
    print("❌ Dataset has issues:")
    for error in results['errors']:
        print(f"  - {error}")
```

### Dataset Format

Your training data should be a CSV file with the following format:

```csv
text,emotions
"¡Estoy muy feliz por esta noticia!","['alegría']"
"Me siento triste y enojado","['tristeza', 'enojo']"
"Tengo mucho miedo de esto","['miedo']"
"El clima está normal hoy","['neutral']"
```

## 🏗️ Library Structure

```
spanish_emotions/
├── __init__.py          # Main package exports
├── detector.py          # EmotionDetector class
├── model_loader.py      # Hugging Face model loading
├── labels.py            # Spanish emotion labels
├── preprocessing.py     # Text normalization
├── utils.py             # Helper functions
└── __main__.py          # CLI interface
```

## 🔬 Technical Details

### Text Preprocessing
- URL removal
- Social media mention and hashtag normalization
- Emoji handling (preserves emotional content in Spanish)
- Spanish accent preservation
- Excessive punctuation normalization
- Case normalization

### Model Architecture
- Multi-label classification approach
- Sigmoid activation for independent emotion probabilities
- Support for custom confidence thresholds
- Efficient batch processing

### Performance Considerations
- Automatic GPU detection and usage
- Optimized tokenization and batching
- Model caching for faster repeated usage
- Memory-efficient batch processing

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_spanish_emotions.py::TestEmotionDetector -v

# Run with coverage
python -m pytest tests/ --cov=spanish_emotions
```

## 🔌 Integration Examples

### Flask Web Application
```python
from flask import Flask, request, jsonify
from spanish_emotions import EmotionDetector

app = Flask(__name__)
detector = EmotionDetector(model_key='beto')

@app.route('/emotions', methods=['POST'])
def analyze_emotions():
    text = request.json.get('text')
    emotions = detector.detect(text, return_scores=True)
    return jsonify({'emotions': emotions})
```

### Pandas Integration
```python
import pandas as pd
from spanish_emotions import EmotionDetector

detector = EmotionDetector()

# Apply to pandas DataFrame
df['emotions'] = df['spanish_text'].apply(
    lambda text: detector.detect(text, threshold=0.5)
)

# Batch processing for better performance
emotions_batch = detector.detect_batch(df['spanish_text'].tolist())
df['emotions'] = emotions_batch
```

### Data Pipeline
```python
from spanish_emotions import EmotionDetector, normalize_spanish_text

def process_spanish_texts(texts):
    detector = EmotionDetector(model_key='beto')
    
    # Preprocess texts
    normalized_texts = [normalize_spanish_text(text) for text in texts]
    
    # Batch emotion detection
    emotions = detector.detect_batch(normalized_texts, threshold=0.4)
    
    return list(zip(texts, emotions))
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python -m pytest tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. This allows for **commercial use** without restrictions.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/) for the transformer models
- [BETO](https://github.com/dccuchile/beto) for Spanish BERT
- [RoBERTuito](https://github.com/pysentimiento/robertuito) for Spanish RoBERTa
- The open-source NLP community for tools and inspiration

## 📞 Support

- 📧 Email: daniel.gm78@emotionwise.ai
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/emotion_detection_spanish/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/emotion_detection_spanish/discussions)

---

**Made with ❤️ for the Spanish NLP community**
