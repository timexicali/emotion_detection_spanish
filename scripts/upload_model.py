#!/usr/bin/env python3
"""
Upload Spanish emotion detection models to Hugging Face Hub.
Supports model upload, metadata generation, and model card creation.

Usage:
    python scripts/upload_model.py --model-path models/spanish_emotion_model --repo-name my-spanish-emotions
    python scripts/upload_model.py --model-path models/spanish_emotion_model --repo-name my-spanish-emotions --private
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from huggingface_hub import HfApi, create_repo, upload_folder
    from huggingface_hub.utils import HfHubHTTPError
except ImportError:
    print("Error: huggingface_hub is required for uploading models.")
    print("Install with: pip install huggingface-hub")
    sys.exit(1)

from spanish_emotions.trainer import EXTENDED_EMOTION_LABELS


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Upload Spanish emotion detection models to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload model to Hugging Face Hub
  python scripts/upload_model.py \\
    --model-path models/spanish_emotion_model \\
    --repo-name username/spanish-emotion-detector

  # Upload private model with custom settings
  python scripts/upload_model.py \\
    --model-path models/spanish_emotion_model \\
    --repo-name username/spanish-emotion-detector \\
    --private \\
    --model-description "Fine-tuned Spanish emotion detection model"

  # Generate model card only (no upload)
  python scripts/upload_model.py \\
    --model-path models/spanish_emotion_model \\
    --generate-card-only

Prerequisites:
  1. Install huggingface_hub: pip install huggingface-hub
  2. Login to Hugging Face: huggingface-cli login
  3. Have a trained model in the specified path
        """
    )

    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to the trained model directory")
    parser.add_argument("--repo-name", type=str, 
                       help="Repository name on Hugging Face Hub (e.g., 'username/model-name')")
    parser.add_argument("--model-description", type=str,
                       default="Spanish emotion detection model fine-tuned for multi-label classification",
                       help="Description of the model")
    parser.add_argument("--private", action="store_true",
                       help="Make the repository private")
    parser.add_argument("--generate-card-only", action="store_true",
                       help="Only generate model card, don't upload")
    parser.add_argument("--token", type=str,
                       help="Hugging Face Hub token (if not logged in)")
    parser.add_argument("--commit-message", type=str,
                       default="Upload Spanish emotion detection model",
                       help="Commit message for the upload")
    
    return parser.parse_args()


def validate_model_directory(model_path: str) -> Dict[str, bool]:
    """Validate that the model directory contains required files."""
    model_path = Path(model_path)
    
    required_files = {
        'config.json': model_path / 'config.json',
        'pytorch_model.bin': model_path / 'pytorch_model.bin',
        'tokenizer.json': model_path / 'tokenizer.json',
        'tokenizer_config.json': model_path / 'tokenizer_config.json',
    }
    
    # Alternative model files
    alternatives = {
        'pytorch_model.bin': [
            model_path / 'model.safetensors',
            model_path / 'pytorch_model.bin'
        ]
    }
    
    validation_results = {}
    
    for name, file_path in required_files.items():
        if name in alternatives:
            # Check alternatives
            found = any(alt.exists() for alt in alternatives[name])
            validation_results[name] = found
        else:
            validation_results[name] = file_path.exists()
    
    return validation_results


def load_model_metadata(model_path: str) -> Dict:
    """Load model metadata and configuration."""
    model_path = Path(model_path)
    
    metadata = {}
    
    # Load model config
    config_path = model_path / 'config.json'
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        metadata['config'] = config
        metadata['num_labels'] = config.get('num_labels', 27)
        metadata['model_type'] = config.get('model_type', 'bert')
    
    # Load emotion labels if available
    labels_path = model_path / 'emotion_labels.json'
    if labels_path.exists():
        with open(labels_path, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        metadata['emotion_labels'] = labels
    else:
        metadata['emotion_labels'] = EXTENDED_EMOTION_LABELS
    
    return metadata


def generate_model_card(
    model_path: str,
    repo_name: str,
    description: str,
    metadata: Dict
) -> str:
    """Generate a comprehensive model card."""
    
    emotion_labels = metadata.get('emotion_labels', EXTENDED_EMOTION_LABELS)
    num_labels = metadata.get('num_labels', len(emotion_labels))
    model_type = metadata.get('model_type', 'bert')
    
    model_card = f"""---
language: es
license: mit
tags:
- emotion-detection
- spanish
- transformers
- pytorch
- multi-label-classification
- sentiment-analysis
- nlp
datasets:
- custom
widget:
- text: "¡Estoy muy feliz por esta noticia increíble!"
  example_title: "Alegría"
- text: "Me siento muy triste y deprimido"
  example_title: "Tristeza"
- text: "¡Qué rabia me da esta situación!"
  example_title: "Enojo"
- text: "Tengo mucho miedo de lo que pueda pasar"
  example_title: "Miedo"
model-index:
- name: {repo_name.split('/')[-1] if '/' in repo_name else repo_name}
  results:
  - task:
      type: text-classification
      name: Multi-label Emotion Classification
    dataset:
      type: custom
      name: Spanish Emotion Dataset
    metrics:
    - type: f1
      name: F1 Score
      value: "TBD"
---

# Spanish Emotion Detection Model

{description}

## Model Description

This model is a fine-tuned transformer for detecting emotions in Spanish text. It supports multi-label classification across {num_labels} different emotional categories.

### Supported Emotions

The model can detect the following emotions in Spanish text:

"""
    
    # Add emotion list
    for i, emotion in enumerate(emotion_labels, 1):
        model_card += f"{i}. **{emotion}** - {emotion.title()}\n"
    
    model_card += f"""

## Model Details

- **Model Type**: {model_type.upper()} for Sequence Classification
- **Language**: Spanish (es)
- **Number of Labels**: {num_labels}
- **License**: MIT
- **Base Model**: dccuchile/bert-base-spanish-wwm-uncased (or compatible)

## Intended Uses & Limitations

### Intended Uses

- Emotion detection in Spanish text
- Content moderation and sentiment analysis
- Research in computational linguistics
- Building emotion-aware applications

### Limitations

- Trained specifically for Spanish language
- Performance may vary on different domains
- May require fine-tuning for specialized applications
- Multi-label predictions may need threshold tuning

## How to Use

### Installation

```bash
pip install transformers torch
```

### Basic Usage

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{repo_name}")
model = AutoModelForSequenceClassification.from_pretrained("{repo_name}")

# Prepare text
text = "¡Estoy muy feliz por esta noticia increíble!"
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

# Get predictions
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.sigmoid(outputs.logits)

# Get emotion labels (you may need to define these)
emotion_labels = {emotion_labels}

# Convert to emotions (using 0.5 threshold)
detected_emotions = []
for i, score in enumerate(predictions[0]):
    if score > 0.5:
        detected_emotions.append(emotion_labels[i])

print(f"Detected emotions: {{detected_emotions}}")
```

### Advanced Usage with spanish_emotions Library

If you have the full `spanish_emotions` library:

```python
from spanish_emotions import EmotionDetector

# Load your custom model
detector = EmotionDetector(model_path="{repo_name}")

# Detect emotions
emotions = detector.detect("¡Estoy súper emocionado!")
print(emotions)

# Get confidence scores
scores = detector.detect("¡Estoy súper emocionado!", return_scores=True)
print(scores)

# Batch processing
texts = ["¡Estoy feliz!", "Me siento triste", "Tengo miedo"]
results = detector.detect_batch(texts)
print(results)
```

## Training Data

The model was fine-tuned on a Spanish emotion dataset with multi-label annotations. Each text sample can have multiple emotion labels.

### Data Format

```json
[
  {{"text": "Estoy feliz de verte", "labels": ["alegría"]}},
  {{"text": "Me siento triste y confundido", "labels": ["tristeza", "confusión"]}}
]
```

## Training Procedure

### Training Hyperparameters

- **Learning Rate**: 2e-5
- **Batch Size**: 16
- **Epochs**: 3
- **Warmup Ratio**: 0.1
- **Weight Decay**: 0.01
- **Max Sequence Length**: 512

### Framework Versions

- **Transformers**: 4.30+
- **PyTorch**: 2.0+
- **Python**: 3.8+

## Evaluation

The model performance should be evaluated on a held-out test set using metrics appropriate for multi-label classification:

- F1 Score (macro average)
- Precision and Recall
- Hamming Loss
- Exact Match Ratio

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{spanish-emotion-model,
  author = {{Your Name}},
  title = {{Spanish Emotion Detection Model}},
  year = {{2024}},
  publisher = {{Hugging Face}},
  url = {{https://huggingface.co/{repo_name}}}
}}
```

## Contact

For questions or issues, please open an issue in the model repository or contact the authors.
"""
    
    return model_card


def upload_to_hub(
    model_path: str,
    repo_name: str,
    token: Optional[str] = None,
    private: bool = False,
    commit_message: str = "Upload Spanish emotion detection model"
):
    """Upload model to Hugging Face Hub."""
    
    api = HfApi(token=token)
    
    try:
        # Create repository
        print(f"Creating repository: {repo_name}")
        repo_url = create_repo(
            repo_id=repo_name,
            token=token,
            private=private,
            exist_ok=True
        )
        print(f"Repository created/exists: {repo_url}")
        
        # Upload model files
        print(f"Uploading model files from {model_path}...")
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_name,
            commit_message=commit_message,
            token=token
        )
        
        print(f"✅ Model successfully uploaded to: https://huggingface.co/{repo_name}")
        return True
        
    except HfHubHTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def main():
    """Main upload function."""
    args = parse_arguments()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    print(f"Validating model directory: {args.model_path}")
    validation = validate_model_directory(args.model_path)
    
    print("File validation results:")
    all_valid = True
    for file_name, exists in validation.items():
        status = "✅" if exists else "❌"
        print(f"  {status} {file_name}")
        if not exists:
            all_valid = False
    
    if not all_valid:
        print("❌ Model directory is missing required files.")
        print("Make sure the model was saved properly after training.")
        sys.exit(1)
    
    # Load model metadata
    print("Loading model metadata...")
    metadata = load_model_metadata(args.model_path)
    print(f"Model type: {metadata.get('model_type', 'unknown')}")
    print(f"Number of labels: {metadata.get('num_labels', 'unknown')}")
    print(f"Emotion labels: {len(metadata.get('emotion_labels', []))}")
    
    # Generate model card
    if args.generate_card_only or args.repo_name:
        repo_name = args.repo_name or "username/spanish-emotion-model"
        print("Generating model card...")
        
        model_card = generate_model_card(
            model_path=args.model_path,
            repo_name=repo_name,
            description=args.model_description,
            metadata=metadata
        )
        
        # Save model card
        readme_path = Path(args.model_path) / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(model_card)
        
        print(f"✅ Model card saved to: {readme_path}")
    
    # Upload to Hub (if not generate-only)
    if not args.generate_card_only:
        if not args.repo_name:
            print("❌ --repo-name is required for uploading to Hugging Face Hub")
            print("Use --generate-card-only to only generate the model card")
            sys.exit(1)
        
        print(f"Starting upload to Hugging Face Hub...")
        print(f"Repository: {args.repo_name}")
        print(f"Private: {args.private}")
        
        success = upload_to_hub(
            model_path=args.model_path,
            repo_name=args.repo_name,
            token=args.token,
            private=args.private,
            commit_message=args.commit_message
        )
        
        if success:
            print("\n" + "="*50)
            print("UPLOAD COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"Model URL: https://huggingface.co/{args.repo_name}")
            print(f"To use your model:")
            print(f"  from transformers import AutoTokenizer, AutoModelForSequenceClassification")
            print(f"  tokenizer = AutoTokenizer.from_pretrained('{args.repo_name}')")
            print(f"  model = AutoModelForSequenceClassification.from_pretrained('{args.repo_name}')")
            print("="*50)
        else:
            print("❌ Upload failed. Check the error messages above.")
            sys.exit(1)


if __name__ == "__main__":
    main() 