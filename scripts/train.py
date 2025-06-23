#!/usr/bin/env python3
"""
Training script for Spanish emotion detection models.
Supports fine-tuning, transfer learning, and multi-label classification.

Usage:
    python scripts/train.py --train-data data/train.json --eval-data data/eval.json
    python scripts/train.py --train-data data/train.json --config config.json
    python scripts/train.py --generate-sample-data
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spanish_emotions.trainer import (
    SpanishEmotionTrainer, 
    EmotionTrainingConfig,
    create_sample_training_data,
    EXTENDED_EMOTION_LABELS
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Spanish emotion detection models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default configuration
  python scripts/train.py --train-data data/train.json

  # Train with evaluation data
  python scripts/train.py --train-data data/train.json --eval-data data/eval.json

  # Train with custom configuration
  python scripts/train.py --train-data data/train.json --config custom_config.json

  # Resume from checkpoint
  python scripts/train.py --train-data data/train.json --resume-from models/checkpoint-1000

  # Generate sample training data
  python scripts/train.py --generate-sample-data --output data/sample_train.json

  # Train with specific model and custom settings
  python scripts/train.py --train-data data/train.json \\
    --base-model dccuchile/bert-base-spanish-wwm-uncased \\
    --epochs 5 \\
    --batch-size 8 \\
    --learning-rate 3e-5 \\
    --output-dir models/my_spanish_emotion_model

Data format expected in JSON files:
  [
    {"text": "Estoy feliz de verte", "labels": ["alegría"]},
    {"text": "Me siento triste y confundido", "labels": ["tristeza", "confusión"]}
  ]
        """
    )

    # Data arguments
    parser.add_argument("--train-data", type=str, help="Path to training data JSON file")
    parser.add_argument("--eval-data", type=str, help="Path to evaluation data JSON file")
    parser.add_argument("--config", type=str, help="Path to training configuration JSON file")

    # Model arguments
    parser.add_argument("--base-model", type=str, default="dccuchile/bert-base-spanish-wwm-uncased",
                       help="Base model for fine-tuning")
    parser.add_argument("--num-labels", type=int, default=27,
                       help="Number of emotion labels")
    parser.add_argument("--max-length", type=int, default=512,
                       help="Maximum sequence length")

    # Training arguments
    parser.add_argument("--output-dir", type=str, default="models/spanish_emotion_model",
                       help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="Training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=32,
                       help="Evaluation batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                       help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                       help="Weight decay")

    # Advanced training arguments
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1,
                       help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--eval-steps", type=int, default=500,
                       help="Number of steps between evaluations")
    parser.add_argument("--save-steps", type=int, default=500,
                       help="Number of steps between model saves")
    parser.add_argument("--logging-steps", type=int, default=100,
                       help="Number of steps between log outputs")
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                       help="Early stopping patience")

    # Resume and utility arguments
    parser.add_argument("--resume-from", type=str,
                       help="Path to checkpoint to resume training from")
    parser.add_argument("--generate-sample-data", action="store_true",
                       help="Generate sample training data")
    parser.add_argument("--output", type=str, default="data/sample_train.json",
                       help="Output path for sample data generation")
    parser.add_argument("--num-samples", type=int, default=200,
                       help="Number of samples to generate")

    # Logging and monitoring
    parser.add_argument("--wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--tensorboard", action="store_true",
                       help="Enable TensorBoard logging")

    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load training configuration from JSON file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_config_from_args(args) -> EmotionTrainingConfig:
    """Create training configuration from command line arguments."""
    
    # Determine report_to setting
    report_to = None
    if args.wandb and args.tensorboard:
        report_to = ["wandb", "tensorboard"]
    elif args.wandb:
        report_to = "wandb"
    elif args.tensorboard:
        report_to = "tensorboard"

    return EmotionTrainingConfig(
        # Model configuration
        base_model=args.base_model,
        num_labels=args.num_labels,
        max_length=args.max_length,
        
        # Training configuration
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        
        # Evaluation and saving
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        early_stopping_patience=args.early_stopping_patience,
        
        # Logging
        report_to=report_to,
    )


def validate_data_file(data_path: str):
    """Validate the format of a training data file."""
    print(f"Validating data file: {data_path}")
    
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            raise ValueError("Data must be a list of objects")
        
        required_fields = ['text', 'labels']
        valid_emotions = set(EXTENDED_EMOTION_LABELS)
        
        for i, item in enumerate(data[:5]):  # Check first 5 items
            if not isinstance(item, dict):
                raise ValueError(f"Item {i} is not a dictionary")
            
            for field in required_fields:
                if field not in item:
                    raise ValueError(f"Item {i} missing required field '{field}'")
            
            # Validate labels
            labels = item['labels']
            if isinstance(labels, str):
                labels = [labels]
            
            unknown_labels = set(labels) - valid_emotions
            if unknown_labels:
                print(f"Warning: Unknown labels in item {i}: {unknown_labels}")
        
        print(f"✓ Data file is valid. Contains {len(data)} samples.")
        
        # Show label distribution
        all_labels = []
        for item in data:
            labels = item['labels']
            if isinstance(labels, str):
                labels = [labels]
            all_labels.extend(labels)
        
        from collections import Counter
        label_counts = Counter(all_labels)
        print(f"Label distribution: {dict(label_counts.most_common(10))}")
        
    except Exception as e:
        print(f"✗ Data validation failed: {e}")
        sys.exit(1)


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Handle sample data generation
    if args.generate_sample_data:
        print("Generating sample training data...")
        output_path = create_sample_training_data(args.output, args.num_samples)
        print(f"Sample data saved to: {output_path}")
        return
    
    # Validate required arguments
    if not args.train_data:
        print("Error: --train-data is required for training")
        sys.exit(1)
    
    if not os.path.exists(args.train_data):
        print(f"Error: Training data file not found: {args.train_data}")
        sys.exit(1)
    
    # Validate data files
    validate_data_file(args.train_data)
    if args.eval_data:
        if not os.path.exists(args.eval_data):
            print(f"Error: Evaluation data file not found: {args.eval_data}")
            sys.exit(1)
        validate_data_file(args.eval_data)
    
    # Create training configuration
    if args.config:
        print(f"Loading configuration from: {args.config}")
        config_dict = load_config_from_file(args.config)
        config = EmotionTrainingConfig(**config_dict)
    else:
        config = create_config_from_args(args)
    
    # Print configuration
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION")
    print("="*50)
    print(f"Base model: {config.base_model}")
    print(f"Number of labels: {config.num_labels}")
    print(f"Output directory: {config.output_dir}")
    print(f"Epochs: {config.num_train_epochs}")
    print(f"Batch size: {config.per_device_train_batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max length: {config.max_length}")
    print(f"Training data: {args.train_data}")
    print(f"Evaluation data: {args.eval_data if args.eval_data else 'None'}")
    print(f"Resume from: {args.resume_from if args.resume_from else 'None'}")
    print("="*50 + "\n")
    
    # Initialize trainer
    trainer = SpanishEmotionTrainer(config)
    
    # Start training
    try:
        print("Starting training...")
        result = trainer.train(
            train_data_path=args.train_data,
            eval_data_path=args.eval_data,
            resume_from_checkpoint=args.resume_from
        )
        
        print("\n" + "="*50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*50)
        print(f"Model saved to: {config.output_dir}")
        print("\nFiles created:")
        
        output_path = Path(config.output_dir)
        if output_path.exists():
            for file in output_path.glob("*"):
                print(f"  {file}")
        
        print(f"\nTo use your trained model:")
        print(f"  from spanish_emotions.detector import EmotionDetector")
        print(f"  detector = EmotionDetector(model_path='{config.output_dir}')")
        print(f"  emotions = detector.detect('Tu texto en español aquí')")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 