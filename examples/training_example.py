#!/usr/bin/env python3
"""
Example script demonstrating how to train Spanish emotion detection models.
Shows different training scenarios and configurations.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spanish_emotions import (
    SpanishEmotionTrainer,
    EmotionTrainingConfig,
    EXTENDED_EMOTION_LABELS,
    create_sample_training_data
)


def example_1_basic_training():
    """Example 1: Basic training with default configuration."""
    print("="*60)
    print("EXAMPLE 1: Basic Training")
    print("="*60)
    
    # Create sample data for demonstration
    print("Creating sample training data...")
    train_data_path = "data/sample_train.json"
    eval_data_path = "data/sample_eval.json"
    
    create_sample_training_data(train_data_path, num_samples=100)
    create_sample_training_data(eval_data_path, num_samples=30)
    
    # Basic configuration
    config = EmotionTrainingConfig(
        output_dir="models/basic_spanish_emotion_model",
        num_train_epochs=1,  # Small for demo
        per_device_train_batch_size=8,
        learning_rate=2e-5,
    )
    
    # Initialize trainer
    trainer = SpanishEmotionTrainer(config)
    
    # Start training
    print("Starting basic training...")
    try:
        trainer.train(
            train_data_path=train_data_path,
            eval_data_path=eval_data_path
        )
        print("✅ Basic training completed!")
    except Exception as e:
        print(f"❌ Training failed: {e}")


def example_2_custom_configuration():
    """Example 2: Training with custom configuration."""
    print("="*60)
    print("EXAMPLE 2: Custom Configuration")
    print("="*60)
    
    # Custom configuration for better performance
    config = EmotionTrainingConfig(
        # Model settings
        base_model="dccuchile/bert-base-spanish-wwm-uncased",
        num_labels=27,
        max_length=256,  # Shorter for faster training
        
        # Training settings
        output_dir="models/custom_spanish_emotion_model",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        # Evaluation settings
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
        logging_steps=25,
        
        # Early stopping
        early_stopping_patience=3,
        early_stopping_threshold=0.001,
        
        # Monitoring
        report_to=None,  # Set to "wandb" if you have wandb setup
    )
    
    print("Training configuration:")
    print(f"  Base model: {config.base_model}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Max length: {config.max_length}")


def example_3_resume_training():
    """Example 3: Resume training from checkpoint."""
    print("="*60)
    print("EXAMPLE 3: Resume Training")
    print("="*60)
    
    config = EmotionTrainingConfig(
        output_dir="models/resumed_spanish_emotion_model",
        num_train_epochs=5,
        save_steps=50,
    )
    
    trainer = SpanishEmotionTrainer(config)
    
    # Check for existing checkpoints
    checkpoint_dir = Path(config.output_dir)
    checkpoints = list(checkpoint_dir.glob("checkpoint-*")) if checkpoint_dir.exists() else []
    
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.name.split('-')[1]))
        print(f"Found checkpoint: {latest_checkpoint}")
        print("Would resume training from this checkpoint...")
        # trainer.train(
        #     train_data_path="data/sample_train.json",
        #     resume_from_checkpoint=str(latest_checkpoint)
        # )
    else:
        print("No checkpoints found. Would start fresh training...")


def example_4_different_models():
    """Example 4: Training with different base models."""
    print("="*60)
    print("EXAMPLE 4: Different Base Models")
    print("="*60)
    
    # Different Spanish models you can use
    spanish_models = [
        "dccuchile/bert-base-spanish-wwm-uncased",
        "pysentimiento/robertuito-base-uncased",
        "BSC-TeMU/roberta-base-bne",
    ]
    
    for model_name in spanish_models:
        print(f"\nModel: {model_name}")
        print(f"  Recommended for: {get_model_recommendation(model_name)}")
        
        # Create configuration for each model
        config = EmotionTrainingConfig(
            base_model=model_name,
            output_dir=f"models/{model_name.split('/')[-1]}_emotion_model",
            num_train_epochs=2,
        )
        
        print(f"  Would save to: {config.output_dir}")


def example_5_production_config():
    """Example 5: Production-ready configuration."""
    print("="*60)
    print("EXAMPLE 5: Production Configuration")
    print("="*60)
    
    # Production configuration with best practices
    config = EmotionTrainingConfig(
        # Model
        base_model="dccuchile/bert-base-spanish-wwm-uncased",
        num_labels=27,
        max_length=512,
        
        # Training
        output_dir="models/production_spanish_emotion_model",
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,  # Effective batch size: 8*4=32
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        
        # Evaluation and saving
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps", 
        save_steps=500,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        
        # Early stopping for efficiency
        early_stopping_patience=5,
        early_stopping_threshold=0.001,
        
        # Logging
        logging_steps=100,
        report_to="tensorboard",  # Enable TensorBoard logging
    )
    
    print("Production configuration summary:")
    print(f"  Total epochs: {config.num_train_epochs}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  Early stopping: {config.early_stopping_patience} steps")
    print(f"  Monitoring: {config.report_to}")
    print(f"  Checkpoints: Every {config.save_steps} steps")


def get_model_recommendation(model_name: str) -> str:
    """Get recommendation for when to use each model."""
    recommendations = {
        "dccuchile/bert-base-spanish-wwm-uncased": "General purpose Spanish text, balanced performance",
        "pysentimiento/robertuito-base-uncased": "Social media, informal text, Twitter-style content",
        "BSC-TeMU/roberta-base-bne": "Formal text, news articles, academic content",
    }
    
    for key, rec in recommendations.items():
        if key in model_name:
            return rec
    return "Spanish text processing"


def show_extended_emotions():
    """Show the extended emotion labels available for training."""
    print("="*60)
    print("EXTENDED EMOTION LABELS (27 total)")
    print("="*60)
    
    print("The training system supports these emotions:")
    for i, emotion in enumerate(EXTENDED_EMOTION_LABELS, 1):
        print(f"{i:2d}. {emotion}")
    
    print(f"\nTotal: {len(EXTENDED_EMOTION_LABELS)} emotions")
    print("These can be used in multi-label classification.")


def main():
    """Run all training examples."""
    print("🇪🇸 Spanish Emotion Detection - Training Examples")
    print("This script demonstrates different training scenarios.")
    print("Note: Examples are for demonstration and may not run actual training.")
    print()
    
    # Show available emotions
    show_extended_emotions()
    
    # Run examples
    example_1_basic_training()
    example_2_custom_configuration()
    example_3_resume_training()
    example_4_different_models()
    example_5_production_config()
    
    print("\n" + "="*60)
    print("TRAINING EXAMPLES COMPLETED")
    print("="*60)
    print("To run actual training:")
    print("1. Prepare your training data in JSON format:")
    print('   [{"text": "Spanish text", "labels": ["emotion1", "emotion2"]}]')
    print()
    print("2. Use the training script:")
    print("   python scripts/train.py --train-data data/train.json")
    print()
    print("3. Upload to Hugging Face:")
    print("   python scripts/upload_model.py --model-path models/your_model --repo-name username/model-name")
    print()
    print("For more details, check the scripts in the scripts/ directory.")


if __name__ == "__main__":
    main() 