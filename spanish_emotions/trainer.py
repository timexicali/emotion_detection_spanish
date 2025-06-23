"""
Training module for Spanish emotion detection models.
Supports fine-tuning, transfer learning, and multi-label classification.
"""

import json
import os
import logging
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, hamming_loss
import pandas as pd

from .labels import EMOTION_LABELS
from .preprocessing import normalize_spanish_text

logger = logging.getLogger(__name__)

# Extended emotion labels for multi-label classification (27 labels)
EXTENDED_EMOTION_LABELS = [
    # Basic emotions (7)
    "alegría", "tristeza", "enojo", "miedo", "sorpresa", "desagrado", "neutral",
    
    # Extended emotions (20)
    "amor", "optimismo", "gratitud", "esperanza", "satisfacción",  # positive emotions
    "frustración", "ansiedad", "decepción", "culpa", "vergüenza",  # negative emotions
    "curiosidad", "confusión", "nostalgia", "orgullo", "celos",    # complex emotions
    "emoción", "tranquilidad", "excitación", "aburrimiento", "compasión"  # additional states
]

@dataclass
class EmotionTrainingConfig:
    """Configuration for emotion detection training."""
    
    # Model configuration
    base_model: str = "dccuchile/bert-base-spanish-wwm-uncased"
    num_labels: int = 27
    max_length: int = 512
    
    # Training configuration
    output_dir: str = "models/spanish_emotion_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Evaluation and saving
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_f1"
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Logging
    logging_steps: int = 100
    report_to: Optional[str] = None  # "wandb", "tensorboard", or None
    
    # Multi-label specific
    multi_label: bool = True
    problem_type: str = "multi_label_classification"


class SpanishEmotionDataset(Dataset):
    """Dataset for Spanish emotion detection."""
    
    def __init__(
        self, 
        texts: List[str], 
        labels: List[List[str]], 
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        label_encoder: Optional[MultiLabelBinarizer] = None
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_encoder = label_encoder
        
        # Encode labels
        if self.label_encoder is None:
            self.label_encoder = MultiLabelBinarizer()
            self.encoded_labels = self.label_encoder.fit_transform(labels)
        else:
            self.encoded_labels = self.label_encoder.transform(labels)
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # Preprocess text
        text = normalize_spanish_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.encoded_labels[idx], dtype=torch.float)
        }


class SpanishEmotionTrainer:
    """Trainer for Spanish emotion detection models."""
    
    def __init__(self, config: EmotionTrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.label_encoder = None
        self.trainer = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        
    def setup_model_and_tokenizer(self):
        """Initialize the model and tokenizer."""
        logger.info(f"Loading tokenizer from {self.config.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
        
        logger.info(f"Loading model from {self.config.base_model}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.base_model,
            num_labels=self.config.num_labels,
            problem_type=self.config.problem_type
        )
        
        # Ensure we have padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def load_data(self, data_path: str) -> Tuple[List[str], List[List[str]]]:
        """
        Load training data from JSON file.
        
        Expected format:
        [
            {"text": "Estoy feliz de verte", "labels": ["alegría"]},
            {"text": "Me siento triste y confundido", "labels": ["tristeza", "confusión"]}
        ]
        """
        logger.info(f"Loading data from {data_path}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = []
        labels = []
        
        for item in data:
            texts.append(item['text'])
            
            # Handle both single string and list of strings for labels
            item_labels = item['labels']
            if isinstance(item_labels, str):
                item_labels = [item_labels]
            
            # Validate labels
            valid_labels = []
            for label in item_labels:
                if label in EXTENDED_EMOTION_LABELS:
                    valid_labels.append(label)
                else:
                    logger.warning(f"Unknown label '{label}' found, skipping")
            
            if not valid_labels:
                valid_labels = ["neutral"]  # Default to neutral if no valid labels
                
            labels.append(valid_labels)
        
        logger.info(f"Loaded {len(texts)} samples")
        return texts, labels
    
    def create_datasets(
        self, 
        train_texts: List[str], 
        train_labels: List[List[str]],
        eval_texts: Optional[List[str]] = None,
        eval_labels: Optional[List[List[str]]] = None
    ) -> Tuple[SpanishEmotionDataset, Optional[SpanishEmotionDataset]]:
        """Create training and evaluation datasets."""
        
        # Setup label encoder
        self.label_encoder = MultiLabelBinarizer()
        all_labels = train_labels + (eval_labels if eval_labels else [])
        self.label_encoder.fit(all_labels)
        
        logger.info(f"Label encoder classes: {self.label_encoder.classes_}")
        
        # Create datasets
        train_dataset = SpanishEmotionDataset(
            train_texts, 
            train_labels, 
            self.tokenizer, 
            self.config.max_length,
            self.label_encoder
        )
        
        eval_dataset = None
        if eval_texts and eval_labels:
            eval_dataset = SpanishEmotionDataset(
                eval_texts, 
                eval_labels, 
                self.tokenizer, 
                self.config.max_length,
                self.label_encoder
            )
        
        return train_dataset, eval_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics for multi-label classification."""
        predictions, labels = eval_pred
        predictions = torch.sigmoid(torch.tensor(predictions)).numpy()
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, binary_predictions, average='macro', zero_division=0
        )
        
        accuracy = accuracy_score(labels, binary_predictions)
        hamming = hamming_loss(labels, binary_predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'hamming_loss': hamming
        }
    
    def train(
        self, 
        train_data_path: str,
        eval_data_path: Optional[str] = None,
        resume_from_checkpoint: Optional[str] = None
    ):
        """Train the emotion detection model."""
        
        # Load data first to determine actual number of labels
        train_texts, train_labels = self.load_data(train_data_path)
        
        eval_texts, eval_labels = None, None
        if eval_data_path:
            eval_texts, eval_labels = self.load_data(eval_data_path)
        
        # Create datasets to determine actual number of labels
        temp_label_encoder = MultiLabelBinarizer()
        all_labels = train_labels + (eval_labels if eval_labels else [])
        temp_label_encoder.fit(all_labels)
        actual_num_labels = len(temp_label_encoder.classes_)
        
        logger.info(f"Found {actual_num_labels} unique labels in data")
        logger.info(f"Labels: {list(temp_label_encoder.classes_)}")
        
        # Now setup model with correct number of labels
        self.config.num_labels = actual_num_labels
        self.setup_model_and_tokenizer()
        
        # Create datasets properly
        train_dataset, eval_dataset = self.create_datasets(
            train_texts, train_labels, eval_texts, eval_labels
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy if eval_dataset else "no",
            eval_steps=self.config.eval_steps,
            save_strategy=self.config.save_strategy,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            report_to=self.config.report_to,
            push_to_hub=False,
            dataloader_pin_memory=False,
        )
        
        # Setup callbacks
        callbacks = []
        if eval_dataset:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics if eval_dataset else None,
            callbacks=callbacks,
        )
        
        # Start training
        logger.info("Starting training...")
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save final model
        logger.info(f"Saving model to {self.config.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save label encoder
        import pickle
        with open(os.path.join(self.config.output_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save emotion labels
        labels_path = os.path.join(self.config.output_dir, 'emotion_labels.json')
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(list(self.label_encoder.classes_), f, ensure_ascii=False, indent=2)
        
        logger.info("Training completed!")
        
        return self.trainer


def create_sample_training_data(output_path: str, num_samples: int = 100):
    """Create sample training data for testing."""
    
    sample_data = []
    
    # Sample texts with corresponding emotions
    samples = [
        {"text": "¡Estoy súper feliz por esta noticia increíble!", "labels": ["alegría", "emoción"]},
        {"text": "Me siento muy triste y deprimido por lo que pasó", "labels": ["tristeza", "decepción"]},
        {"text": "¡Qué rabia me da esta situación injusta!", "labels": ["enojo", "frustración"]},
        {"text": "Tengo mucho miedo de lo que pueda pasar", "labels": ["miedo", "ansiedad"]},
        {"text": "¡No puedo creer lo que acaba de suceder!", "labels": ["sorpresa", "confusión"]},
        {"text": "Esto me da mucho asco y repulsión", "labels": ["desagrado"]},
        {"text": "Hoy es un día normal, sin nada especial", "labels": ["neutral", "tranquilidad"]},
        {"text": "Te amo mucho mi vida", "labels": ["amor", "alegría"]},
        {"text": "Tengo muchas esperanzas en el futuro", "labels": ["esperanza", "optimismo"]},
        {"text": "Me siento muy culpable por lo que hice", "labels": ["culpa", "vergüenza"]},
        {"text": "Estoy muy orgulloso de mis logros", "labels": ["orgullo", "satisfacción"]},
        {"text": "Siento mucha nostalgia por mi infancia", "labels": ["nostalgia", "tristeza"]},
        {"text": "Estoy muy aburrido en casa sin hacer nada", "labels": ["aburrimiento", "neutral"]},
        {"text": "Me da mucha curiosidad saber más de esto", "labels": ["curiosidad", "emoción"]},
        {"text": "Siento compasión por las personas necesitadas", "labels": ["compasión", "amor"]},
    ]
    
    # Repeat and vary the samples
    for i in range(num_samples):
        sample = samples[i % len(samples)]
        sample_data.append(sample)
    
    # Save to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Created sample training data: {output_path}")
    return output_path 