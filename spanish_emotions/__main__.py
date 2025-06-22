#!/usr/bin/env python3
"""
Command-line interface for Spanish Emotion Detection Library.
"""

import argparse
import json
import sys
from typing import List

from . import EmotionDetector, EMOTION_LABELS, get_version
from .preprocessing import create_spanish_sample_dataset
from .utils import validate_dataset

def detect_emotion_cli(args):
    """Handle emotion detection from CLI."""
    try:
        # Initialize detector
        detector = EmotionDetector(
            model_path=args.model_path,
            model_key=args.model_key,
            device=args.device
        )
        
        # Process input
        if args.text:
            texts = [args.text]
        elif args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        else:
            # Read from stdin
            texts = [line.strip() for line in sys.stdin if line.strip()]
        
        if not texts:
            print("❌ No text provided. Use --text, --file, or pipe text.", file=sys.stderr)
            return 1
        
        # Detect emotions
        results = []
        for text in texts:
            if args.scores:
                emotion_scores = detector.detect(text, args.threshold, return_scores=True)
                results.append({
                    'text': text,
                    'emotions': [e for e, s in emotion_scores.items() if s >= args.threshold],
                    'scores': emotion_scores
                })
            else:
                emotions = detector.detect(text, args.threshold)
                results.append({
                    'text': text,
                    'emotions': emotions
                })
        
        # Output results
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        else:
            for result in results:
                print(f"Text: {result['text']}")
                print(f"Emotions: {', '.join(result['emotions'])}")
                if 'scores' in result:
                    print("Scores:")
                    for emotion, score in result['scores'].items():
                        print(f"  {emotion}: {score:.3f}")
                print()
        
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

def validate_dataset_cli(args):
    """Handle dataset validation from CLI."""
    try:
        import pandas as pd
        
        # Load dataset
        if args.dataset.endswith('.csv'):
            df = pd.read_csv(args.dataset)
        elif args.dataset.endswith('.json'):
            df = pd.read_json(args.dataset)
        else:
            print("❌ Unsupported file format. Use .csv or .json", file=sys.stderr)
            return 1
        
        # Validate
        results = validate_dataset(
            df, 
            text_column=args.text_column,
            emotion_column=args.emotion_column,
            verbose=True
        )
        
        if args.json:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        
        return 0 if results['valid'] else 1
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

def create_sample_cli(args):
    """Create sample dataset."""
    try:
        df = create_spanish_sample_dataset()
        
        if args.output.endswith('.csv'):
            df.to_csv(args.output, index=False)
        elif args.output.endswith('.json'):
            df.to_json(args.output, orient='records', force_ascii=False, indent=2)
        else:
            print("❌ Unsupported output format. Use .csv or .json", file=sys.stderr)
            return 1
        
        print(f"✅ Sample dataset created: {args.output}")
        print(f"📊 Contains {len(df)} samples")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1

def info_cli(args):
    """Show library information."""
    from . import info
    info()
    return 0

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Spanish Emotion Detection Library CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect emotions in text
  python -m spanish_emotions detect "¡Estoy muy feliz por esta noticia!"
  
  # With confidence scores
  python -m spanish_emotions detect "Me siento triste" --scores
  
  # From file
  python -m spanish_emotions detect --file texts.txt --json
  
  # From stdin
  echo "Tengo miedo" | python -m spanish_emotions detect
  
  # Validate dataset
  python -m spanish_emotions validate dataset.csv
  
  # Create sample dataset
  python -m spanish_emotions sample sample_data.csv
        """
    )
    
    parser.add_argument('--version', action='version', version=f'Spanish Emotions {get_version()}')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Detect emotions in text')
    detect_parser.add_argument('text', nargs='?', help='Text to analyze')
    detect_parser.add_argument('--file', '-f', help='File with texts (one per line)')
    detect_parser.add_argument('--model-path', help='Path to fine-tuned model')
    detect_parser.add_argument('--model-key', default='beto', 
                              choices=['beto', 'robertuito', 'roberta-bne', 'beto-cased'],
                              help='Pre-configured model to use')
    detect_parser.add_argument('--device', choices=['cpu', 'cuda'], help='Device to use')
    detect_parser.add_argument('--threshold', type=float, default=0.5, 
                              help='Confidence threshold (0.0-1.0)')
    detect_parser.add_argument('--scores', action='store_true', 
                              help='Include confidence scores')
    detect_parser.add_argument('--json', action='store_true', 
                              help='Output in JSON format')
    detect_parser.set_defaults(func=detect_emotion_cli)
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate emotion dataset')
    validate_parser.add_argument('dataset', help='Path to dataset file (.csv or .json)')
    validate_parser.add_argument('--text-column', default='text', help='Name of text column')
    validate_parser.add_argument('--emotion-column', default='emotions', help='Name of emotion column')
    validate_parser.add_argument('--json', action='store_true', help='Output in JSON format')
    validate_parser.set_defaults(func=validate_dataset_cli)
    
    # Sample command
    sample_parser = subparsers.add_parser('sample', help='Create sample dataset')
    sample_parser.add_argument('output', help='Output file path (.csv or .json)')
    sample_parser.set_defaults(func=create_sample_cli)
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show library information')
    info_parser.set_defaults(func=info_cli)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    return args.func(args)

if __name__ == '__main__':
    sys.exit(main()) 