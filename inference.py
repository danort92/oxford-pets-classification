"""
Inference Script - Predict on New Images

This script loads a trained model and makes predictions on new images.

Usage:
    # Binary classification
    python inference.py --model binary_v3 --image path/to/image.jpg
    
    # Multi-class classification
    python inference.py --model multiclass --image path/to/image.jpg
    
    # Transfer learning
    python inference.py --model transfer --image path/to/image.jpg --top-k 5
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from configs.config import get_config
from models.architectures import get_model
from data.datasets import get_transforms
from torchvision import datasets


def load_trained_model(model_type, checkpoint_path, device):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_type: Type of model ('binary_v0', 'binary_v1', etc.)
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    # Create model
    if model_type.startswith('binary'):
        model = get_model(model_type)
    elif model_type == 'multiclass':
        model = get_model('multiclass', num_classes=37)
    elif model_type == 'transfer':
        model = get_model('resnet50', num_classes=37, pretrained=False, freeze_backbone=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def predict_image(model, image_path, transform, device, top_k=1):
    """
    Make prediction on a single image.
    
    Args:
        model: Trained PyTorch model
        image_path: Path to input image
        transform: Image transformation pipeline
        device: Device to run inference on
        top_k: Number of top predictions to return
    
    Returns:
        Predictions (class indices and probabilities)
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get probabilities
    if output.shape[1] == 1:  # Binary classification
        prob = torch.sigmoid(output).item()
        pred_class = 1 if prob > 0.5 else 0
        return [(pred_class, prob if pred_class == 1 else 1-prob)], image
    else:  # Multi-class classification
        probs = torch.softmax(output, dim=1)
        top_probs, top_indices = torch.topk(probs, top_k, dim=1)
        
        predictions = [
            (idx.item(), prob.item()) 
            for idx, prob in zip(top_indices[0], top_probs[0])
        ]
        return predictions, image


def visualize_prediction(image, predictions, class_names=None, binary=False):
    """
    Visualize image with predictions.
    
    Args:
        image: PIL Image
        predictions: List of (class_idx, probability) tuples
        class_names: List of class names
        binary: Whether this is binary classification
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Display image
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image', fontsize=14, fontweight='bold')
    
    # Display predictions
    ax2.axis('off')
    
    if binary:
        pred_class, prob = predictions[0]
        class_name = "Dog" if pred_class == 1 else "Cat"
        
        text = f"Prediction: {class_name}\n"
        text += f"Confidence: {prob*100:.2f}%"
        
        color = 'green' if prob > 0.7 else 'orange' if prob > 0.5 else 'red'
        
        ax2.text(0.5, 0.5, text, 
                ha='center', va='center',
                fontsize=16, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
    else:
        # Multi-class predictions
        text = "Top Predictions:\n\n"
        
        for i, (class_idx, prob) in enumerate(predictions, 1):
            class_name = class_names[class_idx] if class_names else f"Class {class_idx}"
            text += f"{i}. {class_name}\n"
            text += f"   Confidence: {prob*100:.2f}%\n\n"
        
        ax2.text(0.1, 0.5, text,
                ha='left', va='center',
                fontsize=12, fontweight='bold',
                family='monospace')
    
    ax2.set_title('Predictions', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()


def main(args):
    """Main inference function."""
    
    # Determine model configuration
    if args.model.startswith('binary'):
        config = get_config('binary', args.model.replace('binary_', ''))
        experiment_name = f'binary_classification_{args.model.replace("binary_", "")}'
        binary = True
    elif args.model == 'multiclass':
        config = get_config('multiclass')
        experiment_name = 'multiclass_custom_cnn'
        binary = False
    elif args.model == 'transfer':
        config = get_config('transfer')
        experiment_name = 'transfer_learning_resnet50'
        binary = False
    else:
        print(f"❌ Unknown model: {args.model}")
        return
    
    print("="*70)
    print("INFERENCE - PREDICT ON NEW IMAGE")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Image: {args.image}")
    print(f"Device: {config.DEVICE}")
    print("="*70 + "\n")
    
    # Load model checkpoint
    checkpoint_path = config.CHECKPOINT_DIR / experiment_name / 'best_model.pth'
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print(f"Please train the model first using the appropriate script.")
        return
    
    print(f"Loading model from: {checkpoint_path}")
    model = load_trained_model(args.model, checkpoint_path, config.DEVICE)
    print("✅ Model loaded successfully\n")
    
    # Get transforms
    transform = get_transforms(config, augmentation=False)
    
    # Get class names for multi-class
    class_names = None
    if not binary:
        temp_dataset = datasets.OxfordIIITPet(
            root=str(config.DATA_ROOT),
            split="trainval",
            download=True
        )
        class_names = temp_dataset.classes
    
    # Make prediction
    print("Making prediction...")
    predictions, image = predict_image(
        model, args.image, transform, config.DEVICE, top_k=args.top_k
    )
    
    # Print predictions
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    
    if binary:
        pred_class, prob = predictions[0]
        class_name = "Dog" if pred_class == 1 else "Cat"
        print(f"Predicted Class: {class_name}")
        print(f"Confidence: {prob*100:.2f}%")
    else:
        print(f"Top {len(predictions)} Predictions:\n")
        for i, (class_idx, prob) in enumerate(predictions, 1):
            class_name = class_names[class_idx]
            print(f"{i}. {class_name:20s} - {prob*100:.2f}%")
    
    print("="*70 + "\n")
    
    # Visualize
    if args.visualize:
        visualize_prediction(image, predictions, class_names, binary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on new images")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['binary_v0', 'binary_v1', 'binary_v2', 'binary_v3', 
                 'multiclass', 'transfer'],
        help='Model to use for inference'
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show (for multi-class) (default: 5)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Visualize prediction with matplotlib'
    )
    
    args = parser.parse_args()
    main(args)
