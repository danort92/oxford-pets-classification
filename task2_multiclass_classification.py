"""
Task 1 - Step 2: Multi-class Classification (37 breeds)

This script trains a custom CNN to classify 37 different cat and dog breeds
using the best architecture from Task 1 Step 1 (v2/v3 - VGG-like with regularization).

Usage:
    python task2_multiclass_classification.py --epochs 100
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import torch.optim as optim

from configs.config import MultiClassConfig
from models.architectures import get_model, count_parameters
from utils.data_utils import prepare_multiclass_dataloaders, get_predictions
from utils.trainer import MultiClassTrainer
from utils.visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    print_classification_report,
    plot_sample_predictions
)


def main(args):
    """Main training function for multi-class classification."""
    
    # Get configuration
    config = MultiClassConfig
    config.EPOCHS = args.epochs
    config.create_directories()
    
    print("="*70)
    print(f"TASK 1 - STEP 2: MULTI-CLASS CLASSIFICATION (37 BREEDS)")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Number of Classes: {config.NUM_CLASSES}")
    print("="*70 + "\n")
    
    # Prepare data
    print("Loading datasets...")
    train_loader, val_loader, test_loader, class_names = prepare_multiclass_dataloaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {config.NUM_CLASSES}\n")
    
    # Create model
    print("Creating MultiClassCNN model...")
    model = get_model('multiclass', num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create trainer
    save_dir = config.CHECKPOINT_DIR / config.EXPERIMENT_NAME
    trainer = MultiClassTrainer(
        model=model,
        device=config.DEVICE,
        criterion=criterion,
        optimizer=optimizer,
        save_dir=save_dir
    )
    
    # Train
    print("Starting training...\n")
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS,
        verbose=True
    )
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"\n{'='*70}")
    print(f"TEST RESULTS")
    print(f"{'='*70}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"{'='*70}\n")
    
    # Get predictions for confusion matrix
    if args.confusion_matrix:
        print("Generating confusion matrix...")
        y_pred, y_true = get_predictions(model, test_loader, config.DEVICE)
        
        # Plot confusion matrix
        cm_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_confusion_matrix.png"
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=class_names,
            save_path=cm_path,
            normalize=True
        )
        
        # Print classification report
        print_classification_report(y_true, y_pred, class_names=class_names)
    
    # Plot training curves
    curves_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_curves.png"
    plot_training_curves(
        history,
        save_path=curves_path,
        title="Multi-Class Classification - Custom CNN"
    )
    
    # Plot sample predictions
    if args.sample_predictions:
        print("Generating sample predictions...")
        samples_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_samples.png"
        plot_sample_predictions(
            model,
            test_loader.dataset,
            config.DEVICE,
            num_samples=16,
            class_names=class_names,
            save_path=samples_path
        )
    
    # Save results
    results = {
        'model': 'MultiClassCNN',
        'num_classes': config.NUM_CLASSES,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch,
        'total_params': total_params,
        'trainable_params': trainable_params
    }
    
    import json
    results_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_path}")
    print(f"Training curves saved to: {curves_path}")
    print(f"Model checkpoint saved to: {save_dir / 'best_model.pth'}")
    
    return history, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 1 Step 2: Multi-class Classification"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--confusion-matrix',
        action='store_true',
        help='Generate confusion matrix on test set'
    )
    parser.add_argument(
        '--sample-predictions',
        action='store_true',
        help='Visualize sample predictions'
    )
    
    args = parser.parse_args()
    main(args)
