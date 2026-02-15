"""
Task 1 - Step 1: Binary Classification (Cat vs Dog)

This script trains and evaluates CNN models of increasing complexity:
- v0: Simple 2-layer CNN
- v1: 4-layer CNN with AdaptiveAvgPool
- v2: VGG-like with BatchNorm and Dropout
- v3: VGG-like with data augmentation

Usage:
    python task1_binary_classification.py --model v3 --epochs 20
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

from configs.config import get_config
from models.architectures import get_model, count_parameters
from utils.data_utils import prepare_binary_dataloaders
from utils.trainer import BinaryTrainer
from utils.visualization import plot_training_curves


def main(args):
    """Main training function for binary classification."""
    
    # Get configuration
    config = get_config('binary', args.model)
    config.EPOCHS = args.epochs
    config.create_directories()
    
    print("="*70)
    print(f"TASK 1 - STEP 1: BINARY CLASSIFICATION")
    print(f"Model Version: {args.model}")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Learning Rate: {config.LEARNING_RATE}")
    print(f"Data Augmentation: {args.model == 'v3'}")
    print("="*70 + "\n")
    
    # Prepare data
    print("Loading datasets...")
    use_augmentation = (args.model == 'v3')
    train_loader, val_loader, test_loader, id_to_binary = prepare_binary_dataloaders(
        config, use_augmentation=use_augmentation
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}\n")
    
    # Create model
    print(f"Creating model: BinaryCNN_{args.model}")
    model = get_model(f'binary_{args.model}')
    model = model.to(config.DEVICE)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}\n")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Create trainer
    save_dir = config.CHECKPOINT_DIR / config.EXPERIMENT_NAME
    trainer = BinaryTrainer(
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
    
    # Plot training curves
    output_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_curves.png"
    plot_training_curves(
        history,
        save_path=output_path,
        title=f"Binary Classification - Model {args.model.upper()}"
    )
    
    # Save results
    results = {
        'model': args.model,
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
    
    print(f"Results saved to: {results_path}")
    print(f"Training curves saved to: {output_path}")
    print(f"Model checkpoint saved to: {save_dir / 'best_model.pth'}")
    
    return history, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Task 1 Step 1: Binary Classification")
    parser.add_argument(
        '--model',
        type=str,
        default='v3',
        choices=['v0', 'v1', 'v2', 'v3'],
        help='Model version to train (default: v3)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='Number of training epochs (default: 20)'
    )
    
    args = parser.parse_args()
    main(args)
