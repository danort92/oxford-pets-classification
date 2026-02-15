"""
Task 2: Transfer Learning with ResNet50

This script implements two-stage transfer learning:
- Stage 1: Train only the custom classifier head (backbone frozen)
- Stage 2: Fine-tune the last residual block (layer4) of ResNet50

Usage:
    python task3_transfer_learning.py --stage1-epochs 15 --stage2-epochs 10
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

from configs.config import TransferLearningConfig, GradCAMConfig
from models.architectures import get_model, count_parameters
from utils.data_utils import prepare_multiclass_dataloaders, get_predictions
from utils.trainer import MultiClassTrainer
from utils.visualization import (
    plot_transfer_learning_curves,
    plot_confusion_matrix,
    print_classification_report,
    visualize_gradcam_grid,
    plot_sample_predictions
)


def main(args):
    """Main training function for transfer learning."""
    
    # Get configuration
    config = TransferLearningConfig
    config.EPOCHS_STAGE1 = args.stage1_epochs
    config.EPOCHS_STAGE2 = args.stage2_epochs
    config.create_directories()
    
    print("="*70)
    print(f"TASK 2: TRANSFER LEARNING WITH RESNET50")
    print("="*70)
    print(f"Device: {config.DEVICE}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print(f"Stage 1 Epochs: {config.EPOCHS_STAGE1} (classifier only)")
    print(f"Stage 2 Epochs: {config.EPOCHS_STAGE2} (fine-tune {config.UNFREEZE_LAYERS})")
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
    print("Creating ResNet50 with pretrained ImageNet weights...")
    model = get_model(
        'resnet50',
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=True
    )
    model = model.to(config.DEVICE)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (Stage 1): {trainable_params:,}\n")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    # ==================== STAGE 1 ====================
    print("="*70)
    print("STAGE 1: Training classifier only (backbone frozen)")
    print("="*70 + "\n")
    
    optimizer_stage1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE_STAGE1,
        weight_decay=config.WEIGHT_DECAY
    )
    
    save_dir = config.CHECKPOINT_DIR / config.EXPERIMENT_NAME
    trainer = MultiClassTrainer(
        model=model,
        device=config.DEVICE,
        criterion=criterion,
        optimizer=optimizer_stage1,
        save_dir=save_dir
    )
    
    # Train Stage 1
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.EPOCHS_STAGE1,
        verbose=True
    )
    
    # ==================== STAGE 2 ====================
    print("\n" + "="*70)
    print(f"STAGE 2: Fine-tuning {config.UNFREEZE_LAYERS}")
    print("="*70 + "\n")
    
    # Unfreeze specified layers
    model.unfreeze_layers(config.UNFREEZE_LAYERS)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Trainable parameters (Stage 2): {trainable_params:,}\n")
    
    # New optimizer for stage 2 with lower learning rate
    optimizer_stage2 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE_STAGE2,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Update trainer optimizer
    trainer.optimizer = optimizer_stage2
    
    # Continue training (Stage 2)
    for epoch in range(config.EPOCHS_STAGE2):
        train_loss, train_acc = trainer.train_epoch(train_loader)
        val_loss, val_acc = trainer.evaluate(val_loader)
        
        # Update history
        trainer.history['train_loss'].append(train_loss)
        trainer.history['train_acc'].append(train_acc)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > trainer.best_val_acc:
            trainer.best_val_acc = val_acc
            trainer.best_epoch = config.EPOCHS_STAGE1 + epoch + 1
            trainer.save_checkpoint('best_model.pth', trainer.best_epoch)
        
        print(
            f"Epoch {epoch+1}/{config.EPOCHS_STAGE2} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
        )
    
    # ==================== EVALUATION ====================
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70 + "\n")
    
    # Evaluate on test set
    test_loss, test_acc = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {trainer.best_val_acc:.4f} (Epoch {trainer.best_epoch})")
    print("="*70 + "\n")
    
    # Get predictions for confusion matrix
    if args.confusion_matrix:
        print("Generating confusion matrix...")
        y_pred, y_true = get_predictions(model, test_loader, config.DEVICE)
        
        cm_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_confusion_matrix.png"
        plot_confusion_matrix(
            y_true, y_pred,
            class_names=class_names,
            save_path=cm_path,
            normalize=True
        )
        
        print_classification_report(y_true, y_pred, class_names=class_names)
    
    # Plot training curves
    curves_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_curves.png"
    plot_transfer_learning_curves(
        trainer.history,
        stage1_epochs=config.EPOCHS_STAGE1,
        save_path=curves_path
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
    
    # Grad-CAM visualization
    if args.gradcam:
        print("Generating Grad-CAM visualizations...")
        gradcam_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_gradcam.png"
        
        gradcam_config = GradCAMConfig()
        visualize_gradcam_grid(
            model.resnet,  # Use the ResNet model directly
            test_loader.dataset,
            target_layer=gradcam_config.TARGET_LAYER,
            num_samples=12,
            class_names=class_names,
            save_path=gradcam_path
        )
    
    # Save results
    results = {
        'model': 'ResNet50_Transfer',
        'num_classes': config.NUM_CLASSES,
        'stage1_epochs': config.EPOCHS_STAGE1,
        'stage2_epochs': config.EPOCHS_STAGE2,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'best_val_acc': trainer.best_val_acc,
        'best_epoch': trainer.best_epoch,
        'total_params': total_params,
        'trainable_params_final': trainable_params
    }
    
    import json
    results_path = config.OUTPUT_DIR / f"{config.EXPERIMENT_NAME}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {results_path}")
    print(f"Training curves saved to: {curves_path}")
    print(f"Model checkpoint saved to: {save_dir / 'best_model.pth'}")
    
    return trainer.history, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task 2: Transfer Learning with ResNet50"
    )
    parser.add_argument(
        '--stage1-epochs',
        type=int,
        default=15,
        help='Number of epochs for stage 1 (classifier only) (default: 15)'
    )
    parser.add_argument(
        '--stage2-epochs',
        type=int,
        default=10,
        help='Number of epochs for stage 2 (fine-tuning) (default: 10)'
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
    parser.add_argument(
        '--gradcam',
        action='store_true',
        help='Generate Grad-CAM visualizations'
    )
    
    args = parser.parse_args()
    main(args)
