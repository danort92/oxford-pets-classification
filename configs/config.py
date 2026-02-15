"""
Configuration file for Oxford-IIIT Pet Classification experiments.

This module centralizes all hyperparameters, paths, and settings for:
- Task 1 Step 1: Binary classification (Cat vs Dog) with custom CNNs
- Task 1 Step 2: Multi-class classification (37 breeds) with custom CNN
- Task 2: Multi-class classification with transfer learning (ResNet50)
"""

import torch
from pathlib import Path


class BaseConfig:
    """Base configuration shared across all experiments."""
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data paths
    DATA_ROOT = Path("./data")
    OUTPUT_DIR = Path("./outputs")
    CHECKPOINT_DIR = Path("./checkpoints")
    LOG_DIR = Path("./logs")
    
    # Dataset parameters
    TRAIN_VAL_SPLIT = 0.7  # 70% train, 30% validation
    NUM_WORKERS = 2
    PIN_MEMORY = True
    
    # ImageNet normalization (standard for pretrained models)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    # Image size
    IMAGE_SIZE = 224
    
    # Random seed for reproducibility
    SEED = 42
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist."""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


class BinaryClassificationConfig(BaseConfig):
    """Configuration for Task 1 Step 1: Binary Classification (Cat vs Dog)."""
    
    # Model selection: 'v0', 'v1', 'v2', 'v3'
    MODEL_VERSION = 'v3'
    
    # Training hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Optimizer
    OPTIMIZER = 'adamw'
    
    # Data augmentation (only for v3)
    USE_AUGMENTATION = True  # Set to False for v0, v1, v2
    
    # Model architectures available
    MODELS = {
        'v0': {
            'description': 'Simple 2-layer CNN',
            'channels': [16, 32],
            'use_batch_norm': False,
            'use_dropout': False
        },
        'v1': {
            'description': '4-layer CNN with AdaptiveAvgPool',
            'channels': [16, 32, 64, 128],
            'use_batch_norm': False,
            'use_dropout': False
        },
        'v2': {
            'description': 'VGG-like CNN with BatchNorm and Dropout',
            'channels': [32, 64, 128, 256, 512],
            'use_batch_norm': True,
            'use_dropout': True,
            'dropout_conv': 0.25,
            'dropout_fc': 0.5
        },
        'v3': {
            'description': 'VGG-like CNN with data augmentation',
            'channels': [32, 64, 128, 256, 512],
            'use_batch_norm': True,
            'use_dropout': True,
            'dropout_conv': 0.25,
            'dropout_fc': 0.5
        }
    }
    
    # Experiment name
    EXPERIMENT_NAME = f'binary_classification_{MODEL_VERSION}'


class MultiClassConfig(BaseConfig):
    """Configuration for Task 1 Step 2: Multi-class Classification (37 breeds)."""
    
    # Training hyperparameters
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Optimizer
    OPTIMIZER = 'adamw'
    
    # Number of classes (37 breeds)
    NUM_CLASSES = 37
    
    # Data augmentation
    USE_AUGMENTATION = True
    
    # Model architecture (same as binary v2/v3)
    CHANNELS = [32, 64, 128, 256, 512]
    DROPOUT_CONV = 0.25
    DROPOUT_FC = 0.5
    
    # Experiment name
    EXPERIMENT_NAME = 'multiclass_custom_cnn'


class TransferLearningConfig(BaseConfig):
    """Configuration for Task 2: Transfer Learning with ResNet50."""
    
    # Training hyperparameters
    BATCH_SIZE = 64
    
    # Stage 1: Train only classifier
    EPOCHS_STAGE1 = 15
    LEARNING_RATE_STAGE1 = 1e-3
    
    # Stage 2: Fine-tune last layers
    EPOCHS_STAGE2 = 10
    LEARNING_RATE_STAGE2 = 1e-4
    
    WEIGHT_DECAY = 1e-4
    
    # Optimizer
    OPTIMIZER = 'adamw'
    
    # Number of classes
    NUM_CLASSES = 37
    
    # Data augmentation
    USE_AUGMENTATION = True
    
    # Transfer learning settings
    PRETRAINED_MODEL = 'resnet50'
    FREEZE_BACKBONE = True  # Initially freeze all layers except classifier
    UNFREEZE_LAYERS = ['layer4']  # Layers to unfreeze in stage 2
    
    # Custom classifier architecture
    CLASSIFIER_HIDDEN_DIM = 512
    CLASSIFIER_DROPOUT = 0.5
    
    # Experiment name
    EXPERIMENT_NAME = 'transfer_learning_resnet50'


class GradCAMConfig:
    """Configuration for Grad-CAM visualization."""
    
    # Target layer for Grad-CAM (ResNet50)
    TARGET_LAYER = 'layer4'
    
    # Visualization grid
    NUM_ROWS = 3
    NUM_COLS = 4
    
    # Overlay parameters
    HEATMAP_ALPHA = 0.4
    IMAGE_ALPHA = 0.6
    
    # Colormap
    COLORMAP = 'jet'


# Export commonly used configs
def get_config(task='binary', model_version='v3'):
    """
    Get configuration for a specific task.
    
    Args:
        task: 'binary', 'multiclass', or 'transfer'
        model_version: For binary classification, choose from 'v0', 'v1', 'v2', 'v3'
    
    Returns:
        Configuration class
    """
    if task == 'binary':
        config = BinaryClassificationConfig
        config.MODEL_VERSION = model_version
        config.EXPERIMENT_NAME = f'binary_classification_{model_version}'
        return config
    elif task == 'multiclass':
        return MultiClassConfig
    elif task == 'transfer':
        return TransferLearningConfig
    else:
        raise ValueError(f"Unknown task: {task}")
