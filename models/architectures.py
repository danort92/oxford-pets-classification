"""
CNN architectures for Oxford-IIIT Pet classification.

This module contains:
- BinaryCNN_v0: Simple 2-layer CNN
- BinaryCNN_v1: 4-layer CNN with AdaptiveAvgPool
- BinaryCNN_v2: VGG-like CNN with BatchNorm and Dropout
- MultiClassCNN: Custom CNN for 37-class classification
- ResNet50Transfer: ResNet50 with custom classifier for transfer learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BinaryCNN_v0(nn.Module):
    """
    Simple 2-layer CNN for binary classification.
    
    Architecture:
        - Conv1: 3 → 16 channels
        - Conv2: 16 → 32 channels
        - Fully connected: 32*56*56 → 1
    
    No batch normalization or dropout.
    """
    
    def __init__(self):
        super(BinaryCNN_v0, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 → 56
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 1)  # Binary logit
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BinaryCNN_v1(nn.Module):
    """
    4-layer CNN with AdaptiveAvgPool for binary classification.
    
    Architecture:
        - 4 convolutional blocks with increasing channels
        - AdaptiveAvgPool to reduce spatial dimensions
        - Fully connected: 128 → 1
    
    Improvements over v0:
        - Deeper network (4 layers)
        - AdaptiveAvgPool for dimension reduction
    """
    
    def __init__(self):
        super(BinaryCNN_v1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 → 56
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 → 28
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))  # → (128, 1, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 1)  # Binary logit
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class BinaryCNN_v2(nn.Module):
    """
    VGG-like CNN with BatchNorm and Dropout for binary classification.
    
    Architecture:
        - 5 convolutional blocks (each with 2 conv layers)
        - BatchNormalization after each conv
        - Dropout for regularization
        - 3-layer classifier with dropout
    
    Improvements over v1:
        - Batch normalization for stable training
        - Dropout to prevent overfitting
        - Double convolutions per block (VGG-style)
        - Deeper classifier
    """
    
    def __init__(self):
        super(BinaryCNN_v2, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224 → 112
            nn.Dropout(0.25),
            
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112 → 56
            nn.Dropout(0.25),
            
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56 → 28
            nn.Dropout(0.25),
            
            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28 → 14
            nn.Dropout(0.25),
            
            # Block 5: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14 → 7
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # Binary logit
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class MultiClassCNN(nn.Module):
    """
    VGG-like CNN for 37-class breed classification.
    
    Similar architecture to BinaryCNN_v2 but with:
        - Output layer adapted for 37 classes
        - CrossEntropyLoss compatible output (no sigmoid)
    
    Args:
        num_classes: Number of output classes (default: 37)
    """
    
    def __init__(self, num_classes=37):
        super(MultiClassCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1: 3 → 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 2: 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 3: 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 4: 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Block 5: 256 → 512
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # Multi-class output
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResNet50Transfer(nn.Module):
    """
    ResNet50 with custom classifier for transfer learning.
    
    Features:
        - Pretrained ResNet50 backbone (ImageNet weights)
        - Option to freeze/unfreeze layers
        - Custom classifier head
    
    Args:
        num_classes: Number of output classes (default: 37)
        pretrained: Use ImageNet pretrained weights (default: True)
        freeze_backbone: Freeze all backbone layers initially (default: True)
    """
    
    def __init__(self, num_classes=37, pretrained=True, freeze_backbone=True):
        super(ResNet50Transfer, self).__init__()
        
        # Load pretrained ResNet50
        if pretrained:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
            self.resnet = models.resnet50(weights=weights)
        else:
            self.resnet = models.resnet50(weights=None)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)
    
    def unfreeze_layers(self, layer_names):
        """
        Unfreeze specific layers for fine-tuning.
        
        Args:
            layer_names: List of layer names to unfreeze (e.g., ['layer4'])
        """
        for name, param in self.resnet.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = True
                    break
    
    def get_layer(self, layer_name):
        """
        Get a specific layer by name (useful for Grad-CAM).
        
        Args:
            layer_name: Name of the layer (e.g., 'layer4')
        
        Returns:
            The requested layer module
        """
        for name, module in self.resnet.named_modules():
            if name == layer_name:
                return module
        raise ValueError(f"Layer {layer_name} not found in model")


def get_model(model_type, num_classes=None, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('binary_v0', 'binary_v1', 'binary_v2', 
                    'multiclass', 'resnet50')
        num_classes: Number of classes (for multiclass and resnet50)
        **kwargs: Additional arguments for specific models
    
    Returns:
        PyTorch model
    """
    if model_type == 'binary_v0':
        return BinaryCNN_v0()
    elif model_type == 'binary_v1':
        return BinaryCNN_v1()
    elif model_type == 'binary_v2' or model_type == 'binary_v3':
        return BinaryCNN_v2()
    elif model_type == 'multiclass':
        return MultiClassCNN(num_classes=num_classes or 37)
    elif model_type == 'resnet50':
        return ResNet50Transfer(
            num_classes=num_classes or 37,
            pretrained=kwargs.get('pretrained', True),
            freeze_backbone=kwargs.get('freeze_backbone', True)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """
    Count total and trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Tuple of (total_params, trainable_params)
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
