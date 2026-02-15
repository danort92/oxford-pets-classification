"""
Data loading utilities for Oxford-IIIT Pet classification.

This module provides functions to:
- Load and prepare datasets
- Create data loaders
- Handle train/val/test splits
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from data.datasets import (
    BinaryOxfordPets, 
    BinaryOxfordPetsSubset,
    MultiClassOxfordPets,
    load_class_mappings,
    get_transforms,
    get_simple_transforms
)


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def prepare_binary_dataloaders(config, use_augmentation=True):
    """
    Prepare dataloaders for binary classification (Cat vs Dog).
    
    Args:
        config: Configuration object
        use_augmentation: Whether to use data augmentation for training
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, id_to_binary)
    """
    set_seed(config.SEED)
    
    # Load class mappings
    id_to_breed, id_to_binary = load_class_mappings(config.DATA_ROOT)
    
    # Choose transforms based on augmentation flag
    if use_augmentation:
        train_transform = get_transforms(config, augmentation=True)
        val_test_transform = get_transforms(config, augmentation=False)
    else:
        # For v0, v1, v2: simple transforms without normalization
        train_transform = get_simple_transforms(config.IMAGE_SIZE)
        val_test_transform = get_simple_transforms(config.IMAGE_SIZE)
    
    # Load base datasets (without transforms)
    base_trainval = datasets.OxfordIIITPet(
        root=str(config.DATA_ROOT),
        split="trainval",
        transform=None,
        download=True
    )
    
    base_test = datasets.OxfordIIITPet(
        root=str(config.DATA_ROOT),
        split="test",
        transform=None,
        download=True
    )
    
    # Train-validation split
    train_size = int(config.TRAIN_VAL_SPLIT * len(base_trainval))
    val_size = len(base_trainval) - train_size
    
    train_indices_obj, val_indices_obj = random_split(
        range(len(base_trainval)), [train_size, val_size]
    )
    
    # Create datasets with appropriate transforms
    if use_augmentation:
        # Use custom subset with transforms
        train_dataset = BinaryOxfordPetsSubset(
            base_trainval,
            train_indices_obj.indices,
            id_to_binary,
            train_transform
        )
        
        val_dataset = BinaryOxfordPetsSubset(
            base_trainval,
            val_indices_obj.indices,
            id_to_binary,
            val_test_transform
        )
    else:
        # Use simple wrapper (for v0, v1, v2)
        base_trainval_transformed = datasets.OxfordIIITPet(
            root=str(config.DATA_ROOT),
            split="trainval",
            transform=train_transform,
            download=False
        )
        
        trainval_binary = BinaryOxfordPets(base_trainval_transformed, id_to_binary)
        
        # Split the wrapped dataset
        train_dataset, val_dataset = random_split(
            trainval_binary, [train_size, val_size]
        )
    
    # Test dataset
    base_test_transformed = datasets.OxfordIIITPet(
        root=str(config.DATA_ROOT),
        split="test",
        transform=val_test_transform,
        download=False
    )
    
    test_dataset = BinaryOxfordPets(base_test_transformed, id_to_binary)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader, id_to_binary


def prepare_multiclass_dataloaders(config):
    """
    Prepare dataloaders for 37-class breed classification.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names)
    """
    set_seed(config.SEED)
    
    # Get transforms
    train_transform = get_transforms(config, augmentation=True)
    val_test_transform = get_transforms(config, augmentation=False)
    
    # Load base datasets (without transforms)
    base_trainval = datasets.OxfordIIITPet(
        root=str(config.DATA_ROOT),
        split="trainval",
        transform=None,
        download=True
    )
    
    base_test = datasets.OxfordIIITPet(
        root=str(config.DATA_ROOT),
        split="test",
        transform=None,
        download=True
    )
    
    # Get class names
    class_names = base_trainval.classes
    
    # Train-validation split
    train_size = int(config.TRAIN_VAL_SPLIT * len(base_trainval))
    val_size = len(base_trainval) - train_size
    
    train_indices_obj, val_indices_obj = random_split(
        range(len(base_trainval)), [train_size, val_size]
    )
    
    # Create datasets
    train_dataset = MultiClassOxfordPets(
        base_trainval,
        train_indices_obj.indices,
        train_transform
    )
    
    val_dataset = MultiClassOxfordPets(
        base_trainval,
        val_indices_obj.indices,
        val_test_transform
    )
    
    test_dataset = MultiClassOxfordPets(
        base_test,
        transform=val_test_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader, test_loader, class_names


def get_predictions(model, dataloader, device):
    """
    Get predictions for all samples in a dataloader.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader
        device: Device to run inference on
    
    Returns:
        Tuple of (all_predictions, all_labels)
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            # Get predictions
            if outputs.shape[1] == 1:  # Binary classification
                preds = (torch.sigmoid(outputs) > 0.5).long().squeeze()
            else:  # Multi-class classification
                preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return all_preds, all_labels
