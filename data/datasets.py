"""
Dataset classes for Oxford-IIIT Pet classification.

This module provides custom PyTorch Dataset wrappers for:
- Binary classification (Cat vs Dog)
- Multi-class classification (37 breeds)
- Proper handling of train/val/test splits with different transforms
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class BinaryOxfordPets(Dataset):
    """
    Dataset wrapper for binary classification (Cat vs Dog).
    
    Converts the original 37-class Oxford-IIIT Pet dataset into a binary
    classification task where:
        - 0 = Cat
        - 1 = Dog
    
    Args:
        base_dataset: Original OxfordIIITPet dataset
        id_to_binary: Dictionary mapping class_id (1-37) to binary label (0 or 1)
    """
    
    def __init__(self, base_dataset, id_to_binary):
        self.base_dataset = base_dataset
        self.id_to_binary = id_to_binary
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, class_id = self.base_dataset[idx]  # class_id is 0-36
        class_id += 1  # Convert to 1-37 to match list.txt
        binary_label = self.id_to_binary[class_id]
        return img, binary_label


class BinaryOxfordPetsSubset(Dataset):
    """
    Custom subset for binary classification with specific transforms.
    
    Used to create train/validation datasets with different augmentations
    after splitting the trainval dataset.
    
    Args:
        original_dataset: Original OxfordIIITPet dataset (without transforms)
        indices: List of indices for this subset
        id_to_binary: Dictionary mapping class_id to binary label
        transform: Transformations to apply to images
    """
    
    def __init__(self, original_dataset, indices, id_to_binary, transform=None):
        self.original_dataset = original_dataset
        self.indices = indices
        self.id_to_binary = id_to_binary
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, class_id = self.original_dataset[original_idx]  # PIL Image, class_id 0-36
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        # Convert to binary label
        class_id_1_indexed = class_id + 1
        binary_label = self.id_to_binary[class_id_1_indexed]
        
        return img, torch.tensor(binary_label, dtype=torch.float32)


class MultiClassOxfordPets(Dataset):
    """
    Dataset wrapper for 37-class breed classification.
    
    Args:
        base_dataset: Original OxfordIIITPet dataset (without transforms)
        indices: Optional list of indices for subsetting (for train/val split)
        transform: Transformations to apply to images
    """
    
    def __init__(self, base_dataset, indices=None, transform=None):
        self.base_dataset = base_dataset
        self.indices = indices if indices is not None else list(range(len(base_dataset)))
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        img, class_id = self.base_dataset[original_idx]  # class_id: 0-36
        
        # Apply transforms
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(class_id, dtype=torch.long)


def load_class_mappings(data_root):
    """
    Load class mappings from list.txt annotation file.
    
    Args:
        data_root: Root directory containing the dataset
    
    Returns:
        Tuple of (id_to_breed, id_to_binary) dictionaries
    """
    list_file = os.path.join(data_root, "oxford-iiit-pet/annotations/list.txt")
    
    if not os.path.exists(list_file):
        raise FileNotFoundError(
            f"Annotation file not found: {list_file}\n"
            "Please ensure the Oxford-IIIT Pet dataset is properly downloaded."
        )
    
    id_to_breed = {}
    id_to_binary = {}
    
    with open(list_file, "r") as f:
        for line in f:
            # Skip comment lines
            if line.startswith("#"):
                continue
            
            # Parse line: <image_name> <class_id> <species> <breed_id>
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            
            name, class_id, species, breed_id = parts
            class_id = int(class_id)
            species = int(species)
            
            # Extract breed name from image name
            breed = name.rsplit("_", 1)[0]
            
            # Convert species to binary: 1=cat → 0, 2=dog → 1
            binary = 0 if species == 1 else 1
            
            id_to_breed[class_id] = breed
            id_to_binary[class_id] = binary
    
    return id_to_breed, id_to_binary


def get_transforms(config, augmentation=True):
    """
    Get data transforms based on configuration.
    
    Args:
        config: Configuration object with IMAGENET_MEAN, IMAGENET_STD, IMAGE_SIZE
        augmentation: Whether to apply data augmentation (for training)
    
    Returns:
        torchvision.transforms.Compose object
    """
    if augmentation:
        # Training transforms with augmentation
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])
    else:
        # Validation/test transforms (no augmentation)
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD)
        ])


def get_simple_transforms(image_size=224):
    """
    Get simple transforms without normalization (for v0, v1, v2 models).
    
    Args:
        image_size: Target image size
    
    Returns:
        torchvision.transforms.Compose object
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
