"""
Visualization utilities for training analysis and model interpretation.

This module provides:
- Training curves plotting
- Confusion matrix visualization
- Grad-CAM implementation and visualization
- Sample predictions display
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report


def plot_training_curves(history, save_path=None, title="Training History"):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history: Dictionary with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
        save_path: Path to save the figure (optional)
        title: Overall title for the plot
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_transfer_learning_curves(history, stage1_epochs, save_path=None):
    """
    Plot training curves for two-stage transfer learning.
    
    Args:
        history: Training history dictionary
        stage1_epochs: Number of epochs in stage 1
        save_path: Path to save the figure
    """
    total_epochs = len(history['train_loss'])
    epochs = range(1, total_epochs + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax1.axvline(x=stage1_epochs, color='gray', linestyle='--', 
                label='Stage 2 Start', linewidth=1.5)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Accuracy', linewidth=2)
    ax2.axvline(x=stage1_epochs, color='gray', linestyle='--', 
                label='Stage 2 Start', linewidth=1.5)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training vs Validation Accuracy', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Transfer Learning - Two Stage Training', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None, 
                         normalize=False, figsize=(12, 10)):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the figure
        normalize: Whether to normalize the matrix
        figsize: Figure size
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_classification_report(y_true, y_pred, class_names=None):
    """
    Print detailed classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
    """
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(classification_report(y_true, y_pred, target_names=class_names))
    print("="*70 + "\n")


class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
    
    Visualizes which regions of an image are important for the model's prediction.
    
    Args:
        model: PyTorch model
        target_layer: Name of the target layer for Grad-CAM
    
    Example:
        >>> gradcam = GradCAM(model, target_layer='layer4')
        >>> cam, pred_class = gradcam(image_tensor)
    """
    
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on the target layer."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        
        # Find and register hooks on target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_backward_hook(backward_hook)
                return
        
        raise ValueError(f"Layer '{self.target_layer}' not found in model")
    
    def __call__(self, x, class_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            x: Input image tensor (C, H, W)
            class_idx: Target class index (if None, uses predicted class)
        
        Returns:
            Tuple of (cam_heatmap, predicted_class_idx)
        """
        device = next(self.model.parameters()).device
        x = x.unsqueeze(0).to(device)
        
        # Forward pass
        output = self.model(x)
        
        # Get target class
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        loss = output[0, class_idx]
        loss.backward()
        
        # Compute CAM
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Resize to input size
        cam = F.interpolate(cam, size=x.shape[2:], mode='bilinear', 
                           align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, class_idx


def overlay_gradcam(img_tensor, cam, alpha=0.4, colormap=cv2.COLORMAP_JET,
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Overlay Grad-CAM heatmap on original image.
    
    Args:
        img_tensor: Original image tensor (C, H, W)
        cam: Grad-CAM heatmap (H, W)
        alpha: Transparency of heatmap overlay
        colormap: OpenCV colormap for heatmap
        mean: Mean used for normalization
        std: Std used for normalization
    
    Returns:
        Overlaid image as numpy array (H, W, C)
    """
    # Denormalize image
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    mean = np.array(mean)
    std = np.array(std)
    img = std * img + mean
    img = np.clip(img, 0, 1)
    
    # Create heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = heatmap / 255.0
    
    # Overlay
    overlay = alpha * heatmap + (1 - alpha) * img
    
    return overlay


def visualize_gradcam_grid(model, dataset, target_layer, num_samples=12, 
                          class_names=None, save_path=None, seed=42):
    """
    Visualize Grad-CAM for a grid of random samples.
    
    Args:
        model: PyTorch model
        dataset: Test dataset
        target_layer: Layer name for Grad-CAM
        num_samples: Number of samples to visualize
        class_names: List of class names
        save_path: Path to save the figure
        seed: Random seed for reproducibility
    """
    import random
    random.seed(seed)
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Sample random indices
    num_rows = 3
    num_cols = 4
    num_samples = min(num_samples, num_rows * num_cols)
    
    indices = random.sample(range(len(dataset)), num_samples)
    
    # Create figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, dataset_idx in enumerate(indices):
        img_tensor, true_label = dataset[dataset_idx]
        
        # Generate Grad-CAM
        cam, pred_idx = gradcam(img_tensor)
        overlay = overlay_gradcam(img_tensor, cam)
        
        # Plot
        axes[idx].imshow(overlay)
        axes[idx].axis('off')
        
        # Set title
        if class_names:
            true_class = class_names[true_label]
            pred_class = class_names[pred_idx]
            title = f"True: {true_class}\nPred: {pred_class}"
        else:
            title = f"True: {true_label}\nPred: {pred_idx}"
        
        # Color code: green if correct, red if wrong
        color = 'green' if true_label == pred_idx else 'red'
        axes[idx].set_title(title, fontsize=10, color=color)
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Grad-CAM Visualizations', fontsize=16, y=0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_sample_predictions(model, dataset, device, num_samples=16, 
                           class_names=None, save_path=None, seed=42):
    """
    Plot sample predictions from the model.
    
    Args:
        model: PyTorch model
        dataset: Test dataset
        device: Device to run inference on
        num_samples: Number of samples to show
        class_names: List of class names
        save_path: Path to save the figure
        seed: Random seed
    """
    import random
    random.seed(seed)
    
    model.eval()
    indices = random.sample(range(len(dataset)), num_samples)
    
    num_rows = 4
    num_cols = 4
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 16))
    axes = axes.flatten()
    
    with torch.no_grad():
        for idx, dataset_idx in enumerate(indices):
            img_tensor, true_label = dataset[dataset_idx]
            
            # Predict
            img_input = img_tensor.unsqueeze(0).to(device)
            output = model(img_input)
            pred_idx = output.argmax(dim=1).item()
            
            # Denormalize for display
            img = img_tensor.cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)
            
            # Plot
            axes[idx].imshow(img)
            axes[idx].axis('off')
            
            # Title
            if class_names:
                true_class = class_names[true_label]
                pred_class = class_names[pred_idx]
                title = f"True: {true_class}\nPred: {pred_class}"
            else:
                title = f"True: {true_label}\nPred: {pred_idx}"
            
            color = 'green' if true_label == pred_idx else 'red'
            axes[idx].set_title(title, fontsize=9, color=color)
    
    plt.suptitle('Sample Predictions', fontsize=16, y=0.99)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
