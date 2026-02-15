"""
Training and evaluation utilities for deep learning experiments.

This module provides:
- Trainer classes for binary and multi-class classification
- Training loops with progress tracking
- Evaluation functions
- Model checkpointing
- Early stopping
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import json


class BinaryTrainer:
    """
    Trainer for binary classification tasks.
    
    Features:
        - Training and validation loops
        - Progress tracking with tqdm
        - History logging
        - Checkpoint saving
        - Early stopping
    
    Args:
        model: PyTorch model
        device: Device to train on ('cuda' or 'cpu')
        criterion: Loss function (default: BCEWithLogitsLoss)
        optimizer: Optimizer (default: AdamW)
        save_dir: Directory to save checkpoints
    """
    
    def __init__(self, model, device, criterion=None, optimizer=None, save_dir=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.BCEWithLogitsLoss()
        self.optimizer = optimizer
        self.save_dir = Path(save_dir) if save_dir else None
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader):
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader, desc="Training", leave=False)
        
        for images, labels in progress:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            progress.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader):
        """
        Evaluate on validation/test set.
        
        Args:
            val_loader: DataLoader for validation/test data
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                
                running_loss += loss.item() * images.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def fit(self, train_loader, val_loader, epochs, verbose=True):
        """
        Train the model for multiple epochs.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of epochs to train
            verbose: Print progress information
        
        Returns:
            History dictionary
        """
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update history
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                if self.save_dir:
                    self.save_checkpoint('best_model.pth', epoch + 1)
            
            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )
        
        total_time = time.time() - total_start_time
        
        if verbose:
            print(f"\nTotal Training Time: {total_time:.2f}s")
            print(f"Best Validation Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        
        return self.history
    
    def save_checkpoint(self, filename, epoch):
        """Save model checkpoint."""
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_acc': self.best_val_acc,
                'history': self.history
            }
            torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        return checkpoint['epoch']


class MultiClassTrainer:
    """
    Trainer for multi-class classification tasks.
    
    Similar to BinaryTrainer but adapted for multi-class classification
    with CrossEntropyLoss.
    
    Args:
        model: PyTorch model
        device: Device to train on
        criterion: Loss function (default: CrossEntropyLoss)
        optimizer: Optimizer
        save_dir: Directory to save checkpoints
    """
    
    def __init__(self, model, device, criterion=None, optimizer=None, save_dir=None):
        self.model = model.to(device)
        self.device = device
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.optimizer = optimizer
        self.save_dir = Path(save_dir) if save_dir else None
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        progress = tqdm(train_loader, desc="Training", leave=False)
        
        for images, labels in progress:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def evaluate(self, val_loader):
        """Evaluate on validation/test set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        
        return epoch_loss, epoch_acc
    
    def fit(self, train_loader, val_loader, epochs, verbose=True):
        """Train the model for multiple epochs."""
        total_start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train and validate
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)
            
            # Update history
            epoch_time = time.time() - epoch_start
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1
                if self.save_dir:
                    self.save_checkpoint('best_model.pth', epoch + 1)
            
            # Print progress
            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
                    f"Time: {epoch_time:.2f}s"
                )
        
        total_time = time.time() - total_start_time
        
        if verbose:
            print(f"\nTotal Training Time: {total_time:.2f}s")
            print(f"Best Validation Accuracy: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        
        return self.history
    
    def save_checkpoint(self, filename, epoch):
        """Save model checkpoint."""
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_acc': self.best_val_acc,
                'history': self.history
            }
            torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        checkpoint = torch.load(self.save_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_acc = checkpoint['best_val_acc']
        self.history = checkpoint['history']
        return checkpoint['epoch']


def save_history(history, filepath):
    """
    Save training history to JSON file.
    
    Args:
        history: History dictionary
        filepath: Path to save JSON file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=4)


def load_history(filepath):
    """
    Load training history from JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        History dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)
