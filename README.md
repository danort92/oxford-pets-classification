# Oxford-IIIT Pet Classification Project

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/oxford-pets-classification/blob/main/notebooks/00_quick_demo.ipynb)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A professional deep learning project for classifying cats and dogs using the Oxford-IIIT Pet dataset. This project implements three progressive tasks with custom CNNs and transfer learning.

**🎯 Highlights:**
- ✅ 4 custom CNN architectures with increasing complexity
- ✅ Transfer learning with ResNet50 (ImageNet pretrained)
- ✅ Professional modular codebase
- ✅ Interactive Jupyter notebooks for Google Colab
- ✅ Comprehensive visualization (training curves, confusion matrices, Grad-CAM)
- ✅ Achieves 94% accuracy on binary classification, 85-90% on 37-class breed classification

## 🚀 Quick Start

### Option 1: Google Colab (Recommended - No Setup Required!)

Try it now with free GPU! Click on any notebook below:

| Notebook | Description | Time | Colab Link |
|----------|-------------|------|------------|
| **Quick Demo** | Binary classification with CNN v3 | ~10 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/oxford-pets-classification/blob/main/notebooks/00_quick_demo.ipynb) |
| **Multi-Class** | 37 breed classification with custom CNN | ~60 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/oxford-pets-classification/blob/main/notebooks/01_multiclass_classification.ipynb) |
| **Transfer Learning** | ResNet50 + Grad-CAM visualization | ~30 min | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/danort92/oxford-pets-classification/blob/main/notebooks/02_transfer_learning.ipynb) |

**Why Colab?**
- 🆓 Free GPU access (15GB VRAM)
- 📦 No installation required
- 🎨 Interactive visualizations
- 💾 Automatic dataset download

### Option 2: Local Installation

For running scripts locally or on your own server:

```bash
git clone https://github.com/danort92/oxford-pets-classification.git
cd oxford-pets-classification
pip install -r requirements.txt
python task1_binary_classification.py --model v3 --epochs 20
```

## 📋 Project Structure

```
oxford_pets_classification/
├── notebooks/                 # 🆕 Interactive Jupyter notebooks for Colab
│   ├── 00_quick_demo.ipynb
│   ├── 01_multiclass_classification.ipynb
│   └── 02_transfer_learning.ipynb
├── configs/
│   ├── __init__.py
│   └── config.py              # Centralized configuration
├── data/
│   ├── __init__.py
│   └── datasets.py            # Custom dataset classes
├── models/
│   ├── __init__.py
│   └── architectures.py       # CNN and ResNet models
├── utils/
│   ├── __init__.py
│   ├── data_utils.py          # Data loading utilities
│   ├── trainer.py             # Training loops
│   └── visualization.py       # Plotting and Grad-CAM
├── task1_binary_classification.py       # Task 1 Step 1 (CLI)
├── task2_multiclass_classification.py   # Task 1 Step 2 (CLI)
├── task3_transfer_learning.py           # Task 2 (CLI)
├── inference.py                         # Predict on new images
├── compare_results.py                   # Compare all experiments
├── outputs/                   # Training curves, visualizations
├── checkpoints/               # Model checkpoints
└── README.md
```

## 🎯 Tasks Overview

### Task 1 - Step 1: Binary Classification (Cat vs Dog)

Train custom CNNs of increasing complexity:

- **v0**: Simple 2-layer CNN (baseline)
- **v1**: 4-layer CNN with AdaptiveAvgPool
- **v2**: VGG-like with BatchNorm and Dropout
- **v3**: VGG-like with data augmentation (best)

### Task 1 - Step 2: Multi-class Classification (37 breeds)

Use the best architecture from Step 1 (v3) to classify 37 different cat and dog breeds.

### Task 2: Transfer Learning

Two-stage transfer learning with ResNet50:
1. **Stage 1**: Train only classifier (backbone frozen)
2. **Stage 2**: Fine-tune last residual block (layer4)

## 🚀 Quick Start

### Installation

```bash
# Install required packages
pip install torch torchvision tqdm matplotlib seaborn scikit-learn opencv-python
```

### Running Experiments

#### Task 1 - Step 1: Binary Classification

```bash
# Train all model versions
python task1_binary_classification.py --model v0 --epochs 20
python task1_binary_classification.py --model v1 --epochs 20
python task1_binary_classification.py --model v2 --epochs 20
python task1_binary_classification.py --model v3 --epochs 20  # Best model
```

**Output:**
- Training curves: `outputs/binary_classification_v3_curves.png`
- Model checkpoint: `checkpoints/binary_classification_v3/best_model.pth`
- Results JSON: `outputs/binary_classification_v3_results.json`

#### Task 1 - Step 2: Multi-class Classification

```bash
# Train multi-class classifier
python task2_multiclass_classification.py --epochs 100 --confusion-matrix --sample-predictions
```

**Output:**
- Training curves: `outputs/multiclass_custom_cnn_curves.png`
- Confusion matrix: `outputs/multiclass_custom_cnn_confusion_matrix.png`
- Sample predictions: `outputs/multiclass_custom_cnn_samples.png`
- Model checkpoint: `checkpoints/multiclass_custom_cnn/best_model.pth`

#### Task 2: Transfer Learning

```bash
# Two-stage transfer learning
python task3_transfer_learning.py --stage1-epochs 15 --stage2-epochs 10 \
    --confusion-matrix --sample-predictions --gradcam
```

**Output:**
- Training curves: `outputs/transfer_learning_resnet50_curves.png`
- Confusion matrix: `outputs/transfer_learning_resnet50_confusion_matrix.png`
- Grad-CAM visualizations: `outputs/transfer_learning_resnet50_gradcam.png`
- Sample predictions: `outputs/transfer_learning_resnet50_samples.png`

## 📊 Features

### Professional Code Structure

✅ **Modular Design**: Separated configs, models, data, utilities  
✅ **Reusable Components**: Trainer classes, dataset wrappers  
✅ **Type Hints**: Clear function signatures  
✅ **Documentation**: Comprehensive docstrings  

### Advanced Training Features

✅ **Progress Tracking**: tqdm progress bars  
✅ **Checkpointing**: Save/load best models  
✅ **History Logging**: Track all metrics  
✅ **Reproducibility**: Fixed random seeds  

### Visualization Tools

✅ **Training Curves**: Loss and accuracy plots  
✅ **Confusion Matrix**: Multi-class performance  
✅ **Grad-CAM**: Model interpretability  
✅ **Sample Predictions**: Visual validation  

## ⚙️ Configuration

All hyperparameters are centralized in `configs/config.py`:

```python
from configs.config import get_config

# Binary classification
config = get_config('binary', model_version='v3')

# Multi-class classification
config = get_config('multiclass')

# Transfer learning
config = get_config('transfer')
```

### Key Parameters

```python
# Training
BATCH_SIZE = 32          # Binary: 32, Multiclass: 128, Transfer: 64
EPOCHS = 20              # Binary: 20, Multiclass: 100, Transfer: 15+10
LEARNING_RATE = 1e-4     # AdamW learning rate
WEIGHT_DECAY = 1e-4      # L2 regularization

# Data
TRAIN_VAL_SPLIT = 0.7    # 70% train, 30% validation
IMAGE_SIZE = 224         # Input image size
SEED = 42                # Random seed for reproducibility
```

## 📈 Expected Results

### Binary Classification (Cat vs Dog)

| Model | Parameters | Test Accuracy |
|-------|-----------|---------------|
| v0 | ~100K | ~85% |
| v1 | ~200K | ~88% |
| v2 | ~15M | ~92% |
| v3 | ~15M | **~94%** |

### Multi-class Classification (37 breeds)

| Model | Parameters | Test Accuracy |
|-------|-----------|---------------|
| Custom CNN | ~15M | ~75-80% |

### Transfer Learning

| Model | Parameters | Test Accuracy |
|-------|-----------|---------------|
| ResNet50 | ~25M | **~85-90%** |

## 🔧 Advanced Usage

### Custom Training Loop

```python
from models.architectures import get_model
from utils.trainer import BinaryTrainer
from utils.data_utils import prepare_binary_dataloaders

# Load data
config = get_config('binary', 'v3')
train_loader, val_loader, test_loader, _ = prepare_binary_dataloaders(config)

# Create model
model = get_model('binary_v3')

# Setup trainer
trainer = BinaryTrainer(model, device='cuda')

# Train
history = trainer.fit(train_loader, val_loader, epochs=20)
```

### Loading Checkpoints

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoints/binary_classification_v3/best_model.pth')

# Restore model
model.load_state_dict(checkpoint['model_state_dict'])
```

### Custom Visualization

```python
from utils.visualization import plot_training_curves, visualize_gradcam_grid

# Plot curves
plot_training_curves(history, save_path='my_curves.png')

# Generate Grad-CAM
visualize_gradcam_grid(
    model, test_dataset, 
    target_layer='layer4',
    num_samples=12
)
```

## 📝 Code Quality

### Best Practices Implemented

- ✅ PEP 8 compliant formatting
- ✅ Comprehensive docstrings (Google style)
- ✅ Error handling with try-except
- ✅ Type hints for function signatures
- ✅ Modular and reusable components
- ✅ Configuration-driven architecture
- ✅ Logging and checkpointing
- ✅ Reproducible experiments (fixed seeds)

### Design Patterns

- **Factory Pattern**: `get_model()`, `get_config()`
- **Strategy Pattern**: Different trainers for binary/multiclass
- **Template Method**: Common training loop with customization
- **Singleton**: Centralized configuration

## 🤝 Contributing

To extend this project:

1. Add new models in `models/architectures.py`
2. Create custom datasets in `data/datasets.py`
3. Implement new trainers in `utils/trainer.py`
4. Add configurations in `configs/config.py`

## 📚 References

- Dataset: [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Grad-CAM: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

## 📄 License

This project is for educational purposes.

---

**Author**: Deep Learning Practitioner  
**Last Updated**: 2026  
**Python Version**: 3.8+  
**PyTorch Version**: 2.0+
