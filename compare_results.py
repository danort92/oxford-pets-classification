"""
Compare Results Across All Experiments

This script loads and compares results from all three tasks:
- Task 1 Step 1: Binary classification (v0, v1, v2, v3)
- Task 1 Step 2: Multi-class classification
- Task 2: Transfer learning

Usage:
    python compare_results.py
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from configs.config import BaseConfig


def load_results(experiment_name):
    """Load results JSON for an experiment."""
    results_path = BaseConfig.OUTPUT_DIR / f"{experiment_name}_results.json"
    if results_path.exists():
        with open(results_path, 'r') as f:
            return json.load(f)
    return None


def compare_binary_models():
    """Compare all binary classification models."""
    print("\n" + "="*70)
    print("TASK 1 - STEP 1: BINARY CLASSIFICATION COMPARISON")
    print("="*70)
    
    models = ['v0', 'v1', 'v2', 'v3']
    results = []
    
    for model in models:
        experiment_name = f'binary_classification_{model}'
        result = load_results(experiment_name)
        if result:
            results.append({
                'Model': model.upper(),
                'Parameters': f"{result['total_params']:,}",
                'Test Accuracy': f"{result['test_acc']:.4f}",
                'Best Val Accuracy': f"{result['best_val_acc']:.4f}",
                'Best Epoch': result['best_epoch']
            })
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        print("\n✅ Best Model: v3 (VGG-like with data augmentation)")
    else:
        print("❌ No results found. Run experiments first.")
    
    print("="*70)


def compare_all_tasks():
    """Compare final results across all three main tasks."""
    print("\n" + "="*70)
    print("OVERALL COMPARISON: ALL TASKS")
    print("="*70)
    
    tasks = [
        ('binary_classification_v3', 'Binary (v3)', 'Custom CNN'),
        ('multiclass_custom_cnn', 'Multi-class', 'Custom CNN'),
        ('transfer_learning_resnet50', 'Multi-class', 'ResNet50 Transfer')
    ]
    
    results = []
    
    for experiment_name, task, model_type in tasks:
        result = load_results(experiment_name)
        if result:
            results.append({
                'Task': task,
                'Model': model_type,
                'Parameters': f"{result.get('total_params', 'N/A'):,}" if isinstance(result.get('total_params'), int) else 'N/A',
                'Test Accuracy': f"{result['test_acc']:.4f}",
                'Best Val Accuracy': f"{result['best_val_acc']:.4f}"
            })
    
    if results:
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
        print("\n📊 Key Insights:")
        print("  • Binary classification is easier (2 classes) → Higher accuracy")
        print("  • Multi-class (37 breeds) is more challenging")
        print("  • Transfer learning (ResNet50) outperforms custom CNN for multi-class")
    else:
        print("❌ No results found. Run experiments first.")
    
    print("="*70)


def plot_accuracy_comparison():
    """Plot test accuracy comparison across all experiments."""
    experiments = [
        ('binary_classification_v0', 'Binary v0'),
        ('binary_classification_v1', 'Binary v1'),
        ('binary_classification_v2', 'Binary v2'),
        ('binary_classification_v3', 'Binary v3'),
        ('multiclass_custom_cnn', 'Multiclass\nCustom CNN'),
        ('transfer_learning_resnet50', 'Multiclass\nResNet50')
    ]
    
    names = []
    accuracies = []
    
    for exp_name, display_name in experiments:
        result = load_results(exp_name)
        if result:
            names.append(display_name)
            accuracies.append(result['test_acc'] * 100)  # Convert to percentage
    
    if not names:
        print("\n❌ No results to plot. Run experiments first.")
        return
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
    bars = ax.bar(names, accuracies, color=colors[:len(names)], alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy Comparison Across All Experiments', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save plot
    save_path = BaseConfig.OUTPUT_DIR / 'all_experiments_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Comparison plot saved to: {save_path}")
    
    plt.show()


def main():
    """Main comparison function."""
    BaseConfig.create_directories()
    
    print("\n" + "="*70)
    print("OXFORD-IIIT PET CLASSIFICATION - RESULTS SUMMARY")
    print("="*70)
    
    # Compare binary models
    compare_binary_models()
    
    # Compare all tasks
    compare_all_tasks()
    
    # Plot comparison
    plot_accuracy_comparison()
    
    print("\n✅ Analysis complete!\n")


if __name__ == "__main__":
    main()
