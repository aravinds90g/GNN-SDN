"""
Complete Training Pipeline for GNN-based IoT Attack Detection
Includes data loading, model training, evaluation, and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import argparse
import os
from datetime import datetime

from graph_builder import build_graph_from_csv
from gnn_model import create_model


class GNNTrainer:
    """
    Trainer class for GNN models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 device: str = 'cpu',
                 learning_rate: float = 0.001,
                 weight_decay: float = 5e-4):
        """
        Initialize trainer.
        
        Args:
            model: GNN model to train
            device: Device to use ('cpu' or 'cuda')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        
    def train_epoch(self, data, train_mask):
        """
        Train for one epoch.
        
        Args:
            data: PyTorch Geometric Data object
            train_mask: Boolean mask for training nodes
            
        Returns:
            Training loss and accuracy
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        out = self.model(data)
        
        # Compute loss only on training nodes
        loss = self.criterion(out[train_mask], data.y[train_mask])
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        # Compute accuracy
        pred = out[train_mask].argmax(dim=1)
        acc = (pred == data.y[train_mask]).float().mean()
        
        return loss.item(), acc.item()
    
    @torch.no_grad()
    def evaluate(self, data, mask):
        """
        Evaluate model on validation/test set.
        
        Args:
            data: PyTorch Geometric Data object
            mask: Boolean mask for evaluation nodes
            
        Returns:
            Loss and accuracy
        """
        self.model.eval()
        
        out = self.model(data)
        loss = self.criterion(out[mask], data.y[mask])
        
        pred = out[mask].argmax(dim=1)
        acc = (pred == data.y[mask]).float().mean()
        
        return loss.item(), acc.item()
    
    def train(self,
              data,
              train_mask,
              val_mask,
              epochs: int = 200,
              early_stopping_patience: int = 20,
              verbose: bool = True):
        """
        Train model with early stopping.
        
        Args:
            data: PyTorch Geometric Data object
            train_mask: Training node mask
            val_mask: Validation node mask
            epochs: Maximum number of epochs
            early_stopping_patience: Patience for early stopping
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        data = data.to(self.device)
        best_val_acc = 0
        patience_counter = 0
        
        # Initialize best_model_state to prevent AttributeError
        self.best_model_state = self.model.state_dict()
        
        for epoch in range(1, epochs + 1):
            # Train
            train_loss, train_acc = self.train_epoch(data, train_mask)
            
            # Validate
            val_loss, val_acc = self.evaluate(data, val_mask)
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                self.best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
            
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
            
            # Print progress
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Load best model
        self.model.load_state_dict(self.best_model_state)
        
        return {
            'train_losses': self.train_losses,
            'train_accs': self.train_accs,
            'val_losses': self.val_losses,
            'val_accs': self.val_accs,
            'best_val_acc': best_val_acc
        }
    
    def get_predictions(self, data, mask):
        """
        Get predictions for nodes.
        
        Args:
            data: PyTorch Geometric Data object
            mask: Node mask
            
        Returns:
            Predictions and true labels
        """
        self.model.eval()
        data = data.to(self.device)
        
        with torch.no_grad():
            out = self.model(data)
            pred = out[mask].argmax(dim=1).cpu().numpy()
            true = data.y[mask].cpu().numpy()
        
        return pred, true


def create_train_val_test_masks(num_nodes: int,
                                 train_ratio: float = 0.6,
                                 val_ratio: float = 0.2,
                                 seed: int = 42):
    """
    Create train/val/test masks for nodes.
    
    Args:
        num_nodes: Total number of nodes
        train_ratio: Ratio of training nodes
        val_ratio: Ratio of validation nodes
        seed: Random seed
        
    Returns:
        Train, validation, and test masks
    """
    np.random.seed(seed)
    indices = np.random.permutation(num_nodes)
    
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return train_mask, val_mask, test_mask


def plot_training_history(history: dict, save_path: str = None):
    """
    Plot training history.
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    ax1.plot(history['train_losses'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_losses'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(history['train_accs'], label='Train Accuracy', linewidth=2)
    ax2.plot(history['val_accs'], label='Val Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Optional path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Attack'],
                yticklabels=['Normal', 'Attack'],
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def evaluate_model(trainer: GNNTrainer, data, test_mask):
    """
    Comprehensive model evaluation.
    
    Args:
        trainer: Trained GNN trainer
        data: PyTorch Geometric Data object
        test_mask: Test node mask
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    y_pred, y_true = trainer.get_predictions(data, test_mask)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='binary')
    recall = recall_score(y_true, y_pred, average='binary')
    f1 = f1_score(y_true, y_pred, average='binary')
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, 
                                target_names=['Normal', 'Attack'],
                                labels=[0, 1],
                                digits=4,
                                zero_division=0))
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def main():
    parser = argparse.ArgumentParser(description='Train GNN for IoT attack detection')
    parser.add_argument('--data', type=str, default='test_preprocessed.csv',
                        help='Path to preprocessed CSV file')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--gnn-type', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'SAGE'],
                        help='Type of GNN layer')
    parser.add_argument('--save-model', type=str, default=None,
                        help='Path to save trained model')
    parser.add_argument('--test-mode', action='store_true',
                        help='Run in test mode (fewer epochs)')
    parser.add_argument('--no-plot', action='store_true',
                        help='Disable plotting')
    
    args = parser.parse_args()
    
    # Override epochs in test mode
    if args.test_mode:
        args.epochs = 10
        print("Running in TEST MODE (10 epochs only)")
    
    print("="*60)
    print("GNN TRAINING PIPELINE FOR IOT ATTACK DETECTION")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Hidden Dim: {args.hidden_dim}")
    print(f"  Dropout: {args.dropout}")
    print(f"  GNN Type: {args.gnn_type}")
    
    # Load and build graph
    print(f"\n{'='*60}")
    print("STEP 1: Building Graph")
    print("="*60)
    data, builder = build_graph_from_csv(args.data, normalize=True)
    
    # Create train/val/test splits
    print(f"\n{'='*60}")
    print("STEP 2: Creating Train/Val/Test Splits")
    print("="*60)
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        data.num_nodes,
        train_ratio=0.6,
        val_ratio=0.2
    )
    
    print(f"  Training nodes:   {train_mask.sum().item()}")
    print(f"  Validation nodes: {val_mask.sum().item()}")
    print(f"  Test nodes:       {test_mask.sum().item()}")
    
    # Create model
    print(f"\n{'='*60}")
    print("STEP 3: Creating GNN Model")
    print("="*60)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    model = create_model(
        input_dim=data.num_features,
        model_type='simple',
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        gnn_type=args.gnn_type
    )
    
    print(f"  Model: {args.gnn_type}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print(f"\n{'='*60}")
    print("STEP 4: Training Model")
    print("="*60)
    trainer = GNNTrainer(model, device=device, learning_rate=args.lr)
    
    history = trainer.train(
        data,
        train_mask,
        val_mask,
        epochs=args.epochs,
        early_stopping_patience=20,
        verbose=True
    )
    
    print(f"\nBest validation accuracy: {history['best_val_acc']:.4f}")
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("STEP 5: Evaluating on Test Set")
    print("="*60)
    metrics = evaluate_model(trainer, data, test_mask)
    
    # Plot results
    if not args.no_plot:
        print(f"\n{'='*60}")
        print("STEP 6: Generating Plots")
        print("="*60)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_training_history(history, f'training_history_{timestamp}.png')
        
        y_pred, y_true = trainer.get_predictions(data, test_mask)
        plot_confusion_matrix(y_true, y_pred, f'confusion_matrix_{timestamp}.png')
    
    # Save model
    if args.save_model:
        torch.save({
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_dim': data.num_features,
                'hidden_dim': args.hidden_dim,
                'dropout': args.dropout,
                'gnn_type': args.gnn_type
            },
            'metrics': metrics,
            'history': history
        }, args.save_model)
        print(f"\nModel saved to {args.save_model}")
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
