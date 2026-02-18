"""
GNN Model Architecture for IoT Attack Detection
Implements Graph Convolutional Network (GCN) for network intrusion detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from typing import Optional


class IoTGNN(nn.Module):
    """
    Graph Neural Network for IoT attack detection.
    
    Architecture:
        Input → GCN Layer 1 → ReLU → Dropout
             → GCN Layer 2 → ReLU → Dropout
             → Dense Layer → Output (2 classes: Normal/Attack)
    """
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 64,
                 output_dim: int = 2,
                 dropout: float = 0.5,
                 gnn_type: str = 'GCN'):
        """
        Initialize GNN model.
        
        Args:
            input_dim: Number of input features per node
            hidden_dim: Hidden layer dimension
            output_dim: Number of output classes (2 for binary classification)
            dropout: Dropout rate for regularization
            gnn_type: Type of GNN layer ('GCN', 'GAT', or 'SAGE')
        """
        super(IoTGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.gnn_type = gnn_type
        
        # Choose GNN layer type
        if gnn_type == 'GCN':
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim // 2)
        elif gnn_type == 'GAT':
            self.conv1 = GATConv(input_dim, hidden_dim, heads=4, concat=True)
            self.conv2 = GATConv(hidden_dim * 4, hidden_dim // 2, heads=1)
        elif gnn_type == 'SAGE':
            self.conv1 = SAGEConv(input_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim // 2)
        else:
            raise ValueError(f"Unknown GNN type: {gnn_type}")
        
        # Batch normalization for stability
        self.bn1 = nn.BatchNorm1d(hidden_dim if gnn_type != 'GAT' else hidden_dim * 4)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 2)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric Data object with x and edge_index
            
        Returns:
            Output logits (num_nodes x num_classes)
        """
        x, edge_index = data.x, data.edge_index
        
        # First GNN layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Second GNN layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification layer
        x = self.fc(x)
        
        return x
    
    def predict(self, data):
        """
        Make predictions (inference mode).
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Predicted class labels
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            predictions = logits.argmax(dim=1)
        return predictions
    
    def predict_proba(self, data):
        """
        Get prediction probabilities.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Class probabilities (num_nodes x num_classes)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            probabilities = F.softmax(logits, dim=1)
        return probabilities


class DeepIoTGNN(nn.Module):
    """
    Deeper GNN model with 3 layers for larger graphs.
    """
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [64, 32, 16],
                 output_dim: int = 2,
                 dropout: float = 0.5):
        """
        Initialize deep GNN model.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            output_dim: Number of output classes
            dropout: Dropout rate
        """
        super(DeepIoTGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Build GNN layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(input_dim, hidden_dims[0]))
        self.bns.append(nn.BatchNorm1d(hidden_dims[0]))
        
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.convs.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
            self.bns.append(nn.BatchNorm1d(hidden_dims[i + 1]))
        
        # Output layer
        self.fc = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, data):
        """Forward pass through deep GNN."""
        x, edge_index = data.x, data.edge_index
        
        # Pass through GNN layers
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Classification
        x = self.fc(x)
        return x
    
    def predict(self, data):
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(data)
            predictions = logits.argmax(dim=1)
        return predictions


def create_model(input_dim: int,
                 model_type: str = 'simple',
                 **kwargs) -> nn.Module:
    """
    Factory function to create GNN models.
    
    Args:
        input_dim: Number of input features
        model_type: 'simple' or 'deep'
        **kwargs: Additional arguments for model
        
    Returns:
        GNN model instance
    """
    if model_type == 'simple':
        return IoTGNN(input_dim, **kwargs)
    elif model_type == 'deep':
        return DeepIoTGNN(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    print("="*60)
    print("GNN Model Architecture Test")
    print("="*60)
    
    # Create sample data
    from torch_geometric.data import Data
    
    num_nodes = 100
    num_features = 5
    num_edges = 200
    
    x = torch.randn(num_nodes, num_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    y = torch.randint(0, 2, (num_nodes,))
    
    data = Data(x=x, edge_index=edge_index, y=y)
    
    # Test simple model
    print("\n1. Testing Simple GNN (GCN):")
    model = IoTGNN(input_dim=num_features, hidden_dim=64)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    output = model(data)
    print(f"   Input shape: {data.x.shape}")
    print(f"   Output shape: {output.shape}")
    
    predictions = model.predict(data)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Unique predictions: {torch.unique(predictions).tolist()}")
    
    # Test GAT model
    print("\n2. Testing GAT Model:")
    model_gat = IoTGNN(input_dim=num_features, hidden_dim=16, gnn_type='GAT')
    print(f"   Model parameters: {sum(p.numel() for p in model_gat.parameters()):,}")
    output_gat = model_gat(data)
    print(f"   Output shape: {output_gat.shape}")
    
    # Test deep model
    print("\n3. Testing Deep GNN:")
    model_deep = DeepIoTGNN(input_dim=num_features, hidden_dims=[64, 32, 16])
    print(f"   Model parameters: {sum(p.numel() for p in model_deep.parameters()):,}")
    output_deep = model_deep(data)
    print(f"   Output shape: {output_deep.shape}")
    
    print("\n" + "="*60)
    print("✅ All models created successfully!")
    print("="*60)
