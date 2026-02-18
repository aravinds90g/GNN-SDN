"""
Utility Functions for GNN Pipeline
Helper functions for data processing, visualization, and evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from typing import Tuple


def aggregate_time_windows(df: pd.DataFrame, 
                           window_seconds: int = 60) -> pd.DataFrame:
    """
    Aggregate network flows into time windows.
    
    This groups flows by source/destination IP and time window,
    aggregating packet counts, bytes, and other metrics.
    
    Args:
        df: DataFrame with network flows
        window_seconds: Time window size in seconds
        
    Returns:
        Aggregated DataFrame
    """
    print(f"Aggregating flows into {window_seconds}s time windows...")
    
    # Group by source and destination
    agg_dict = {
        'Packet_Count': 'sum',
        'Bytes_Transferred': 'sum',
        'Duration': 'mean',
        'Label': 'max'  # Mark as attack if any flow is attack
    }
    
    # Add optional columns if they exist
    optional_cols = [
        'Source_Packets', 'Destination_Packets',
        'Source_Bytes', 'Destination_Bytes',
        'Bytes_Per_Packet', 'Packets_Per_Second'
    ]
    
    for col in optional_cols:
        if col in df.columns:
            agg_dict[col] = 'sum' if 'Packets' in col or 'Bytes' in col else 'mean'
    
    # Group and aggregate
    df_agg = df.groupby(['Source_IP', 'Destination_IP'], as_index=False).agg(agg_dict)
    
    print(f"Aggregated from {len(df)} to {len(df_agg)} flows")
    return df_agg


def normalize_features(features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize features using StandardScaler.
    
    Args:
        features: Feature matrix (num_samples x num_features)
        
    Returns:
        Normalized features and fitted scaler
    """
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    return normalized, scaler


def balance_dataset(df: pd.DataFrame, 
                    label_col: str = 'Label',
                    method: str = 'undersample') -> pd.DataFrame:
    """
    Balance dataset by class.
    
    Args:
        df: DataFrame with imbalanced classes
        label_col: Name of label column
        method: 'undersample' or 'oversample'
        
    Returns:
        Balanced DataFrame
    """
    print(f"\nBalancing dataset using {method}...")
    
    # Separate classes
    df_majority = df[df[label_col] == 0]
    df_minority = df[df[label_col] == 1]
    
    print(f"Original distribution:")
    print(f"  Normal (0): {len(df_majority)}")
    print(f"  Attack (1): {len(df_minority)}")
    
    if method == 'undersample':
        # Downsample majority class
        df_majority_downsampled = resample(
            df_majority,
            replace=False,
            n_samples=len(df_minority),
            random_state=42
        )
        df_balanced = pd.concat([df_majority_downsampled, df_minority])
    
    elif method == 'oversample':
        # Upsample minority class
        df_minority_upsampled = resample(
            df_minority,
            replace=True,
            n_samples=len(df_majority),
            random_state=42
        )
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    else:
        raise ValueError(f"Unknown balancing method: {method}")
    
    # Shuffle
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Balanced distribution:")
    print(f"  Normal (0): {len(df_balanced[df_balanced[label_col] == 0])}")
    print(f"  Attack (1): {len(df_balanced[df_balanced[label_col] == 1])}")
    
    return df_balanced


def plot_graph_statistics(G, save_path: str = None):
    """
    Plot graph statistics.
    
    Args:
        G: NetworkX graph
        save_path: Optional path to save plot
    """
    import networkx as nx
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Degree distribution
    degrees = [G.degree(n) for n in G.nodes()]
    axes[0, 0].hist(degrees, bins=30, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Degree', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Degree Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Edge weight distribution
    weights = [G[u][v]['packet_count'] for u, v in G.edges()]
    axes[0, 1].hist(weights, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_xlabel('Packet Count', fontsize=11)
    axes[0, 1].set_ylabel('Frequency', fontsize=11)
    axes[0, 1].set_title('Edge Weight Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Label distribution
    labels = [G[u][v]['label'] for u, v in G.edges()]
    label_counts = pd.Series(labels).value_counts()
    axes[1, 0].bar(['Normal', 'Attack'], 
                   [label_counts.get(0, 0), label_counts.get(1, 0)],
                   color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Edge Label Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Graph statistics text
    stats_text = f"""
    Graph Statistics:
    
    Nodes: {G.number_of_nodes():,}
    Edges: {G.number_of_edges():,}
    
    Avg Degree: {np.mean(degrees):.2f}
    Max Degree: {np.max(degrees)}
    Min Degree: {np.min(degrees)}
    
    Density: {nx.density(G):.4f}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                    verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graph statistics plot saved to {save_path}")
    
    plt.show()


def plot_node_feature_distributions(features: np.ndarray, 
                                    feature_names: list = None,
                                    save_path: str = None):
    """
    Plot distributions of node features.
    
    Args:
        features: Node feature matrix
        feature_names: List of feature names
        save_path: Optional path to save plot
    """
    num_features = features.shape[1]
    
    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(num_features)]
    
    fig, axes = plt.subplots(1, num_features, figsize=(5*num_features, 4))
    
    if num_features == 1:
        axes = [axes]
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        ax.hist(features[:, i], bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel(name, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{name} Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature distribution plot saved to {save_path}")
    
    plt.show()


def save_predictions(predictions: np.ndarray,
                     true_labels: np.ndarray,
                     node_ids: list,
                     output_path: str):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: Predicted labels
        true_labels: True labels
        node_ids: List of node IDs (IP addresses)
        output_path: Path to save CSV
    """
    df = pd.DataFrame({
        'Node_ID': node_ids,
        'True_Label': true_labels,
        'Predicted_Label': predictions,
        'Correct': predictions == true_labels
    })
    
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def load_model(model_path: str, model_class):
    """
    Load saved model.
    
    Args:
        model_path: Path to saved model
        model_class: Model class to instantiate
        
    Returns:
        Loaded model
    """
    checkpoint = torch.load(model_path)
    
    # Create model with saved config
    config = checkpoint['model_config']
    model = model_class(**config)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model loaded from {model_path}")
    print(f"Saved metrics: {checkpoint.get('metrics', 'N/A')}")
    
    return model, checkpoint


if __name__ == "__main__":
    # Test utilities
    print("="*60)
    print("GNN Utilities Test")
    print("="*60)
    
    # Test feature normalization
    print("\n1. Testing feature normalization:")
    features = np.random.randn(100, 5) * 10 + 50
    normalized, scaler = normalize_features(features)
    print(f"   Original mean: {features.mean(axis=0)}")
    print(f"   Normalized mean: {normalized.mean(axis=0)}")
    print(f"   Normalized std: {normalized.std(axis=0)}")
    
    # Test dataset balancing
    print("\n2. Testing dataset balancing:")
    df_test = pd.DataFrame({
        'Source_IP': range(1000),
        'Destination_IP': range(1000, 2000),
        'Label': [0]*900 + [1]*100  # Imbalanced
    })
    df_balanced = balance_dataset(df_test, method='oversample')
    
    print("\n" + "="*60)
    print("✅ All utilities tested successfully!")
    print("="*60)
