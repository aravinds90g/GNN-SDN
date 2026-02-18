"""
Graph Builder Module for Network Flow Data
Converts preprocessed network flow data into graph representation for GNN training.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, List


class NetworkFlowGraphBuilder:
    """
    Builds graph representation from network flow data.
    
    Nodes: IP addresses (source and destination)
    Edges: Network flows between IPs
    Node Features: Degree, traffic volume, packet statistics
    Edge Features: Packet count, bytes, duration
    """
    
    def __init__(self, normalize_features=True):
        """
        Initialize graph builder.
        
        Args:
            normalize_features: Whether to normalize node features
        """
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        self.node_to_idx = {}
        self.idx_to_node = {}
        self.malicious_sources = set()  # Track source IPs of malicious traffic
        
    def build_graph(self, df: pd.DataFrame, time_window: int = None) -> nx.Graph:
        """
        Build NetworkX graph from preprocessed flow data.
        
        Args:
            df: Preprocessed DataFrame with network flows
            time_window: Optional time window in seconds for aggregation
            
        Returns:
            NetworkX graph with nodes (IPs) and edges (flows)
        """
        print(f"Building graph from {len(df)} flows...")
        
        # Aggregate by time window if specified
        if time_window:
            df = self._aggregate_time_windows(df, time_window)
        
        G = nx.DiGraph()  # Use directed graph to track source->destination
        
        # Track which IPs are sources of malicious traffic
        self.malicious_sources = set()
        
        # Add edges with attributes
        for _, row in df.iterrows():
            src = row['Source_IP']
            dst = row['Destination_IP']
            
            # Get edge attributes
            edge_attrs = {
                'packet_count': row.get('Packet_Count', 0),
                'bytes': row.get('Bytes_Transferred', 0),
                'duration': row.get('Duration', 0),
                'label': row.get('Label', 0)
            }
            
            # Track malicious sources
            if edge_attrs['label'] == 1:
                self.malicious_sources.add(src)
            
            # Add optional features if available
            if 'Source_Packets' in row:
                edge_attrs['src_packets'] = row['Source_Packets']
                edge_attrs['dst_packets'] = row['Destination_Packets']
            
            if 'Bytes_Per_Packet' in row:
                edge_attrs['bytes_per_packet'] = row['Bytes_Per_Packet']
            
            # Add edge (or update if exists)
            if G.has_edge(src, dst):
                # Aggregate multiple flows between same nodes
                for key in edge_attrs:
                    if key == 'label':
                        # Mark as malicious if any flow is malicious
                        G[src][dst][key] = max(G[src][dst][key], edge_attrs[key])
                    else:
                        G[src][dst][key] += edge_attrs[key]
            else:
                G.add_edge(src, dst, **edge_attrs)
        
        print(f"Graph created: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"Malicious source IPs: {self.malicious_sources}")
        return G
    
    def extract_node_features(self, G: nx.Graph) -> np.ndarray:
        """
        Extract node features from graph.
        
        Features per node:
        1. Degree (number of connections)
        2. Total packets sent/received
        3. Total bytes sent/received
        4. Average packet size
        5. Number of malicious connections
        
        Args:
            G: NetworkX graph
            
        Returns:
            Node feature matrix (num_nodes x num_features)
        """
        print("Extracting node features...")
        
        node_features = []
        nodes = list(G.nodes())
        
        # Create node index mapping
        self.node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        self.idx_to_node = {idx: node for idx, node in enumerate(nodes)}
        
        for node in nodes:
            # Feature 1: Degree
            degree = G.degree(node)
            
            # Feature 2-3: Total packets and bytes
            total_packets = 0
            total_bytes = 0
            malicious_count = 0
            
            for neighbor in G.neighbors(node):
                edge_data = G[node][neighbor]
                total_packets += edge_data.get('packet_count', 0)
                total_bytes += edge_data.get('bytes', 0)
                if edge_data.get('label', 0) == 1:
                    malicious_count += 1
            
            # Feature 4: Average packet size
            avg_packet_size = total_bytes / (total_packets + 1)  # +1 to avoid division by zero
            
            # Feature 5: Malicious connection ratio
            malicious_ratio = malicious_count / (degree + 1)
            
            node_features.append([
                degree,
                total_packets,
                total_bytes,
                avg_packet_size,
                malicious_ratio
            ])
        
        node_features = np.array(node_features, dtype=np.float32)
        
        # Normalize features
        if self.normalize_features:
            node_features = self.scaler.fit_transform(node_features)
        
        print(f"Extracted features shape: {node_features.shape}")
        return node_features
    
    def extract_node_labels(self, G: nx.Graph) -> np.ndarray:
        """
        Extract node labels based on edge labels.
        
        A node is labeled as malicious (1) ONLY if it is the SOURCE of malicious traffic.
        Destination nodes (like the server) are NOT labeled as malicious even if they
        receive malicious traffic.
        
        Args:
            G: NetworkX graph (should be DiGraph for directional info)
            
        Returns:
            Node labels (num_nodes,)
        """
        print("Extracting node labels...")
        
        labels = []
        nodes = list(G.nodes())
        
        for node in nodes:
            # Check if this node is a source of malicious traffic
            is_malicious_source = False
            
            # For directed graphs, check outgoing edges
            if isinstance(G, nx.DiGraph):
                for neighbor in G.successors(node):
                    if G[node][neighbor].get('label', 0) == 1:
                        is_malicious_source = True
                        break
            else:
                # For undirected graphs, use the tracked malicious sources
                if hasattr(self, 'malicious_sources') and node in self.malicious_sources:
                    is_malicious_source = True
            
            labels.append(1 if is_malicious_source else 0)
        
        labels = np.array(labels, dtype=np.int64)
        
        print(f"Label distribution: Normal={np.sum(labels == 0)}, Malicious={np.sum(labels == 1)}")
        
        # Print which nodes are labeled as malicious
        malicious_nodes = [nodes[i] for i in range(len(labels)) if labels[i] == 1]
        if malicious_nodes:
            print(f"Malicious nodes: {malicious_nodes}")
        
        return labels
    
    def to_pytorch_geometric(self, G: nx.Graph) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            G: NetworkX graph (can be directed or undirected)
            
        Returns:
            PyTorch Geometric Data object
        """
        print("Converting to PyTorch Geometric format...")
        
        # Extract features and labels
        node_features = self.extract_node_features(G)
        node_labels = self.extract_node_labels(G)
        
        # Convert to PyTorch Geometric
        # Create a simple graph with just edges (no attributes)
        if isinstance(G, nx.DiGraph):
            G_simple = nx.DiGraph()
        else:
            G_simple = nx.Graph()
        G_simple.add_edges_from(G.edges())
        
        # Convert to PyTorch Geometric
        data = from_networkx(G_simple)
        
        # Add node features and labels
        data.x = torch.tensor(node_features, dtype=torch.float)
        data.y = torch.tensor(node_labels, dtype=torch.long)
        
        # Add number of features
        data.num_features = node_features.shape[1]
        
        print(f"PyTorch Geometric Data created:")
        print(f"  - Nodes: {data.num_nodes}")
        print(f"  - Edges: {data.num_edges}")
        print(f"  - Features: {data.num_features}")
        print(f"  - Classes: {len(torch.unique(data.y))}")
        
        return data
    
    def _aggregate_time_windows(self, df: pd.DataFrame, window_seconds: int) -> pd.DataFrame:
        """
        Aggregate flows into time windows.
        
        Args:
            df: DataFrame with network flows
            window_seconds: Time window size in seconds
            
        Returns:
            Aggregated DataFrame
        """
        print(f"Aggregating flows into {window_seconds}s time windows...")
        
        # This is a simplified version - assumes you have a timestamp column
        # If not available, skip aggregation
        if 'Start_Time' not in df.columns and 'Timestamp' not in df.columns:
            print("No timestamp column found, skipping time aggregation")
            return df
        
        # Implementation would group by time windows here
        # For now, return original df
        return df


def build_graph_from_csv(csv_path: str, 
                         normalize: bool = True,
                         time_window: int = None) -> Data:
    """
    Convenience function to build graph directly from CSV file.
    
    Args:
        csv_path: Path to preprocessed CSV file
        normalize: Whether to normalize node features
        time_window: Optional time window for aggregation
        
    Returns:
        PyTorch Geometric Data object
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    builder = NetworkFlowGraphBuilder(normalize_features=normalize)
    G = builder.build_graph(df, time_window=time_window)
    data = builder.to_pytorch_geometric(G)
    
    return data, builder


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "test_preprocessed.csv"
    
    print("="*60)
    print("Network Flow Graph Builder - Test")
    print("="*60)
    
    data, builder = build_graph_from_csv(csv_file)
    
    print("\n" + "="*60)
    print("Graph Statistics")
    print("="*60)
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Number of features: {data.num_features}")
    print(f"Average degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"\nLabel distribution:")
    print(f"  Normal (0): {(data.y == 0).sum().item()}")
    print(f"  Malicious (1): {(data.y == 1).sum().item()}")
