#!/usr/bin/env python3
"""
GNN Detection with SDN Integration
Real-time attack detection with automatic network blocking
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import argparse
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graph_builder import build_graph_from_csv
from gnn_model import create_model
from train_gnn import GNNTrainer, create_train_val_test_masks
from sdn.gnn_alert_sender import SDNAlertSender


# IP mapping for demo (in real scenario, this would come from network metadata)
NODE_TO_IP_MAPPING = {
    0: "192.168.1.2",  # h1
    1: "192.168.1.3",  # h2
    2: "192.168.1.5",  # h3 (potentially malicious)
    3: "192.168.1.6",  # h4
}


class GNNSDNIntegration:
    """
    Integrates GNN attack detection with SDN controller
    """
    
    def __init__(self, 
                 model_path: str = None,
                 controller_url: str = "http://127.0.0.1:8080",
                 enable_blocking: bool = True):
        """
        Initialize GNN-SDN integration
        
        Args:
            model_path: Path to trained model (optional)
            controller_url: URL of Ryu controller
            enable_blocking: Whether to actually send blocking commands
        """
        self.model_path = model_path
        self.enable_blocking = enable_blocking
        
        # Initialize alert sender
        self.alert_sender = SDNAlertSender(controller_url)
        
        # Check SDN controller connection
        if self.enable_blocking:
            if self.alert_sender.check_connection():
                print("✅ Connected to SDN controller")
            else:
                print("⚠️  Cannot connect to SDN controller")
                print("   Blocking will be disabled")
                self.enable_blocking = False
    
    def detect_and_block(self,
                         data,
                         trainer: GNNTrainer,
                         test_mask,
                         node_to_ip: Dict[int, str] = None) -> Dict:
        """
        Detect malicious nodes and send blocking alerts
        
        Args:
            data: PyTorch Geometric Data object
            trainer: Trained GNN trainer
            test_mask: Test node mask
            node_to_ip: Mapping from node indices to IP addresses
            
        Returns:
            Detection results dictionary
        """
        if node_to_ip is None:
            node_to_ip = NODE_TO_IP_MAPPING
        
        print("\n" + "="*60)
        print("🔍 REAL-TIME ATTACK DETECTION & BLOCKING")
        print("="*60)
        
        # Get predictions
        y_pred, y_true = trainer.get_predictions(data, test_mask)
        
        # Find malicious nodes
        test_indices = torch.where(test_mask)[0].numpy()
        
        malicious_nodes = []
        blocked_ips = []
        
        for i, (pred, true) in enumerate(zip(y_pred, y_true)):
            node_idx = test_indices[i]
            
            if pred == 1:  # Malicious
                ip = node_to_ip.get(node_idx, f"Unknown-{node_idx}")
                malicious_nodes.append({
                    'node_idx': int(node_idx),
                    'ip': ip,
                    'predicted': int(pred),
                    'actual': int(true),
                    'correct': bool(pred == true)
                })
                
                print(f"\n🚨 ALERT: Malicious device detected!")
                print(f"   Node Index: {node_idx}")
                print(f"   IP Address: {ip}")
                print(f"   Prediction: {'Attack' if pred == 1 else 'Normal'}")
                print(f"   Actual: {'Attack' if true == 1 else 'Normal'}")
                print(f"   Status: {'✅ Correct' if pred == true else '❌ False Positive'}")
                
                # Send blocking alert
                if self.enable_blocking and ip.startswith("192.168"):
                    print(f"   📡 Sending block command to SDN controller...")
                    success = self.alert_sender.send_alert(ip)
                    if success:
                        print(f"   ✅ IP {ip} blocked successfully!")
                        blocked_ips.append(ip)
                    else:
                        print(f"   ❌ Failed to block IP {ip}")
                else:
                    print(f"   ⚠️  Blocking disabled or invalid IP")
        
        # Summary
        print("\n" + "="*60)
        print("📊 DETECTION SUMMARY")
        print("="*60)
        print(f"Total nodes tested: {len(y_pred)}")
        print(f"Malicious detected: {len(malicious_nodes)}")
        print(f"IPs blocked: {len(blocked_ips)}")
        
        if blocked_ips:
            print(f"\n🚫 Blocked IPs:")
            for ip in blocked_ips:
                print(f"   - {ip}")
        
        # Get controller status
        if self.enable_blocking:
            status = self.alert_sender.get_status()
            if status:
                print(f"\n🎮 SDN Controller Status:")
                print(f"   Total blocked IPs: {status['total_blocked']}")
                print(f"   All blocked IPs: {status['blocked_ips']}")
        
        return {
            'malicious_nodes': malicious_nodes,
            'blocked_ips': blocked_ips,
            'total_tested': len(y_pred),
            'total_malicious': len(malicious_nodes)
        }


def main():
    parser = argparse.ArgumentParser(
        description='GNN Detection with SDN Integration'
    )
    parser.add_argument('--data', type=str, required=True,
                        help='Path to preprocessed CSV file')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (optional)')
    parser.add_argument('--controller-url', type=str, 
                        default='http://127.0.0.1:8080',
                        help='URL of Ryu controller')
    parser.add_argument('--no-blocking', action='store_true',
                        help='Disable actual blocking (detection only)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Training epochs (if model not provided)')
    parser.add_argument('--gnn-type', type=str, default='GCN',
                        choices=['GCN', 'GAT', 'SAGE'],
                        help='Type of GNN layer')
    
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 GNN-SDN INTEGRATED ATTACK DETECTION SYSTEM")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Data: {args.data}")
    print(f"  Controller: {args.controller_url}")
    print(f"  Blocking: {'Disabled' if args.no_blocking else 'Enabled'}")
    print(f"  GNN Type: {args.gnn_type}")
    
    # Initialize integration
    integration = GNNSDNIntegration(
        model_path=args.model,
        controller_url=args.controller_url,
        enable_blocking=not args.no_blocking
    )
    
    # Load and build graph
    print(f"\n{'='*60}")
    print("📊 Loading Data and Building Graph")
    print("="*60)
    data, builder = build_graph_from_csv(args.data, normalize=True)
    
    # Create splits
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        data.num_nodes,
        train_ratio=0.6,
        val_ratio=0.2
    )
    
    print(f"  Total nodes: {data.num_nodes}")
    print(f"  Training nodes: {train_mask.sum().item()}")
    print(f"  Test nodes: {test_mask.sum().item()}")
    
    # Create and train model
    print(f"\n{'='*60}")
    print("🧠 Training GNN Model")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    model = create_model(
        input_dim=data.num_features,
        model_type='simple',
        hidden_dim=64,
        dropout=0.5,
        gnn_type=args.gnn_type
    )
    
    trainer = GNNTrainer(model, device=device, learning_rate=0.001)
    
    # Train
    history = trainer.train(
        data,
        train_mask,
        val_mask,
        epochs=args.epochs,
        early_stopping_patience=20,
        verbose=True
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Best validation accuracy: {history['best_val_acc']:.4f}")
    
    # Detect and block
    results = integration.detect_and_block(
        data,
        trainer,
        test_mask,
        NODE_TO_IP_MAPPING
    )
    
    print("\n" + "="*60)
    print("✅ DETECTION & BLOCKING COMPLETE!")
    print("="*60)
    
    if results['blocked_ips']:
        print("\n🎯 Next Steps:")
        print("   1. In Mininet CLI, test blocked IPs:")
        print("      mininet> h3 ping h1  (should fail)")
        print("   2. Test normal communication:")
        print("      mininet> h1 ping h2  (should work)")
        print("   3. Check controller status:")
        print(f"      curl {args.controller_url}/status")


if __name__ == "__main__":
    main()
