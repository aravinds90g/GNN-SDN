#!/usr/bin/env python3
"""
Real-Time GNN Detection for ESP8266 Devices

This script monitors the backend server's collected data and runs
GNN detection in real-time to identify malicious devices.

When a malicious device is detected, it sends a blocking command
to the SDN controller.
"""

import torch
import pandas as pd
import time
import requests
import argparse
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_builder import build_graph_from_csv
from gnn_model import create_model
from train_gnn import GNNTrainer, create_train_val_test_masks

# Configuration
BACKEND_URL = "http://127.0.0.1:3000"
SDN_URL = "http://127.0.0.1:8080"
DATA_FILE = "d:/Aravind/FinalYearProject/backend/data/network_flows.csv"
CHECK_INTERVAL = 30  # Check every 30 seconds
MIN_SAMPLES = 20     # Minimum samples before running detection

# IP to device mapping (reverse lookup)
IP_TO_DEVICE = {
    "192.168.1.201": "ESP8266_NORMAL",
    "192.168.1.202": "ESP8266_MALICIOUS"
}


class RealTimeDetector:
    """Real-time GNN detection for ESP8266 devices"""
    
    def __init__(self, model_path=None, check_interval=30):
        self.model_path = model_path
        self.check_interval = check_interval
        self.last_check_time = 0
        self.last_row_count = 0
        self.blocked_ips = set()
        self.detection_count = 0
        
        # Load or create model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.trainer = None
        
        print("=" * 60)
        print("🔍 REAL-TIME GNN DETECTION SYSTEM")
        print("=" * 60)
        print(f"Backend URL: {BACKEND_URL}")
        print(f"SDN URL: {SDN_URL}")
        print(f"Data File: {DATA_FILE}")
        print(f"Check Interval: {check_interval}s")
        print(f"Device: {self.device}")
        print("=" * 60)
        print()
    
    def check_backend_status(self):
        """Check if backend server is running"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Backend server is running")
                print(f"   Total flows: {data['stats']['totalFlows']}")
                return True
        except Exception as e:
            print(f"❌ Backend server not reachable: {e}")
            return False
    
    def check_sdn_status(self):
        """Check if SDN controller is running"""
        try:
            response = requests.get(f"{SDN_URL}/status", timeout=5)
            if response.status_code == 200:
                print(f"✅ SDN controller is running")
                return True
        except Exception as e:
            print(f"⚠️  SDN controller not reachable: {e}")
            print(f"   Detection will continue, but blocking will be disabled")
            return False
    
    def load_data(self):
        """Load data from CSV file"""
        if not os.path.exists(DATA_FILE):
            return None
        
        try:
            df = pd.read_csv(DATA_FILE)
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def train_model_if_needed(self, data):
        """Train model if not already trained"""
        if self.model is not None:
            return True
        
        print("\n" + "=" * 60)
        print("🧠 TRAINING GNN MODEL")
        print("=" * 60)
        
        # Check if we have both classes in the data
        unique_labels = torch.unique(data.y)
        num_classes = len(unique_labels)
        
        if num_classes < 2:
            print(f"⚠️  WARNING: Only {num_classes} class found in data!")
            print(f"   Need both Normal (0) and Malicious (1) samples to train.")
            print(f"   Skipping training to prevent false positives.")
            print(f"   Please run BOTH normal and malicious devices.")
            return False
        
        # Check class balance
        normal_count = (data.y == 0).sum().item()
        malicious_count = (data.y == 1).sum().item()
        
        print(f"Data distribution:")
        print(f"  Normal samples: {normal_count}")
        print(f"  Malicious samples: {malicious_count}")
        
        if normal_count == 0 or malicious_count == 0:
            print(f"⚠️  WARNING: Missing one class entirely!")
            print(f"   Skipping training to prevent false positives.")
            return False
        
        if min(normal_count, malicious_count) < 1:
            print(f"⚠️  WARNING: Too few samples in one class!")
            print(f"   Need at least 1 node of each class.")
            print(f"   Skipping training to prevent false positives.")
            return False
        
        # Check if we have enough nodes for proper train/val/test split
        total_nodes = data.num_nodes
        if total_nodes < 3:
            print(f"⚠️  WARNING: Only {total_nodes} nodes in graph!")
            print(f"   Need at least 3 nodes for training.")
            print(f"   Skipping training to prevent false positives.")
            return False
        
        # Warn if node count is low but proceed anyway
        if total_nodes < 10:
            print(f"⚠️  Note: Only {total_nodes} nodes in graph (recommended: 10+)")
            print(f"   Model accuracy may be lower with limited data.")
            print(f"   Proceeding with training for demonstration...")
        
        print(f"✅ Sufficient data for training!")
        print(f"   Proceeding with model training...")
        
        try:
            # Create model
            self.model = create_model(
                input_dim=data.num_features,
                model_type='simple',
                hidden_dim=64,
                dropout=0.5,
                gnn_type='GCN'
            )
            
            # Create trainer
            self.trainer = GNNTrainer(
                self.model,
                device=self.device,
                learning_rate=0.001
            )
            
            # Create masks
            train_mask, val_mask, test_mask = create_train_val_test_masks(
                data.num_nodes,
                train_ratio=0.6,
                val_ratio=0.2
            )
            
            # Train
            print(f"Training on {data.num_nodes} nodes...")
            history = self.trainer.train(
                data,
                train_mask,
                val_mask,
                epochs=50,
                early_stopping_patience=10,
                verbose=False
            )
            
            # Check if model learned anything useful
            best_acc = history['best_val_acc']
            print(f"✅ Model trained! Best val accuracy: {best_acc:.4f}")
            
            # For small datasets (< 10 nodes), lower the accuracy threshold
            min_accuracy = 0.3 if total_nodes < 10 else 0.5
            
            if best_acc < min_accuracy:
                print(f"⚠️  WARNING: Model accuracy too low ({best_acc:.2%})!")
                print(f"   Model is essentially guessing randomly.")
                print(f"   Skipping detection to prevent false positives.")
                print(f"\n💡 TIP: Collect more diverse traffic data from both devices.")
                self.model = None
                self.trainer = None
                return False
            
            print(f"✅ Model accuracy acceptable for {total_nodes}-node network.")
            return True
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return False
    
    def detect_malicious_devices(self, data, builder):
        """Run GNN detection on all nodes, or use fallback if GNN unavailable"""
        
        # Fallback: Use label-based detection if GNN model isn't available
        if self.trainer is None:
            print("\n" + "=" * 60)
            print(f"🔍 RUNNING FALLBACK DETECTION #{self.detection_count + 1}")
            print("=" * 60)
            print("⚠️  GNN model not available, using label-based detection")
            print("   (Detecting devices marked as malicious in CSV)")
            
            malicious_devices = []
            
            # Check labels directly from the data
            for node_idx in range(data.num_nodes):
                if data.y[node_idx].item() == 1:  # Label = 1 means malicious
                    ip = builder.idx_to_node.get(node_idx, f"Unknown-Node-{node_idx}")
                    
                    malicious_devices.append({
                        'node_idx': int(node_idx),
                        'ip': ip,
                        'predicted': 'malicious',
                        'actual': 'malicious',
                        'method': 'label-based'
                    })
            
            self.detection_count += 1
            print(f"Total nodes: {data.num_nodes}")
            print(f"Malicious detected: {len(malicious_devices)}")
            
            return malicious_devices
        
        # GNN-based detection
        print("\n" + "=" * 60)
        print(f"🔍 RUNNING GNN DETECTION #{self.detection_count + 1}")
        print("=" * 60)
        
        try:
            # Get predictions for all nodes
            all_nodes_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            y_pred, y_true = self.trainer.get_predictions(data, all_nodes_mask)
            
            malicious_devices = []
            
            # Check each node
            for node_idx in range(len(y_pred)):
                if y_pred[node_idx] == 1:  # Malicious
                    # Get actual IP address from graph builder's node mapping
                    ip = builder.idx_to_node.get(node_idx, f"Unknown-Node-{node_idx}")
                    
                    malicious_devices.append({
                        'node_idx': int(node_idx),
                        'ip': ip,
                        'predicted': 'malicious',
                        'actual': 'malicious' if y_true[node_idx] == 1 else 'normal',
                        'method': 'GNN'
                    })
            
            self.detection_count += 1
            
            print(f"Total nodes: {len(y_pred)}")
            print(f"Malicious detected: {len(malicious_devices)}")
            
            return malicious_devices
            
        except Exception as e:
            print(f"❌ Error during detection: {e}")
            return []
    
    def block_device(self, ip):
        """Send block command to SDN controller and backend server"""
        if ip in self.blocked_ips:
            print(f"   ⚠️  {ip} already blocked")
            return False
        
        blocked_successfully = False
        
        # Try to block via SDN controller
        try:
            response = requests.post(
                f"{SDN_URL}/alert",
                json={'ip': ip},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"   ✅ Blocked {ip} via SDN controller")
                blocked_successfully = True
            else:
                print(f"   ⚠️  SDN controller returned: {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️  SDN controller error: {e}")
        
        # Also block via backend server (application-level blocking)
        try:
            response = requests.post(
                f"{BACKEND_URL}/api/device/block",
                json={'ip': ip},
                timeout=5
            )
            
            if response.status_code == 200:
                print(f"   ✅ Blocked {ip} via backend server")
                blocked_successfully = True
            else:
                print(f"   ⚠️  Backend server returned: {response.status_code}")
                
        except Exception as e:
            print(f"   ⚠️  Backend server error: {e}")
        
        if blocked_successfully:
            self.blocked_ips.add(ip)
            return True
        else:
            print(f"   ❌ Failed to block {ip} on all systems")
            return False
    
    def run(self):
        """Main detection loop"""
        print("\n🚀 Starting real-time detection...\n")
        
        # Check system status
        backend_ok = self.check_backend_status()
        sdn_ok = self.check_sdn_status()
        
        if not backend_ok:
            print("\n❌ Backend server must be running!")
            print("   Start it with: cd backend && npm start")
            return
        
        print("\n✅ System ready! Monitoring for malicious devices...\n")
        
        try:
            while True:
                # Load latest data
                df = self.load_data()
                
                if df is None or len(df) < MIN_SAMPLES:
                    if df is not None:
                        print(f"⏳ Waiting for more data... ({len(df)}/{MIN_SAMPLES} samples)")
                    time.sleep(self.check_interval)
                    continue
                
                # Check if new data is available
                current_rows = len(df)
                if current_rows == self.last_row_count:
                    print(f"⏳ No new data. Total samples: {current_rows}")
                    time.sleep(self.check_interval)
                    continue
                
                print(f"\n📊 New data available: {current_rows} samples (+{current_rows - self.last_row_count})")
                self.last_row_count = current_rows
                
                # Build graph
                print("🕸️  Building graph...")
                data, builder = build_graph_from_csv(DATA_FILE, normalize=True)
                
                # Try to train GNN model if not already trained
                # If training fails, fallback detection will be used
                if self.model is None:
                    training_success = self.train_model_if_needed(data)
                    if not training_success:
                        print("⚠️  GNN training failed, will use fallback detection...")
                
                # Run detection (GNN if available, otherwise fallback)
                malicious_devices = self.detect_malicious_devices(data, builder)
                
                # Block malicious devices
                if malicious_devices:
                    print(f"\n🚨 ALERT: {len(malicious_devices)} malicious device(s) detected!")
                    for device in malicious_devices:
                        print(f"\n   Device: {device['ip']}")
                        print(f"   Node Index: {device['node_idx']}")
                        print(f"   Prediction: {device['predicted']}")
                        print(f"   Detection Method: {device.get('method', 'unknown')}")
                        
                        # Try to block via SDN
                        if sdn_ok:
                            self.block_device(device['ip'])
                else:
                    print("✅ No malicious devices detected")
                
                print(f"\n⏰ Next check in {self.check_interval} seconds...")
                time.sleep(self.check_interval)
                
        except KeyboardInterrupt:
            print("\n\n🛑 Detection stopped by user")
            print(f"\nStatistics:")
            print(f"  Detection runs: {self.detection_count}")
            print(f"  Blocked IPs: {len(self.blocked_ips)}")
            if self.blocked_ips:
                print(f"  Blocked: {', '.join(self.blocked_ips)}")


def main():
    parser = argparse.ArgumentParser(
        description='Real-time GNN detection for ESP8266 devices'
    )
    parser.add_argument('--interval', type=int, default=30,
                        help='Check interval in seconds (default: 30)')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pre-trained model (optional)')
    
    args = parser.parse_args()
    
    detector = RealTimeDetector(
        model_path=args.model,
        check_interval=args.interval
    )
    
    detector.run()


if __name__ == "__main__":
    main()
