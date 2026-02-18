#!/usr/bin/env python3
"""
Test script to verify that only malicious devices are blocked, not normal ones.

This script checks the graph labeling logic to ensure:
1. Only source IPs of malicious traffic are labeled as malicious
2. Destination IPs (like the server) are NOT labeled as malicious
3. Normal devices are NOT blocked
"""

import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from graph_builder import build_graph_from_csv

def test_labeling():
    """Test the graph labeling logic"""
    
    print("=" * 60)
    print("TESTING GRAPH LABELING LOGIC")
    print("=" * 60)
    
    # Create test data
    test_data = pd.DataFrame([
        # Normal device traffic
        {
            'Source_IP': '192.168.1.201',
            'Destination_IP': '192.168.1.100',
            'Packet_Count': 25,
            'Bytes_Transferred': 2500,
            'Duration': 5,
            'Label': 0  # Normal
        },
        # Malicious device traffic
        {
            'Source_IP': '192.168.1.202',
            'Destination_IP': '192.168.1.100',
            'Packet_Count': 1500,
            'Bytes_Transferred': 150000,
            'Duration': 2,
            'Label': 1  # Malicious
        }
    ])
    
    # Save test data
    test_file = 'd:/Aravind/FinalYearProject/backend/data/test_network_flows.csv'
    test_data.to_csv(test_file, index=False)
    
    print(f"\nTest data created with {len(test_data)} flows")
    print(f"  - Normal device: 192.168.1.201")
    print(f"  - Malicious device: 192.168.1.202")
    print(f"  - Server: 192.168.1.100")
    
    # Build graph
    print("\nBuilding graph...")
    data, builder = build_graph_from_csv(test_file, normalize=True)
    
    # Check node labels
    print("\n" + "=" * 60)
    print("NODE LABEL VERIFICATION")
    print("=" * 60)
    
    nodes = list(builder.idx_to_node.values())
    labels = data.y.numpy()
    
    print(f"\nTotal nodes: {len(nodes)}")
    
    for i, (node, label) in enumerate(zip(nodes, labels)):
        label_str = "MALICIOUS" if label == 1 else "NORMAL"
        print(f"  Node {i}: {node} -> {label_str}")
    
    # Verify correctness
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS")
    print("=" * 60)
    
    errors = []
    
    # Check normal device
    normal_idx = None
    for i, node in enumerate(nodes):
        if node == '192.168.1.201':
            normal_idx = i
            if labels[i] == 1:
                errors.append(f"❌ FAIL: Normal device (192.168.1.201) incorrectly labeled as MALICIOUS")
            else:
                print(f"✅ PASS: Normal device (192.168.1.201) correctly labeled as NORMAL")
            break
    
    # Check malicious device
    malicious_idx = None
    for i, node in enumerate(nodes):
        if node == '192.168.1.202':
            malicious_idx = i
            if labels[i] == 0:
                errors.append(f"❌ FAIL: Malicious device (192.168.1.202) incorrectly labeled as NORMAL")
            else:
                print(f"✅ PASS: Malicious device (192.168.1.202) correctly labeled as MALICIOUS")
            break
    
    # Check server
    server_idx = None
    for i, node in enumerate(nodes):
        if node == '192.168.1.100':
            server_idx = i
            if labels[i] == 1:
                errors.append(f"❌ FAIL: Server (192.168.1.100) incorrectly labeled as MALICIOUS")
            else:
                print(f"✅ PASS: Server (192.168.1.100) correctly labeled as NORMAL")
            break
    
    # Final result
    print("\n" + "=" * 60)
    if errors:
        print("❌ TEST FAILED")
        print("=" * 60)
        for error in errors:
            print(error)
        return False
    else:
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("\nThe graph labeling logic is working correctly!")
        print("Only malicious SOURCE devices are labeled as malicious.")
        print("Destination nodes (like the server) are NOT labeled as malicious.")
        return True

if __name__ == "__main__":
    success = test_labeling()
    sys.exit(0 if success else 1)
