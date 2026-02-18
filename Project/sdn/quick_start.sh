#!/bin/bash
# Quick Start Script for GNN-SDN Integration
# This script helps you start all components in the correct order

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     GNN-SDN Integration - Quick Start Helper              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if running from correct directory
if [ ! -d "sdn" ]; then
    echo "❌ Error: Please run this script from the 'archive' directory"
    echo "   cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
echo "📋 Checking Prerequisites..."
echo ""

MISSING=0

if command_exists ryu-manager; then
    echo "✅ Ryu Controller installed"
else
    echo "❌ Ryu Controller not found"
    MISSING=1
fi

if command_exists mn; then
    echo "✅ Mininet installed"
else
    echo "❌ Mininet not found"
    MISSING=1
fi

if python3 -c "import torch" 2>/dev/null; then
    echo "✅ PyTorch installed"
else
    echo "❌ PyTorch not found"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "⚠️  Missing dependencies detected!"
    echo "   Run: ./sdn/setup_sdn.sh"
    exit 1
fi

echo ""
echo "✅ All prerequisites met!"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Starting GNN-SDN Integration System"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "You need to run the following commands in 3 SEPARATE terminals:"
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ TERMINAL 1 - Ryu SDN Controller                           │"
echo "└────────────────────────────────────────────────────────────┘"
echo "cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive"
echo "ryu-manager sdn/ryu_blocker.py"
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ TERMINAL 2 - Mininet Network                              │"
echo "└────────────────────────────────────────────────────────────┘"
echo "cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive"
echo "sudo python sdn/iot_topology.py"
echo ""
echo "┌────────────────────────────────────────────────────────────┐"
echo "│ TERMINAL 3 - GNN Detection                                │"
echo "└────────────────────────────────────────────────────────────┘"
echo "cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive"
echo "python sdn/gnn_sdn_detection.py --data test_preprocessed.csv"
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Testing Commands (in Mininet CLI)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "# Test if malicious device (h3) is blocked:"
echo "mininet> h3 ping h1"
echo ""
echo "# Test normal communication:"
echo "mininet> h1 ping h2"
echo ""
echo "════════════════════════════════════════════════════════════"
echo ""
read -p "Press Enter to continue..."
