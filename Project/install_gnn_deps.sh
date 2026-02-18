#!/bin/bash

# Installation script for GNN dependencies
# PyTorch Geometric requires special installation steps

echo "============================================================"
echo "Installing GNN Dependencies"
echo "============================================================"

# Activate virtual environment
source venv/bin/activate

echo ""
echo "Step 1: Installing PyTorch..."
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "Step 2: Installing PyTorch Geometric..."
pip install torch-geometric

echo ""
echo "Step 3: Installing other dependencies..."
pip install networkx matplotlib seaborn

echo ""
echo "============================================================"
echo "✅ Installation Complete!"
echo "============================================================"
echo ""
echo "Test the installation:"
echo "  python graph_builder.py test_preprocessed.csv"
echo "  python train_gnn.py --data test_preprocessed.csv --test-mode"
