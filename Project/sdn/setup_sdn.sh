#!/bin/bash
# Installation script for SDN Integration components

echo "=========================================="
echo "SDN Integration Setup"
echo "=========================================="
echo ""

# Check if running as root for some commands
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Some commands require sudo privileges"
    echo "   You may be prompted for your password"
    echo ""
fi

# Update package list
echo "📦 Updating package list..."
sudo apt update

# Install Mininet
echo ""
echo "📡 Installing Mininet..."
if command -v mn &> /dev/null; then
    echo "✅ Mininet is already installed"
    mn --version
else
    sudo apt install -y mininet
    if [ $? -eq 0 ]; then
        echo "✅ Mininet installed successfully"
    else
        echo "❌ Failed to install Mininet"
        exit 1
    fi
fi

# Install Ryu Controller (using os-ken fork for Python 3.12 compatibility)
echo ""
echo "🎮 Installing Ryu Controller (os-ken)..."
if python3 -c "import ryu" &> /dev/null; then
    echo "✅ Ryu is already installed"
elif python3 -c "import os_ken" &> /dev/null; then
    echo "✅ os-ken (Ryu fork) is already installed"
else
    echo "   Note: Using os-ken (maintained fork of Ryu for Python 3.12+)"
    pip install os-ken
    if [ $? -eq 0 ]; then
        echo "✅ os-ken installed successfully"
    else
        echo "❌ Failed to install os-ken"
        echo "   Trying alternative: install from git..."
        pip install git+https://github.com/openstack/os-ken.git
        if [ $? -eq 0 ]; then
            echo "✅ os-ken installed from git"
        else
            echo "❌ Failed to install os-ken"
            exit 1
        fi
    fi
fi

# Install requests library for GNN alert sender
echo ""
echo "📨 Installing requests library..."
pip install requests

# Verify installations
echo ""
echo "=========================================="
echo "Verifying Installations"
echo "=========================================="

echo ""
echo "1. Mininet:"
if command -v mn &> /dev/null; then
    mn --version
    echo "✅ Mininet OK"
else
    echo "❌ Mininet not found"
fi

echo ""
echo "2. Ryu Controller:"
if command -v ryu-manager &> /dev/null; then
    ryu-manager --version
    echo "✅ Ryu OK"
elif command -v ryu-manager &> /dev/null; then
    ryu-manager --version
    echo "✅ os-ken (Ryu) OK"
else
    echo "⚠️  ryu-manager not found in PATH"
    echo "   You can still use: python -m ryu.cmd.manager"
fi

echo ""
echo "3. Python packages:"
if python3 -c "import ryu" 2>/dev/null; then
    python3 -c "import ryu; print(f'   Ryu: {ryu.__version__}')"
    echo "✅ Ryu Python module OK"
elif python3 -c "import os_ken" 2>/dev/null; then
    python3 -c "import os_ken; print(f'   os-ken: {os_ken.__version__}')"
    echo "✅ os-ken (Ryu fork) Python module OK"
else
    echo "❌ Ryu/os-ken Python module not found"
fi
python3 -c "import requests; print(f'   Requests: {requests.__version__}')" 2>/dev/null && echo "✅ Requests OK" || echo "❌ Requests not found"

# Test Mininet (optional, requires sudo)
echo ""
echo "=========================================="
echo "Testing Mininet (Optional)"
echo "=========================================="
read -p "Run Mininet test? This requires sudo (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running: sudo mn --test pingall"
    sudo mn --test pingall
    if [ $? -eq 0 ]; then
        echo "✅ Mininet test passed!"
    else
        echo "⚠️  Mininet test had issues"
    fi
fi

echo ""
echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start Ryu controller:"
echo "   ryu-manager sdn/ryu_blocker.py"
echo "   (or: python -m ryu.cmd.manager sdn/ryu_blocker.py)"
echo "2. Start Mininet topology: sudo python sdn/iot_topology.py"
echo "3. Run GNN detection: python sdn/gnn_sdn_detection.py --data <your_data.csv>"
echo ""
