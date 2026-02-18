#!/bin/bash
# Alternative installation method for Ryu using os-ken

echo "Installing os-ken (Ryu fork for Python 3.12+)..."
echo ""

# Install os-ken
pip install os-ken

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ os-ken installed successfully!"
    echo ""
    echo "Note: os-ken is a maintained fork of Ryu that works with Python 3.12+"
    echo "All Ryu commands work the same way:"
    echo ""
    echo "  ryu-manager sdn/ryu_blocker.py"
    echo ""
    echo "Or use the Python module directly:"
    echo "  python -m ryu.cmd.manager sdn/ryu_blocker.py"
    echo ""
else
    echo ""
    echo "❌ Failed to install os-ken"
    echo "Trying alternative installation from git..."
    echo ""
    
    pip install git+https://github.com/openstack/os-ken.git
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ os-ken installed from git!"
    else
        echo ""
        echo "❌ Installation failed"
        echo ""
        echo "Manual installation steps:"
        echo "1. git clone https://github.com/openstack/os-ken.git"
        echo "2. cd os-ken"
        echo "3. pip install ."
    fi
fi
