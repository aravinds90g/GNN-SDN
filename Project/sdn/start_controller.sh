#!/bin/bash
# Start Ryu/os-ken Controller

cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive

# Activate virtual environment
source venv/bin/activate

echo "🚀 Starting SDN Controller..."
echo ""

# Run the controller using Python launcher
python sdn/run_controller.py
