# ✅ WORKING COMMANDS FOR GNN-SDN SYSTEM

## 🚀 How to Start the System

### Terminal 1 - SDN Controller
```bash
cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive
source venv/bin/activate
python sdn/run_controller.py
```

### Terminal 2 - Mininet Network
```bash
cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive
sudo python3 sdn/iot_topology.py
```

### Terminal 3 - GNN Detection
```bash
cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive
source venv/bin/activate
python sdn/gnn_sdn_detection.py --data test_preprocessed.csv
```

## 📝 Quick Reference

View commands anytime:
```bash
./sdn/COMMANDS.sh
```

## 🧪 Testing

In Mininet CLI (Terminal 2):
```bash
mininet> h3 ping h1    # Should be blocked
mininet> h1 ping h2    # Should work
```

## ⚠️ Important Notes

- **Don't use `ryu-manager`** - It's not available with os-ken
- **Use `python sdn/run_controller.py`** instead
- **Use `python3` with sudo** for Mininet (not `python`)
- **Activate venv** before running controller and GNN detection

## 🔧 Troubleshooting

**"ModuleNotFoundError: No module named 'ryu'"**
- This is expected! We're using os-ken, not ryu
- Use: `python sdn/run_controller.py`

**"ryu-manager: command not found"**
- Don't use ryu-manager
- Use: `python sdn/run_controller.py`

**"sudo: python: command not found"**
- Use `python3` instead of `python` with sudo
- Command: `sudo python3 sdn/iot_topology.py`
