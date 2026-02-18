# ✅ FINAL WORKING COMMANDS

## Run WITHOUT --verbose flag!

### Terminal 1 - SDN Controller
```bash
cd /mnt/3A7069D670699981/Aravind/FinalYearProject/archive
source venv/bin/activate
python sdn/run_controller.py
```

**DO NOT add `--verbose` or any other flags!**

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

## ⚠️ IMPORTANT
- Run Terminal 1 FIRST and wait for "loading application" message
- Then run Terminal 2
- Finally run Terminal 3

## Testing
In Mininet CLI (Terminal 2):
```
mininet> h3 ping h1    # Should be blocked
mininet> h1 ping h2    # Should work
```
