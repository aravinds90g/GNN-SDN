# SDN Integration - GNN-Based IoT Security

## Overview

This module integrates the GNN attack detection model with an SDN (Software-Defined Networking) controller to automatically block malicious IoT devices at the network level.

```
Mininet (Network Simulation)
    ↓ traffic flows
OpenFlow Switch
    ↓ flow statistics
os-ken/Ryu Controller (SDN)
    ↑ blocking rules
GNN Model (Detection)
    ↑ network data
```

---

## Files

| File | Description |
|---|---|
| `ryu_blocker.py` | SDN controller with REST API for blocking IPs |
| `iot_topology.py` | Mininet network topology with IoT devices |
| `gnn_sdn_detection.py` | Integrated GNN detection + SDN blocking script |
| `gnn_alert_sender.py` | Alert communication module |
| `test_integration.py` | Test suite for the integration |
| `run_controller.py` | Entry point to start the SDN controller |
| `setup_sdn.sh` | Dependency installation script |
| `install_osken.sh` | Installs os-ken (Python 3.12+ compatible Ryu fork) |
| `quick_start.sh` | Quick start helper script |
| `COMMANDS.sh` | Reference command list |

---

## Setup

### 1. Install Dependencies

```bash
cd Project
chmod +x sdn/setup_sdn.sh
./sdn/setup_sdn.sh
```

> **Note on Ryu / os-ken**: The official `ryu` package has compatibility issues with Python 3.12+.
> Use **os-ken** instead — a maintained OpenStack fork that is fully API-compatible:
> ```bash
> pip install os-ken
> ```
> Verify: `python -c "import os_ken; print(os_ken.__version__)"`

---

## Running the System (3 Terminals)

**Terminal 1 — SDN Controller:**
```bash
cd Project
source venv/bin/activate
python sdn/run_controller.py
```
Wait for the "loading application" message before proceeding.

**Terminal 2 — Mininet Network:**
```bash
cd Project
sudo python3 sdn/iot_topology.py
```

**Terminal 3 — GNN Detection:**
```bash
cd Project
source venv/bin/activate
python sdn/gnn_sdn_detection.py --data test_preprocessed.csv
```

---

## Testing

In the Mininet CLI (Terminal 2):
```bash
mininet> h3 ping h1    # Should be BLOCKED ❌
mininet> h1 ping h2    # Should WORK ✅
```

Run the integration test suite:
```bash
python sdn/test_integration.py
```

---

## REST API (SDN Controller — Port 8080)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/alert` | Block an IP |
| GET | `/status` | List blocked IPs |
| POST | `/unblock` | Unblock an IP |

**Example:**
```bash
# Block a device
curl -X POST http://127.0.0.1:8080/alert \
  -H "Content-Type: application/json" \
  -d '{"ip": "192.168.1.5"}'

# Check blocked IPs
curl http://127.0.0.1:8080/status
```

---

## Network Topology

| Device | IP Address | Role |
|---|---|---|
| h1 | 192.168.1.2 | Normal IoT Device |
| h2 | 192.168.1.3 | Normal IoT Device |
| h3 | 192.168.1.5 | Malicious Device |
| h4 | 192.168.1.6 | Normal IoT Device |
| s1 | — | OpenFlow Switch |
| c0 | 127.0.0.1:6633 | SDN Controller |

---

## Troubleshooting

**"Cannot connect to SDN controller"**
- Ensure Terminal 1 is running and shows "loading application"
- Check port 8080: `lsof -i :8080`

**"ModuleNotFoundError: No module named 'ryu'"**
- Use os-ken instead: `pip install os-ken`
- Run via: `python sdn/run_controller.py` (not `ryu-manager`)

**"sudo: python: command not found"**
- Use `python3` with sudo: `sudo python3 sdn/iot_topology.py`

**"Permission denied" for Mininet**
- Mininet requires sudo: `sudo python3 sdn/iot_topology.py`
