# IoT Security System — Deployment Guide

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│              Your Laptop (192.168.1.100)              │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │   Backend    │  │ GNN Detection│  │    SDN     │  │
│  │   Server     │◄─┤    Module    │  │ Controller │  │
│  │ (Port 3000)  │  │              │  │(Port 8080) │  │
│  └──────┬───────┘  └──────────────┘  └────────────┘  │
└─────────┼────────────────────────────────────────────┘
          │ WiFi Network (192.168.1.x)
    ┌─────┴──────┐
┌───▼────┐  ┌───▼─────┐
│ESP8266 │  │ ESP8266 │
│ Normal │  │Malicious│
│ .201   │  │  .202   │
└────────┘  └─────────┘
```

---

## Project Structure

```
FinalYearProject/
├── backend/                    # Node.js backend server
│   ├── server.js               # Main server
│   ├── config.js               # Configuration (set your laptop IP here)
│   ├── package.json
│   ├── .env
│   └── data/
│       └── network_flows.csv   # Collected device data
├── esp8266/
│   ├── normal_device/
│   │   └── normal_device.ino   # Normal device firmware
│   └── malicious_device/
│       └── malicious_device.ino # Malicious device firmware
├── Project/                    # Python GNN + SDN code
│   ├── real_time_detection.py  # GNN detection script
│   ├── train_gnn.py            # Model training
│   ├── gnn_model.py            # Model architecture
│   ├── graph_builder.py        # Graph construction
│   ├── gnn_model.pth           # Trained model weights
│   └── sdn/                    # SDN controller (see sdn/README.md)
├── DEPLOYMENT_GUIDE.md         # This file
└── reset_system.ps1            # Reset/cleanup script
```

---

## Prerequisites

### Hardware
- 1 Laptop (Windows/Linux/Mac)
- 2 × ESP8266 boards (NodeMCU, Wemos D1 Mini, etc.)
- USB cables for programming
- WiFi router (2.4 GHz)

### Software
- Node.js v14+
- Python 3.12+ with `venv`
- Arduino IDE with ESP8266 board support
- ArduinoJson library

---

## Step 1 — Network Configuration

1. Find your laptop's IP address:
   ```powershell
   # Windows
   ipconfig
   # Look for IPv4 under "Wireless LAN adapter"
   ```

2. Update these files with your laptop's IP:
   - `backend/config.js` → set `laptopIP`
   - `esp8266/normal_device/normal_device.ino` → set `serverIP` and WiFi credentials
   - `esp8266/malicious_device/malicious_device.ino` → set `serverIP` and WiFi credentials

---

## Step 2 — Install & Start Backend Server

```bash
cd backend
npm install
npm start
```

Expected output:
```
🚀 IoT Security Backend Server
📡 Server running on http://0.0.0.0:3000
✅ Ready to receive data from ESP8266 devices!
```

---

## Step 3 — Flash ESP8266 Devices

1. Open Arduino IDE
2. Load the `.ino` file for each device from the `esp8266/` folder
3. Update WiFi credentials and `serverIP` in the sketch
4. Select board: **Tools → Board → ESP8266 → NodeMCU 1.0**
5. Select the correct COM port and upload
6. Open Serial Monitor (115200 baud) to verify connection

---

## Step 4 — Start All Components

Open **4 terminal windows**:

**Terminal 1 — Backend Server:**
```bash
cd backend
npm start
```

**Terminal 2 — SDN Controller:**
```powershell
# Windows (PowerShell)
cd Project
& ".\venv\Scripts\python.exe" "sdn/run_controller.py"
```
```bash
# Linux/Mac
cd Project
source venv/bin/activate
python sdn/run_controller.py
```

**Terminal 3 — GNN Detection:**
```powershell
# Windows (PowerShell)
cd Project
& ".\venv\Scripts\python.exe" "real_time_detection.py" "--interval" "30"
```
```bash
# Linux/Mac
cd Project
source venv/bin/activate
python real_time_detection.py --interval 30
```

**Terminal 4 — Monitor (optional):**
```powershell
# Windows — watch stats every 2 seconds
while($true) { Invoke-RestMethod http://localhost:3000/api/stats | ConvertTo-Json; Start-Sleep -Seconds 2 }
```
```bash
# Linux/Mac
watch -n 2 'curl -s http://localhost:3000/api/stats | jq'
```

---

## API Reference

### Backend Server (Port 3000)

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Health check |
| POST | `/api/device/data` | Receive ESP8266 data |
| GET | `/api/data/recent?limit=N` | Get recent flows |
| GET | `/api/devices` | Get device registry |
| GET | `/api/stats` | Get statistics |
| GET | `/api/data/export` | Download CSV |
| POST | `/api/detection/trigger` | Trigger detection |
| POST | `/api/device/block` | Block a device |
| GET | `/api/sdn/status` | Get SDN status |

### SDN Controller (Port 8080)

| Method | Endpoint | Description |
|---|---|---|
| POST | `/alert` | Block an IP |
| GET | `/status` | Get blocked IPs |
| POST | `/unblock` | Unblock an IP |

---

## Expected Device Behavior

| Device | Send Interval | Packet Count | Bytes | Expected Result |
|---|---|---|---|---|
| ESP8266_NORMAL | Every 5 sec | 10–50 | 500–5,000 | ✅ Not blocked |
| ESP8266_MALICIOUS | Every 2 sec | 500–2,000 | 50KB–200KB | 🚨 Detected & blocked |

---

## Troubleshooting

**ESP8266 can't connect to WiFi**
- Check credentials and ensure the router is 2.4 GHz (ESP8266 doesn't support 5 GHz)

**Backend not receiving data**
- Verify laptop IP in `config.js` and the `.ino` files
- Allow port 3000 through Windows Firewall
- Test: `curl http://YOUR_LAPTOP_IP:3000/api/health`

**Port already in use (Windows)**
```powershell
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**GNN detection not running**
- Ensure backend has collected at least 20 samples
- Check `backend/data/network_flows.csv` exists
- Verify Python deps: `pip install os-ken torch torch-geometric`

**SDN controller not blocking**
- Ensure Terminal 2 shows "loading application" before starting Terminal 3
- Check: `curl http://localhost:8080/status`
- See `Project/sdn/README.md` for detailed SDN troubleshooting

---

## Data Flow

```
1. ESP8266 devices → HTTP POST to backend every 2–5 sec
2. Backend stores data in backend/data/network_flows.csv
3. GNN detection script monitors CSV every 30 sec
4. GNN builds a graph and runs malicious node detection
5. Malicious device IP sent to SDN controller
6. SDN controller installs OpenFlow drop rules
7. Blocked device receives 403 Forbidden from backend
```
