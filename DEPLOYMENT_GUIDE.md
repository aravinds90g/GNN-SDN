# 🚀 Real-World IoT Security System - Deployment Guide

## Overview

This guide explains how to deploy the complete IoT security system with **real ESP8266 devices** on a single laptop.

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│              Your Laptop (192.168.1.100)              │
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Backend    │  │ GNN Detection│  │    SDN     │ │
│  │   Server     │◄─┤    Module    │  │ Controller │ │
│  │ (Port 3000)  │  │              │  │(Port 8080) │ │
│  └──────┬───────┘  └──────────────┘  └────────────┘ │
└─────────┼──────────────────────────────────────────┘
          │
          │ WiFi Network (192.168.1.x)
          │
    ┌─────┴──────┐
    │            │
┌───▼────┐  ┌───▼────┐
│ESP8266 │  │ESP8266 │
│ Normal │  │Malicious│
│.201    │  │  .202  │
└────────┘  └────────┘
```

---

## Prerequisites

### Hardware

- 1 Laptop (Linux/Windows/Mac)
- 2 ESP8266 boards (NodeMCU, Wemos D1 Mini, etc.)
- USB cables for programming ESP8266
- WiFi router

### Software

- Node.js (v14+)
- Python 3.12+ with virtual environment
- Arduino IDE with ESP8266 board support (v1.8.13+)
- ArduinoJson library

**Windows Users**: Ensure you have:

- PowerShell 5.0+ or Command Prompt
- Git Bash (optional, for Linux-like commands)

---

## Setup Instructions

### Step 1: Network Configuration

1. **Find your laptop's IP address:**

   **Windows (Command Prompt or PowerShell):**

   ```bash
   ipconfig
   ```

   Look for IPv4 Address under "Wireless LAN adapter" or "Ethernet adapter"

   **Linux/Mac:**

   ```bash
   ip addr show
   # or
   ifconfig
   ```

2. **Update configuration files:**
   - Edit `backend/config.js`: Set `laptopIP` to your laptop's IP
   - Edit `esp8266/normal_device/normal_device.ino`: Update WiFi credentials and server IP
   - Edit `esp8266/malicious_device/malicious_device.ino`: Update WiFi credentials and server IP

---

### Step 2: Install Backend Server

**Windows (Command Prompt or PowerShell):**

```bash
cd backend
npm install
npm start
```

**Linux/Mac:**

```bash
cd backend
npm install
npm start
```

You should see:

```
🚀 IoT Security Backend Server
📡 Server running on http://0.0.0.0:3000
✅ Ready to receive data from ESP8266 devices!
```

---

### Step 3: Flash ESP8266 Devices

#### Normal Device

1. Open Arduino IDE
2. Load `esp8266/normal_device/normal_device.ino`
3. Update WiFi credentials:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   const char* serverIP = "192.168.1.100";  // Your laptop IP
   ```
4. Select board: **Tools → Board → ESP8266 → NodeMCU 1.0**
5. Select port: **Tools → Port:**
   - **Windows**: Select COM port (e.g., COM3, COM4)
   - **Linux**: Select /dev/ttyUSB0 or /dev/ttyUSB1
   - **Mac**: Select /dev/cu.usbserial-xxx
6. Upload the sketch (Sketch → Upload or Ctrl+U)
7. Open Serial Monitor (115200 baud) to verify connection

#### Malicious Device

1. Load `esp8266/malicious_device/malicious_device.ino`
2. Update WiFi credentials (same as above)
3. Upload to second ESP8266
4. Verify in Serial Monitor

---

### Step 4: Start All Components

You need **4 terminals/PowerShell windows**:

#### Terminal 1: Backend Server

**Windows/Linux/Mac:**

```bash
cd backend
npm start
```

#### Terminal 2: SDN Controller

**Windows (Command Prompt):**

```bash
cd archive
venv\Scripts\activate
python sdn/run_controller.py
```

**Windows (PowerShell) - Use this if above doesn't work:**

```powershell
cd archive
&".\venv\Scripts\python.exe" "sdn/run_controller.py"
```

**Linux/Mac:**

```bash
cd archive
source venv/bin/activate
python sdn/run_controller.py
```

#### Terminal 3: Real-Time Detection

**Windows (Command Prompt):**

```bash
cd archive
venv\Scripts\activate
python real_time_detection.py --interval 30
```

**Windows (PowerShell) - Use this if above doesn't work:**

```powershell
cd archive
&".\venv\Scripts\python.exe" "real_time_detection.py" "--interval" "30"
```

**Linux/Mac:**

```bash
cd archive
source venv/bin/activate
python real_time_detection.py --interval 30
```

#### Terminal 4: Monitoring (Optional)

**Windows (PowerShell):**

```powershell
# Watch the data file
while($true) { Get-ChildItem \backend\data\network_flows.csv | Select-Object Length; Start-Sleep -Seconds 2 }

# Or monitor backend stats
while($true) { Invoke-RestMethod http://localhost:3000/api/stats | ConvertTo-Json; Start-Sleep -Seconds 2 }
```

**Windows (Command Prompt):**

```bash
REM Simple monitoring - check file size every 2 seconds
for /l %N in () do @(mode con /t 0 & dir \backend\data\network_flows.csv & timeout /t 2 /nobreak)
```

**Linux/Mac:**

```bash
watch -n 2 'wc -l backend/data/network_flows.csv'
watch -n 2 'curl -s http://localhost:3000/api/stats | jq'
```

---

## Testing the System

### 1. Verify ESP8266 Connectivity

Check Serial Monitor on both devices. You should see:

```
✅ WiFi connected!
📍 IP Address: 192.168.1.201
📡 Sending data to: http://192.168.1.100:3000/api/device/data
✅ Server response (200): {"success":true,...}
```

### 2. Check Backend Data Collection

```bash
# Get statistics
curl http://localhost:3000/api/stats

# Get recent flows
curl http://localhost:3000/api/data/recent?limit=10

# Get device registry
curl http://localhost:3000/api/devices
```

### 3. Monitor GNN Detection

Watch Terminal 3 for detection output:

```
🔍 RUNNING DETECTION #1
Total nodes: 2
Malicious detected: 1

🚨 ALERT: 1 malicious device(s) detected!
   Device: Unknown-Node-1
   Node Index: 1
   Prediction: malicious
   ✅ Blocked 192.168.1.202 via SDN controller
```

### 4. Verify SDN Blocking

```bash
# Check SDN controller status
curl http://localhost:8080/status

# Should show blocked IPs
```

---

## API Endpoints

### Backend Server (Port 3000)

| Method | Endpoint                   | Description          |
| ------ | -------------------------- | -------------------- |
| GET    | `/api/health`              | Health check         |
| POST   | `/api/device/data`         | Receive ESP8266 data |
| GET    | `/api/data/recent?limit=N` | Get recent flows     |
| GET    | `/api/devices`             | Get device registry  |
| GET    | `/api/stats`               | Get statistics       |
| GET    | `/api/data/export`         | Download CSV         |
| POST   | `/api/detection/trigger`   | Trigger detection    |
| POST   | `/api/device/block`        | Block a device       |
| GET    | `/api/sdn/status`          | Get SDN status       |

### SDN Controller (Port 8080)

| Method | Endpoint   | Description     |
| ------ | ---------- | --------------- |
| POST   | `/alert`   | Block an IP     |
| GET    | `/status`  | Get blocked IPs |
| POST   | `/unblock` | Unblock an IP   |

---

## Troubleshooting

### ESP8266 Can't Connect to WiFi

- Check WiFi credentials
- Ensure 2.4GHz WiFi (ESP8266 doesn't support 5GHz)
- Check signal strength

### Backend Server Not Receiving Data

- Verify laptop IP address
- Check firewall settings (allow port 3000)
- Test with curl: `curl http://YOUR_LAPTOP_IP:3000/api/health`

### Port Already in Use (Windows)

```powershell
# Find process using port 3000
netstat -ano | findstr :3000

# Kill process (replace PID with actual number)
taskkill /PID <PID> /F

# For port 8080
netstat -ano | findstr :8080
taskkill /PID <PID> /F
```

### GNN Detection Not Running

- Ensure backend has collected at least 20 samples
- Check CSV file exists: `backend/data/network_flows.csv`
- Verify Python dependencies are installed: `pip list | findstr "os-ken"`
- Run test: `python -c "import os_ken; print('os-ken installed')"` 

### SDN Controller Not Blocking

- Check SDN controller is running on port 8080
- Verify network connectivity
- Check logs for errors

---

## Expected Behavior

1. **Normal Device (ESP8266_NORMAL)**:
   - Sends data every 5 seconds
   - Low packet counts (10-50 packets)
   - Low bytes (500-5000 bytes)
   - Should NOT be blocked

2. **Malicious Device (ESP8266_MALICIOUS)**:
   - Sends data every 2 seconds (faster)
   - High packet counts (500-2000 packets)
   - High bytes (50KB-200KB)
   - Cycles through attack patterns (DoS, port scan, burst)
   - Should be DETECTED and BLOCKED

---

## Data Flow

```
1. ESP8266 devices send HTTP POST to backend every 2-5 seconds
2. Backend stores data in CSV file
3. Real-time detection script monitors CSV file every 30 seconds
4. When enough data collected, GNN builds graph and runs detection
5. Malicious devices identified and sent to SDN controller
6. SDN controller blocks malicious IPs
7. Blocked devices can no longer communicate
```

---

## Files Created

```
FinalYearProject/
├── backend/
│   ├── server.js              # Main backend server
│   ├── config.js              # Configuration
│   ├── package.json           # Dependencies
│   ├── .env                   # Environment variables
│   └── data/
│       └── network_flows.csv  # Collected data
├── esp8266/
│   ├── normal_device/
│   │   └── normal_device.ino  # Normal device firmware
│   └── malicious_device/
│       └── malicious_device.ino # Malicious device firmware
└── archive/
    └── real_time_detection.py # GNN detection script
```

---

## Next Steps

1. **Collect Real Data**: Run system for 10-30 minutes
2. **Analyze Results**: Check detection accuracy
3. **Fine-tune Model**: Adjust GNN parameters if needed
4. **Scale Up**: Add more ESP8266 devices
5. **Deploy**: Move to production environment

---

## Quick Start Commands

**Windows (Command Prompt or PowerShell):**

```bash
# Terminal 1: Backend
cd backend & npm start

# Terminal 2: SDN Controller
cd archive & venv\Scripts\activate & python sdn/run_controller.py

# Terminal 3: GNN Detection
cd archive & venv\Scripts\activate & python real_time_detection.py

# Terminal 4: Monitor (PowerShell)
while($true) { Invoke-RestMethod http://localhost:3000/api/stats | ConvertTo-Json; Start-Sleep -Seconds 2 }
```

**Linux/Mac:**

```bash
# Terminal 1: Backend
cd backend && npm start

# Terminal 2: SDN Controller
cd archive && source venv/bin/activate && python sdn/run_controller.py

# Terminal 3: GNN Detection
cd archive && source venv/bin/activate && python real_time_detection.py

# Terminal 4: Monitor
watch -n 2 'curl -s http://localhost:3000/api/stats | jq'
```

---

**🎉 You now have a complete, real-world IoT security system!**
