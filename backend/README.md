# Backend Server - README

## Overview

Node.js backend server that collects network traffic data from ESP8266 IoT devices and integrates with the GNN detection system and SDN controller.

## Features

- ✅ REST API for ESP8266 data collection
- ✅ CSV data storage
- ✅ Device registry and tracking
- ✅ Real-time statistics
- ✅ SDN controller integration
- ✅ GNN detection triggers

## Installation

```bash
cd backend
npm install
```

## Configuration

Edit `config.js` or `.env` file:

```javascript
// Server
PORT=3000

// SDN Controller
SDN_URL=http://127.0.0.1:8080

// Network
LAPTOP_IP=192.168.1.100  // Update with your laptop's IP
```

## Running

```bash
# Development
npm start

# With auto-reload (if nodemon installed)
npm run dev
```

## API Endpoints

### Data Collection

**POST /api/device/data**
```json
{
  "device_id": "ESP8266_NORMAL",
  "source_ip": "192.168.1.201",
  "destination_ip": "192.168.1.100",
  "packet_count": 25,
  "bytes_transferred": 2500,
  "duration": 5,
  "protocol": "tcp",
  "source_port": 54321,
  "destination_port": 80
}
```

### Monitoring

- `GET /api/health` - Health check
- `GET /api/stats` - Get statistics
- `GET /api/devices` - Get device registry
- `GET /api/data/recent?limit=100` - Get recent flows

### Management

- `GET /api/data/export` - Download CSV
- `POST /api/detection/trigger` - Trigger GNN detection
- `POST /api/device/block` - Block a device
- `GET /api/sdn/status` - Get SDN status
- `POST /api/data/clear` - Clear all data (testing)

## Data Storage

Data is stored in `data/network_flows.csv` with columns:
- Timestamp
- Device_ID
- Source_IP
- Destination_IP
- Packet_Count
- Bytes_Transferred
- Duration
- Protocol
- Source_Port
- Destination_Port
- Label (0=normal, 1=malicious)

## Testing

```bash
# Health check
curl http://localhost:3000/api/health

# Get statistics
curl http://localhost:3000/api/stats

# Send test data
curl -X POST http://localhost:3000/api/device/data \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "TEST_DEVICE",
    "source_ip": "192.168.1.100",
    "destination_ip": "192.168.1.1",
    "packet_count": 10,
    "bytes_transferred": 1000
  }'
```

## Troubleshooting

### Port 3000 already in use
```bash
# Find process using port 3000
lsof -i :3000

# Kill the process
kill -9 <PID>
```

### Can't connect from ESP8266
- Check firewall settings
- Verify laptop IP address
- Ensure server is listening on 0.0.0.0 (not 127.0.0.1)

## Dependencies

- express - Web framework
- cors - Cross-origin resource sharing
- body-parser - Request body parsing
- axios - HTTP client for SDN integration
- csv-writer - CSV file writing
- morgan - HTTP request logging
- dotenv - Environment variables
