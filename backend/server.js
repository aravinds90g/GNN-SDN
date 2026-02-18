const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const morgan = require('morgan');
const fs = require('fs');
const path = require('path');
const axios = require('axios');
const { createObjectCsvWriter } = require('csv-writer');
const config = require('./config');

const app = express();

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(morgan('combined'));

// Ensure data directories exist
if (!fs.existsSync(config.data.directory)) {
  fs.mkdirSync(config.data.directory, { recursive: true });
}
if (!fs.existsSync(config.data.logDirectory)) {
  fs.mkdirSync(config.data.logDirectory, { recursive: true });
}

// CSV Writer for network flows
const csvWriter = createObjectCsvWriter({
  path: config.data.csvFile,
  header: [
    { id: 'timestamp', title: 'Timestamp' },
    { id: 'device_id', title: 'Device_ID' },
    { id: 'source_ip', title: 'Source_IP' },
    { id: 'destination_ip', title: 'Destination_IP' },
    { id: 'packet_count', title: 'Packet_Count' },
    { id: 'bytes_transferred', title: 'Bytes_Transferred' },
    { id: 'duration', title: 'Duration' },
    { id: 'protocol', title: 'Protocol' },
    { id: 'source_port', title: 'Source_Port' },
    { id: 'destination_port', title: 'Destination_Port' },
    { id: 'label', title: 'Label' }
  ],
  append: true
});

// In-memory storage for recent data
let recentFlows = [];
let deviceRegistry = {};
let blockedDevices = new Set();

// Statistics
let stats = {
  totalFlows: 0,
  normalFlows: 0,
  maliciousFlows: 0,
  detectionRuns: 0,
  blockedIPs: []
};

// ============================================
// API ENDPOINTS
// ============================================

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'running',
    timestamp: new Date().toISOString(),
    stats: stats
  });
});

// Receive data from ESP8266 devices
app.post('/api/device/data', async (req, res) => {
  try {
    const {
      device_id,
      source_ip,
      destination_ip,
      packet_count,
      bytes_transferred,
      duration,
      protocol,
      source_port,
      destination_port
    } = req.body;

    // Validate required fields
    if (!device_id || !source_ip || !destination_ip) {
      return res.status(400).json({
        error: 'Missing required fields',
        required: ['device_id', 'source_ip', 'destination_ip']
      });
    }

    // 🚫 SDN BLOCKING: Check if IP is blocked
    if (blockedDevices.has(source_ip)) {
      // Silently reject blocked IPs (no log spam)
      return res.status(403).json({
        error: 'Forbidden',
        message: 'This IP address has been blocked by SDN controller',
        ip: source_ip,
        reason: 'Malicious activity detected'
      });
    }

    // Determine label based on device configuration
    let label = 0; // Default: normal
    if (device_id === config.devices.malicious.id) {
      label = 1; // Malicious
    }

    // Create flow record
    const flowRecord = {
      timestamp: new Date().toISOString(),
      device_id: device_id,
      source_ip: source_ip,
      destination_ip: destination_ip,
      packet_count: packet_count || 0,
      bytes_transferred: bytes_transferred || 0,
      duration: duration || 0,
      protocol: protocol || 'tcp',
      source_port: source_port || 0,
      destination_port: destination_port || 80,
      label: label
    };

    // Store in memory
    recentFlows.push(flowRecord);
    
    // Keep only last 1000 flows in memory
    if (recentFlows.length > 1000) {
      recentFlows = recentFlows.slice(-1000);
    }

    // Write to CSV
    await csvWriter.writeRecords([flowRecord]);

    // Update statistics
    stats.totalFlows++;
    if (label === 0) {
      stats.normalFlows++;
    } else {
      stats.maliciousFlows++;
    }

    // Register device
    if (!deviceRegistry[device_id]) {
      deviceRegistry[device_id] = {
        first_seen: new Date().toISOString(),
        ip: source_ip,
        flows: 0
      };
    }
    deviceRegistry[device_id].flows++;
    deviceRegistry[device_id].last_seen = new Date().toISOString();

    console.log(`📊 Flow received from ${device_id} (${source_ip}): ${packet_count} packets, ${bytes_transferred} bytes`);

    res.json({
      success: true,
      message: 'Data received',
      flow_id: recentFlows.length,
      label: label === 0 ? 'normal' : 'malicious'
    });

  } catch (error) {
    console.error('Error processing device data:', error);
    res.status(500).json({
      error: 'Internal server error',
      message: error.message
    });
  }
});

// Get recent flows
app.get('/api/data/recent', (req, res) => {
  const limit = parseInt(req.query.limit) || 100;
  const flows = recentFlows.slice(-limit);
  
  res.json({
    count: flows.length,
    flows: flows
  });
});

// Get device registry
app.get('/api/devices', (req, res) => {
  res.json({
    count: Object.keys(deviceRegistry).length,
    devices: deviceRegistry
  });
});

// Get statistics
app.get('/api/stats', (req, res) => {
  res.json({
    ...stats,
    recentFlowsCount: recentFlows.length,
    registeredDevices: Object.keys(deviceRegistry).length,
    csvFileSize: fs.existsSync(config.data.csvFile) 
      ? fs.statSync(config.data.csvFile).size 
      : 0
  });
});

// Export data as CSV
app.get('/api/data/export', (req, res) => {
  if (!fs.existsSync(config.data.csvFile)) {
    return res.status(404).json({ error: 'No data available' });
  }

  res.download(config.data.csvFile, 'network_flows.csv');
});

// Trigger GNN detection manually
app.post('/api/detection/trigger', async (req, res) => {
  try {
    console.log('🔍 Manual detection triggered');
    
    if (recentFlows.length < config.gnn.minSamplesForDetection) {
      return res.json({
        success: false,
        message: `Not enough samples. Need at least ${config.gnn.minSamplesForDetection}, have ${recentFlows.length}`
      });
    }

    // This would trigger the Python GNN detection script
    // For now, return a placeholder response
    res.json({
      success: true,
      message: 'Detection triggered',
      samples: recentFlows.length,
      note: 'Use real_time_detection.py for actual GNN detection'
    });

  } catch (error) {
    console.error('Error triggering detection:', error);
    res.status(500).json({
      error: 'Detection failed',
      message: error.message
    });
  }
});

// Block a device (send to SDN controller)
app.post('/api/device/block', async (req, res) => {
  try {
    const { ip, device_id } = req.body;

    if (!ip) {
      return res.status(400).json({ error: 'IP address required' });
    }

    console.log(`🚫 Blocking device: ${ip} (${device_id || 'unknown'})`);

    // Send block command to SDN controller
    try {
      const response = await axios.post(
        `${config.sdn.url}${config.sdn.endpoints.alert}`,
        { ip: ip },
        { timeout: 5000 }
      );

      blockedDevices.add(ip);
      stats.blockedIPs.push({
        ip: ip,
        device_id: device_id,
        timestamp: new Date().toISOString()
      });

      console.log(`✅ Device ${ip} blocked successfully`);

      res.json({
        success: true,
        message: `Device ${ip} blocked`,
        sdn_response: response.data
      });

    } catch (sdnError) {
      console.error('SDN controller error:', sdnError.message);
      res.status(503).json({
        error: 'SDN controller unavailable',
        message: sdnError.message
      });
    }

  } catch (error) {
    console.error('Error blocking device:', error);
    res.status(500).json({
      error: 'Block failed',
      message: error.message
    });
  }
});

// Get SDN controller status
app.get('/api/sdn/status', async (req, res) => {
  try {
    const response = await axios.get(
      `${config.sdn.url}${config.sdn.endpoints.status}`,
      { timeout: 5000 }
    );

    res.json({
      success: true,
      sdn_status: response.data
    });

  } catch (error) {
    res.status(503).json({
      error: 'SDN controller unavailable',
      message: error.message
    });
  }
});

// Clear all data (for testing)
app.post('/api/data/clear', (req, res) => {
  recentFlows = [];
  deviceRegistry = {};
  stats = {
    totalFlows: 0,
    normalFlows: 0,
    maliciousFlows: 0,
    detectionRuns: 0,
    blockedIPs: []
  };

  console.log('🗑️  All data cleared');

  res.json({
    success: true,
    message: 'All data cleared'
  });
});

// ============================================
// START SERVER
// ============================================

const PORT = config.server.port;
const HOST = config.server.host;

app.listen(PORT, HOST, () => {
  console.log('\n' + '='.repeat(60));
  console.log('🚀 IoT Security Backend Server');
  console.log('='.repeat(60));
  console.log(`📡 Server running on http://${HOST}:${PORT}`);
  console.log(`📊 Data directory: ${config.data.directory}`);
  console.log(`🔗 SDN Controller: ${config.sdn.url}`);
  console.log(`🧠 GNN Detection: ${config.gnn.pythonPath}`);
  console.log('='.repeat(60));
  console.log('\n📋 Available Endpoints:');
  console.log('  GET  /api/health          - Health check');
  console.log('  POST /api/device/data     - Receive ESP8266 data');
  console.log('  GET  /api/data/recent     - Get recent flows');
  console.log('  GET  /api/devices         - Get device registry');
  console.log('  GET  /api/stats           - Get statistics');
  console.log('  GET  /api/data/export     - Export CSV');
  console.log('  POST /api/detection/trigger - Trigger GNN detection');
  console.log('  POST /api/device/block    - Block a device');
  console.log('  GET  /api/sdn/status      - Get SDN status');
  console.log('  POST /api/data/clear      - Clear all data');
  console.log('\n✅ Ready to receive data from ESP8266 devices!\n');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n\n🛑 Shutting down server...');
  process.exit(0);
});
