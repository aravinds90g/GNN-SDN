// Configuration for IoT Security System
const path = require("path");
const platform = process.platform; // 'win32', 'linux', 'darwin'

// Determine Python path based on OS
const getPythonPath = () => {
  if (process.env.PYTHON_PATH) return process.env.PYTHON_PATH;

  if (platform === "win32") {
    return path.join(
      __dirname,
      "..",
      "archive",
      "venv",
      "Scripts",
      "python.exe",
    );
  } else {
    return path.join(__dirname, "..", "archive", "venv", "bin", "python");
  }
};

module.exports = {
  // Server Configuration
  server: {
    port: process.env.PORT || 3000,
    host: "0.0.0.0", // Listen on all interfaces
  },

  // SDN Controller Configuration
  sdn: {
    url: process.env.SDN_URL || "http://127.0.0.1:8080",
    endpoints: {
      alert: "/alert",
      status: "/status",
      unblock: "/unblock",
    },
  },

  // GNN Detection Configuration
  gnn: {
    pythonPath: getPythonPath(),
    scriptPath:
      process.env.GNN_SCRIPT ||
      path.join(__dirname, "..", "archive", "real_time_detection.py"),
    modelPath:
      process.env.GNN_MODEL ||
      path.join(__dirname, "..", "archive", "gnn_model.pth"),
    detectionInterval: 30000, // Run detection every 30 seconds
    minSamplesForDetection: 10, // Minimum samples before running detection
  },

  // Data Storage Configuration
  data: {
    directory: "./data",
    csvFile: "./data/network_flows.csv",
    logDirectory: "./logs",
    maxFileSize: 100 * 1024 * 1024, // 100MB
    rotateOnSize: true,
  },

  // Network Configuration
  network: {
    subnet: "192.168.1.0/24",
    laptopIP: "10.181.225.181",
    normalDeviceIP: "192.168.1.201",
    maliciousDeviceIP: "192.168.1.202",
  },

  // Device Configuration
  devices: {
    normal: {
      id: "ESP8266_NORMAL",
      expectedIP: "192.168.1.201",
      label: 0, // Normal
    },
    malicious: {
      id: "ESP8266_MALICIOUS",
      expectedIP: "192.168.1.202",
      label: 1, // Attack
    },
  },

  // Detection Thresholds
  thresholds: {
    maxPacketsPerSecond: 1000,
    maxBytesPerSecond: 1000000,
    suspiciousPortScanThreshold: 10,
    ddosPacketThreshold: 500,
  },
};
