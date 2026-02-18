/*
 * ESP8266 Malicious IoT Device (Attack Simulator)
 * 
 * This device simulates a malicious IoT device that exhibits
 * attack-like behavior patterns for testing the GNN detection system.
 * 
 * Attack patterns simulated:
 * - High packet rates (DoS-like)
 * - Port scanning behavior
 * - Unusual traffic patterns
 * - Burst traffic
 * 
 * Hardware: ESP8266 (NodeMCU, Wemos D1 Mini, etc.)
 * 
 * ⚠️  WARNING: This is for TESTING ONLY in a controlled environment!
 * 
 * Setup:
 * 1. Install ESP8266 board in Arduino IDE
 * 2. Install ArduinoJson library
 * 3. Update WiFi credentials below
 * 4. Update server IP address
 * 5. Upload to ESP8266
 */

#include <ESP8266WiFi.h>
#include <ESP8266HTTPClient.h>
#include <WiFiClient.h>
#include <ArduinoJson.h>

// ============================================
// CONFIGURATION - UPDATE THESE!
// ============================================

// WiFi credentials
const char* ssid = "YOUR_WIFI_SSID";           // Change this
const char* password = "YOUR_WIFI_PASSWORD";   // Change this

// Server configuration
const char* serverIP = "192.168.1.100";        // Your laptop IP
const int serverPort = 3000;
const char* serverEndpoint = "/api/device/data";

// Device configuration
const char* deviceID = "ESP8266_MALICIOUS";
const char* deviceIP = "192.168.1.202";        // This device's static IP

// Attack simulation configuration
const unsigned long sendInterval = 2000;       // Send data every 2 seconds (faster than normal)
unsigned long lastSendTime = 0;

// Attack patterns
enum AttackPattern {
  DOS_ATTACK,
  PORT_SCAN,
  BURST_TRAFFIC,
  RANDOM_ATTACK
};

AttackPattern currentAttack = DOS_ATTACK;
int attackCycle = 0;

// ============================================
// GLOBAL VARIABLES
// ============================================

WiFiClient wifiClient;
HTTPClient http;

unsigned long packetCount = 0;
unsigned long bytesTransferred = 0;
unsigned long sessionStartTime = 0;

// ============================================
// SETUP
// ============================================

void setup() {
  Serial.begin(115200);
  delay(1000);
  
  Serial.println("\n\n");
  Serial.println("========================================");
  Serial.println("ESP8266 Malicious Device (Attack Simulator)");
  Serial.println("⚠️  FOR TESTING ONLY!");
  Serial.println("========================================");
  
  // Connect to WiFi
  connectToWiFi();
  
  sessionStartTime = millis();
  
  Serial.println("✅ Setup complete!");
  Serial.println("🚨 Ready to simulate attack patterns");
  Serial.println("========================================\n");
}

// ============================================
// MAIN LOOP
// ============================================

void loop() {
  // Check WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("⚠️  WiFi disconnected. Reconnecting...");
    connectToWiFi();
  }
  
  // Send data at regular intervals
  unsigned long currentTime = millis();
  if (currentTime - lastSendTime >= sendInterval) {
    lastSendTime = currentTime;
    
    // Cycle through different attack patterns
    attackCycle++;
    if (attackCycle % 10 == 0) {
      currentAttack = (AttackPattern)((currentAttack + 1) % 4);
      Serial.println("\n🔄 Switching attack pattern...\n");
    }
    
    // Simulate malicious traffic
    simulateMaliciousTraffic();
    
    // Send data to server
    sendDataToServer();
  }
  
  delay(50);  // Shorter delay for more aggressive behavior
}

// ============================================
// FUNCTIONS
// ============================================

void connectToWiFi() {
  Serial.print("🔌 Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\n✅ WiFi connected!");
    Serial.print("📍 IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("📶 Signal Strength: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println("\n❌ WiFi connection failed!");
  }
}

void simulateMaliciousTraffic() {
  switch (currentAttack) {
    case DOS_ATTACK:
      simulateDosAttack();
      break;
    case PORT_SCAN:
      simulatePortScan();
      break;
    case BURST_TRAFFIC:
      simulateBurstTraffic();
      break;
    case RANDOM_ATTACK:
      simulateRandomAttack();
      break;
  }
}

void simulateDosAttack() {
  // DoS: Very high packet rate, large byte count
  packetCount = random(500, 1500);           // 500-1500 packets (very high)
  bytesTransferred = random(50000, 150000);  // 50-150 KB (very high)
  
  Serial.println("🚨 Attack Pattern: DoS");
  Serial.print("   Packets: ");
  Serial.print(packetCount);
  Serial.println(" (HIGH)");
  Serial.print("   Bytes: ");
  Serial.print(bytesTransferred);
  Serial.println(" (HIGH)");
}

void simulatePortScan() {
  // Port Scan: Multiple connections to different ports
  packetCount = random(100, 300);            // Moderate packets
  bytesTransferred = random(1000, 5000);     // Low bytes per connection
  
  Serial.println("🚨 Attack Pattern: Port Scan");
  Serial.print("   Packets: ");
  Serial.print(packetCount);
  Serial.println(" (MODERATE)");
  Serial.print("   Scanning ports...");
}

void simulateBurstTraffic() {
  // Burst: Sudden spike in traffic
  packetCount = random(800, 2000);           // Very high burst
  bytesTransferred = random(80000, 200000);  // Very high bytes
  
  Serial.println("🚨 Attack Pattern: Burst Traffic");
  Serial.print("   Packets: ");
  Serial.print(packetCount);
  Serial.println(" (BURST)");
  Serial.print("   Bytes: ");
  Serial.print(bytesTransferred);
  Serial.println(" (BURST)");
}

void simulateRandomAttack() {
  // Random: Unpredictable patterns
  packetCount = random(200, 1000);
  bytesTransferred = random(20000, 100000);
  
  Serial.println("🚨 Attack Pattern: Random");
  Serial.print("   Packets: ");
  Serial.println(packetCount);
  Serial.print("   Bytes: ");
  Serial.println(bytesTransferred);
}

void sendDataToServer() {
  // Build server URL
  String url = "http://" + String(serverIP) + ":" + String(serverPort) + String(serverEndpoint);
  
  Serial.print("📡 Sending malicious data to: ");
  Serial.println(url);
  
  // Create JSON payload
  StaticJsonDocument<512> doc;
  doc["device_id"] = deviceID;
  doc["source_ip"] = WiFi.localIP().toString();
  doc["destination_ip"] = serverIP;
  doc["packet_count"] = packetCount;
  doc["bytes_transferred"] = bytesTransferred;
  doc["duration"] = (millis() - sessionStartTime) / 1000;
  doc["protocol"] = "tcp";
  
  // Vary ports based on attack type
  if (currentAttack == PORT_SCAN) {
    doc["source_port"] = random(1024, 65535);
    doc["destination_port"] = random(1, 1024);  // Scanning well-known ports
  } else {
    doc["source_port"] = random(49152, 65535);
    doc["destination_port"] = random(1, 65535);  // Random ports
  }
  
  String jsonPayload;
  serializeJson(doc, jsonPayload);
  
  // Send HTTP POST request
  http.begin(wifiClient, url);
  http.addHeader("Content-Type", "application/json");
  
  int httpResponseCode = http.POST(jsonPayload);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.print("✅ Server response (");
    Serial.print(httpResponseCode);
    Serial.print("): ");
    Serial.println(response);
  } else {
    Serial.print("❌ Error sending data: ");
    Serial.println(httpResponseCode);
  }
  
  http.end();
  Serial.println();
}
