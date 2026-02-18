/*
 * ESP8266 Normal IoT Device
 * 
 * This device simulates a normal IoT device that sends legitimate
 * network traffic data to the backend server.
 * 
 * Hardware: ESP8266 (NodeMCU, Wemos D1 Mini, etc.)
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
const char* deviceID = "ESP8266_NORMAL";
const char* deviceIP = "192.168.1.201";        // This device's static IP

// Timing configuration
const unsigned long sendInterval = 5000;       // Send data every 5 seconds
unsigned long lastSendTime = 0;

// ============================================
// GLOBAL VARIABLES
// ============================================

WiFiClient wifiClient;
HTTPClient http;

// Traffic simulation variables
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
  Serial.println("ESP8266 Normal IoT Device");
  Serial.println("========================================");
  
  // Connect to WiFi
  connectToWiFi();
  
  // Set static IP (optional but recommended)
  // IPAddress local_IP(192, 168, 1, 201);
  // IPAddress gateway(192, 168, 1, 1);
  // IPAddress subnet(255, 255, 255, 0);
  // WiFi.config(local_IP, gateway, subnet);
  
  sessionStartTime = millis();
  
  Serial.println("✅ Setup complete!");
  Serial.println("📡 Ready to send data to server");
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
    
    // Simulate normal traffic
    simulateNormalTraffic();
    
    // Send data to server
    sendDataToServer();
  }
  
  delay(100);
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
    Serial.println("⚠️  Please check credentials and try again");
  }
}

void simulateNormalTraffic() {
  // Simulate normal IoT device behavior
  // - Low to moderate packet rates
  // - Regular intervals
  // - Standard ports (80, 443, 8080)
  
  packetCount = random(10, 50);              // 10-50 packets
  bytesTransferred = random(500, 5000);      // 500-5000 bytes
  
  Serial.println("📊 Simulating normal traffic:");
  Serial.print("   Packets: ");
  Serial.println(packetCount);
  Serial.print("   Bytes: ");
  Serial.println(bytesTransferred);
}

void sendDataToServer() {
  // Build server URL
  String url = "http://" + String(serverIP) + ":" + String(serverPort) + String(serverEndpoint);
  
  Serial.print("📡 Sending data to: ");
  Serial.println(url);
  
  // Create JSON payload
  StaticJsonDocument<512> doc;
  doc["device_id"] = deviceID;
  doc["source_ip"] = WiFi.localIP().toString();
  doc["destination_ip"] = serverIP;
  doc["packet_count"] = packetCount;
  doc["bytes_transferred"] = bytesTransferred;
  doc["duration"] = (millis() - sessionStartTime) / 1000; // seconds
  doc["protocol"] = "tcp";
  doc["source_port"] = random(49152, 65535);  // Ephemeral port range
  doc["destination_port"] = 80;                // HTTP
  
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
    Serial.println(http.errorToString(httpResponseCode));
  }
  
  http.end();
  Serial.println();
}
