# ESP8266 Setup Instructions

## Required Libraries

Install these libraries in Arduino IDE:

1. **ESP8266 Board Support**
   - Open Arduino IDE
   - Go to **File → Preferences**
   - Add to "Additional Board Manager URLs":
     ```
     http://arduino.esp8266.com/stable/package_esp8266com_index.json
     ```
   - Go to **Tools → Board → Boards Manager**
   - Search for "esp8266"
   - Install "esp8266 by ESP8266 Community"

2. **ArduinoJson Library**
   - Go to **Sketch → Include Library → Manage Libraries**
   - Search for "ArduinoJson"
   - Install "ArduinoJson by Benoit Blanchon" (version 6.x)

## Board Configuration

- **Board**: NodeMCU 1.0 (ESP-12E Module)
- **Upload Speed**: 115200
- **CPU Frequency**: 80 MHz
- **Flash Size**: 4MB (FS:2MB OTA:~1019KB)

## Wiring

No additional wiring needed! Just connect ESP8266 to computer via USB.

## Uploading

1. Connect ESP8266 to computer
2. Select correct port in **Tools → Port**
3. Click Upload button
4. Wait for "Done uploading" message

## Troubleshooting

### Upload Failed
- Press and hold FLASH button while clicking upload
- Try different USB cable
- Reduce upload speed to 57600

### WiFi Connection Failed
- Check SSID and password
- Ensure 2.4GHz WiFi (not 5GHz)
- Move closer to router

### Server Connection Failed
- Verify laptop IP address
- Check firewall settings
- Ensure backend server is running
