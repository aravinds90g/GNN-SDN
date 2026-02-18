# IoT Security System - Complete Reset Script
# This script clears all data and resets the system for a fresh start

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "IoT Security System - Reset Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 1. Clear CSV data file
Write-Host "Step 1: Clearing CSV data file..." -ForegroundColor Yellow
$csvPath = "backend\data\network_flows.csv"
if (Test-Path $csvPath) {
    # Backup old file
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $backupPath = "backend\data\network_flows_backup_$timestamp.csv"
    Move-Item $csvPath $backupPath -Force
    Write-Host "  Old data backed up to: $backupPath" -ForegroundColor Green
}

# Create new CSV with headers
$headers = "Timestamp,Device_ID,Source_IP,Destination_IP,Packet_Count,Bytes_Transferred,Duration,Protocol,Source_Port,Destination_Port,Label"
Set-Content -Path $csvPath -Value $headers
Write-Host "  New CSV file created with headers" -ForegroundColor Green

# 2. Clear backend blocked IPs (via API)
Write-Host ""
Write-Host "Step 2: Clearing blocked IPs..." -ForegroundColor Yellow
try {
    $response = Invoke-RestMethod -Uri "http://localhost:3000/api/data/clear" -Method Post -ErrorAction Stop
    Write-Host "  Backend data cleared" -ForegroundColor Green
} catch {
    Write-Host "  Backend not running or already clear" -ForegroundColor DarkYellow
}

# 3. Show current system status
Write-Host ""
Write-Host "Step 3: System Status" -ForegroundColor Yellow
try {
    $stats = Invoke-RestMethod -Uri "http://localhost:3000/api/stats" -ErrorAction Stop
    Write-Host "  Total flows: $($stats.totalFlows)"
    Write-Host "  Blocked IPs: $($stats.blockedIPs.Count)"
} catch {
    Write-Host "  Cannot get stats (backend may not be running)" -ForegroundColor DarkYellow
}

# 4. Instructions
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "RESET COMPLETE!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Fix ESP8266 Device ID:"
Write-Host "   Open: esp8266\malicious_device\malicious_device.ino"
Write-Host "   Line 44: const char* deviceID = ""ESP8266_MALICIOUS"";"
Write-Host ""
Write-Host "2. Upload to ESP8266:"
Write-Host "   - Open Arduino IDE"
Write-Host "   - Upload the corrected code"
Write-Host ""
Write-Host "3. Restart Backend Server (Terminal 1):"
Write-Host "   Ctrl+C, then: npm start"
Write-Host ""
Write-Host "4. Restart Detection Script (Terminal 3):"
Write-Host "   Ctrl+C, then: python real_time_detection.py --interval 30"
Write-Host ""
Write-Host "5. Wait for Detection:"
Write-Host "   - System will collect data for 30 seconds"
Write-Host "   - GNN will detect malicious device"
Write-Host "   - Backend will block the IP"
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
