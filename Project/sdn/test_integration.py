#!/usr/bin/env python3
"""
End-to-End Test Script for GNN-SDN Integration
Tests the complete pipeline from detection to blocking
"""

import subprocess
import time
import requests
import sys
import os


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(text)
    print("="*60)


def check_process_running(process_name):
    """Check if a process is running"""
    try:
        result = subprocess.run(
            ['pgrep', '-f', process_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False


def test_controller_api():
    """Test Ryu controller REST API"""
    print_header("🧪 Testing SDN Controller API")
    
    try:
        # Test status endpoint
        print("\n1. Testing /status endpoint...")
        response = requests.get("http://127.0.0.1:8080/status", timeout=5)
        
        if response.status_code == 200:
            print("✅ Status endpoint working")
            data = response.json()
            print(f"   Status: {data['status']}")
            print(f"   Blocked IPs: {data['blocked_ips']}")
        else:
            print(f"❌ Status endpoint failed: {response.status_code}")
            return False
        
        # Test alert endpoint
        print("\n2. Testing /alert endpoint...")
        response = requests.post(
            "http://127.0.0.1:8080/alert",
            json={"ip": "192.168.1.100"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Alert endpoint working")
            data = response.json()
            print(f"   Message: {data['message']}")
        else:
            print(f"❌ Alert endpoint failed: {response.status_code}")
            return False
        
        # Test unblock endpoint
        print("\n3. Testing /unblock endpoint...")
        response = requests.post(
            "http://127.0.0.1:8080/unblock",
            json={"ip": "192.168.1.100"},
            timeout=5
        )
        
        if response.status_code == 200:
            print("✅ Unblock endpoint working")
        else:
            print(f"❌ Unblock endpoint failed: {response.status_code}")
            return False
        
        print("\n✅ All API tests passed!")
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to controller")
        print("   Make sure Ryu is running: ryu-manager sdn/ryu_blocker.py")
        return False
    except Exception as e:
        print(f"❌ Error testing API: {e}")
        return False


def test_alert_sender():
    """Test GNN alert sender module"""
    print_header("🧪 Testing GNN Alert Sender")
    
    try:
        from sdn.gnn_alert_sender import SDNAlertSender
        
        sender = SDNAlertSender()
        
        print("\n1. Checking controller connection...")
        if sender.check_connection():
            print("✅ Connected to controller")
        else:
            print("❌ Cannot connect to controller")
            return False
        
        print("\n2. Getting status...")
        status = sender.get_status()
        if status:
            print(f"✅ Status retrieved: {status}")
        else:
            print("❌ Failed to get status")
            return False
        
        print("\n3. Sending test alert...")
        success = sender.send_alert("192.168.1.200")
        if success:
            print("✅ Alert sent successfully")
        else:
            print("❌ Failed to send alert")
            return False
        
        print("\n4. Cleaning up (unblocking test IP)...")
        requests.post(
            "http://127.0.0.1:8080/unblock",
            json={"ip": "192.168.1.200"}
        )
        
        print("\n✅ Alert sender tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing alert sender: {e}")
        return False


def main():
    print("="*60)
    print("🧪 GNN-SDN INTEGRATION TEST SUITE")
    print("="*60)
    
    # Check prerequisites
    print_header("📋 Checking Prerequisites")
    
    print("\n1. Checking if Ryu controller is running...")
    try:
        response = requests.get("http://127.0.0.1:8080/status", timeout=2)
        if response.status_code == 200:
            print("✅ Ryu controller is running")
        else:
            print("⚠️  Ryu controller responded but with unexpected status")
    except:
        print("❌ Ryu controller is NOT running")
        print("\n   Start it with: ryu-manager sdn/ryu_blocker.py")
        print("   Then run this test again")
        return 1
    
    print("\n2. Checking Python dependencies...")
    try:
        import torch
        import requests
        print("✅ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return 1
    
    # Run tests
    all_passed = True
    
    if not test_controller_api():
        all_passed = False
    
    if not test_alert_sender():
        all_passed = False
    
    # Summary
    print_header("📊 TEST SUMMARY")
    
    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\n🎯 System is ready for full integration")
        print("\nNext steps:")
        print("1. Start Mininet: sudo python sdn/iot_topology.py")
        print("2. Run detection: python sdn/gnn_sdn_detection.py --data <your_data.csv>")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print("\nPlease fix the issues above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())
