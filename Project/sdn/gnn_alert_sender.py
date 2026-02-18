#!/usr/bin/env python3
"""
GNN Alert Sender Module
Sends alerts to Ryu controller when malicious devices are detected
"""

import requests
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SDNAlertSender:
    """
    Sends alerts to SDN controller when GNN detects malicious activity
    """
    
    def __init__(self, controller_url: str = "http://127.0.0.1:8080"):
        """
        Initialize alert sender
        
        Args:
            controller_url: URL of the Ryu controller REST API
        """
        self.controller_url = controller_url
        self.alert_endpoint = f"{controller_url}/alert"
        self.status_endpoint = f"{controller_url}/status"
        
    def send_alert(self, ip_address: str) -> bool:
        """
        Send alert to block a malicious IP address
        
        Args:
            ip_address: IP address to block
            
        Returns:
            True if alert was sent successfully, False otherwise
        """
        try:
            logger.warning(f"🚨 Sending alert to SDN controller: Block {ip_address}")
            
            response = requests.post(
                self.alert_endpoint,
                json={"ip": ip_address},
                timeout=5
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Alert sent successfully: {result.get('message')}")
                logger.info(f"📊 Currently blocked IPs: {result.get('blocked_ips')}")
                return True
            else:
                logger.error(f"❌ Failed to send alert: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to SDN controller. Is Ryu running?")
            return False
        except requests.exceptions.Timeout:
            logger.error("❌ Request to SDN controller timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error sending alert: {e}")
            return False
    
    def get_status(self) -> Optional[dict]:
        """
        Get current blocking status from controller
        
        Returns:
            Status dictionary or None if request fails
        """
        try:
            response = requests.get(self.status_endpoint, timeout=5)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get status: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            return None
    
    def check_connection(self) -> bool:
        """
        Check if SDN controller is reachable
        
        Returns:
            True if controller is reachable, False otherwise
        """
        try:
            response = requests.get(self.status_endpoint, timeout=2)
            return response.status_code == 200
        except:
            return False


# Convenience function for quick integration
def send_alert(ip_address: str, controller_url: str = "http://127.0.0.1:8080") -> bool:
    """
    Quick function to send an alert
    
    Args:
        ip_address: IP address to block
        controller_url: URL of the Ryu controller
        
    Returns:
        True if successful, False otherwise
    """
    sender = SDNAlertSender(controller_url)
    return sender.send_alert(ip_address)


if __name__ == "__main__":
    # Test the alert sender
    print("Testing SDN Alert Sender")
    print("=" * 60)
    
    sender = SDNAlertSender()
    
    # Check connection
    print("\n1. Checking connection to SDN controller...")
    if sender.check_connection():
        print("✅ Connected to SDN controller")
    else:
        print("❌ Cannot connect to SDN controller")
        print("   Make sure Ryu is running: ryu-manager ryu_blocker.py")
        exit(1)
    
    # Get current status
    print("\n2. Getting current status...")
    status = sender.get_status()
    if status:
        print(f"   Status: {status['status']}")
        print(f"   Blocked IPs: {status['blocked_ips']}")
        print(f"   Total blocked: {status['total_blocked']}")
    
    # Send test alert
    print("\n3. Sending test alert for 192.168.1.5...")
    success = sender.send_alert("192.168.1.5")
    
    if success:
        print("✅ Test alert sent successfully!")
    else:
        print("❌ Failed to send test alert")
    
    print("\n" + "=" * 60)
