#!/usr/bin/env python3
"""
IoT Network Topology for Mininet
Simulates an IoT network with normal and potentially malicious devices
"""

from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.cli import CLI
from mininet.log import setLogLevel, info


def create_iot_topology():
    """
    Create IoT network topology with:
    - 1 OpenFlow switch
    - 3 IoT devices (hosts)
    - Remote controller connection to Ryu
    """
    
    info('*** Creating IoT Network Topology\n')
    
    # Initialize Mininet with remote controller
    net = Mininet(
        controller=RemoteController,
        switch=OVSSwitch,
        autoSetMacs=True
    )
    
    info('*** Adding Remote Controller (Ryu)\n')
    c0 = net.addController(
        'c0',
        controller=RemoteController,
        ip='127.0.0.1',
        port=6633
    )
    
    info('*** Adding OpenFlow Switch\n')
    s1 = net.addSwitch('s1', protocols='OpenFlow13')
    
    info('*** Adding IoT Devices (Hosts)\n')
    # Normal IoT devices
    h1 = net.addHost('h1', ip='192.168.1.2/24', mac='00:00:00:00:00:02')
    h2 = net.addHost('h2', ip='192.168.1.3/24', mac='00:00:00:00:00:03')
    
    # Potentially malicious IoT device
    h3 = net.addHost('h3', ip='192.168.1.5/24', mac='00:00:00:00:00:05')
    
    # Additional normal device
    h4 = net.addHost('h4', ip='192.168.1.6/24', mac='00:00:00:00:00:06')
    
    info('*** Creating Links\n')
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(h3, s1)
    net.addLink(h4, s1)
    
    info('*** Starting Network\n')
    net.start()
    
    info('*** Network Configuration:\n')
    info('  Switch: s1 (OpenFlow 1.3)\n')
    info('  Controller: c0 @ 127.0.0.1:6633\n')
    info('  Hosts:\n')
    info('    h1: 192.168.1.2 (Normal IoT Device)\n')
    info('    h2: 192.168.1.3 (Normal IoT Device)\n')
    info('    h3: 192.168.1.5 (Potentially Malicious)\n')
    info('    h4: 192.168.1.6 (Normal IoT Device)\n')
    info('\n*** Testing connectivity with pingall\n')
    net.pingAll()
    
    info('\n*** Network is ready!\n')
    info('*** You can now test blocking with GNN detection\n')
    info('*** Example commands in CLI:\n')
    info('    h3 ping h1  (test if h3 can reach h1)\n')
    info('    h1 ping h2  (test normal communication)\n')
    info('\n*** Starting Mininet CLI\n')
    CLI(net)
    
    info('*** Stopping Network\n')
    net.stop()


if __name__ == '__main__':
    setLogLevel('info')
    create_iot_topology()
