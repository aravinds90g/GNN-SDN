#!/usr/bin/env python3
"""
Ryu/os-ken SDN Controller with GNN Integration
Automatically blocks malicious IoT devices based on GNN alerts

This version uses os-ken (Python 3.12+ compatible) with a separate REST API server
"""

# Import from os-ken (Python 3.12+ compatible fork of Ryu)
from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3
from os_ken.lib.packet import packet, ethernet, ipv4, arp
import logging
import threading
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse


# Global set of blocked IPs
BLOCKED_IPS = set()

# Global controller instance
controller_instance = None


class APIHandler(BaseHTTPRequestHandler):
    """Simple HTTP request handler for REST API"""
    
    def log_message(self, format, *args):
        """Suppress default logging"""
        pass
    
    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {
                'status': 'running',
                'blocked_ips': list(BLOCKED_IPS),
                'total_blocked': len(BLOCKED_IPS)
            }
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests"""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')
        
        try:
            data = json.loads(body) if body else {}
        except:
            data = {}
        
        if self.path == '/alert':
            ip = data.get('ip')
            if not ip:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'error', 'message': 'IP required'}).encode())
                return
            
            if controller_instance:
                controller_instance.logger.warning(f"🚨 GNN ALERT: Malicious device at {ip}")
                success = controller_instance.block_ip(ip)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    'status': 'success' if success else 'already_blocked',
                    'message': f'IP {ip} blocked' if success else f'IP {ip} already blocked',
                    'blocked_ips': list(BLOCKED_IPS)
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(500)
                self.end_headers()
        
        elif self.path == '/unblock':
            ip = data.get('ip')
            if controller_instance and ip:
                success = controller_instance.unblock_ip(ip)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    'status': 'success' if success else 'not_found',
                    'message': f'IP {ip} unblocked' if success else f'IP {ip} not blocked',
                    'blocked_ips': list(BLOCKED_IPS)
                }
                self.wfile.write(json.dumps(response).encode())
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()


class RyuBlocker(app_manager.OSKenApp):
    """
    Ryu SDN Controller for GNN-based IoT Attack Detection
    """
    
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]
    
    def __init__(self, *args, **kwargs):
        super(RyuBlocker, self).__init__(*args, **kwargs)
        
        # MAC to IP mapping
        self.mac_to_ip = {}
        self.ip_to_mac = {}
        
        # Store datapath for later use
        self.datapaths = {}
        
        # Store reference to self
        global controller_instance
        controller_instance = self
        
        self.logger.info("🚀 Ryu GNN Blocker Controller Started (os-ken)")
        self.logger.info("📡 Starting REST API server on http://127.0.0.1:8080")
        
        # Start HTTP server in separate thread
        self.api_thread = threading.Thread(target=self.start_api_server, daemon=True)
        self.api_thread.start()
    
    def start_api_server(self):
        """Start HTTP REST API server"""
        server = HTTPServer(('127.0.0.1', 8080), APIHandler)
        server.serve_forever()
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Handle switch connection and install table-miss flow entry
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        # Store datapath
        self.datapaths[datapath.id] = datapath
        
        self.logger.info(f"🔌 Switch connected: DPID {datapath.id}")
        
        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)
        
        self.logger.info(f"✅ Table-miss flow installed on switch {datapath.id}")
    
    def add_flow(self, datapath, priority, match, actions, buffer_id=None, idle_timeout=0):
        """
        Add a flow entry to the switch
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        
        if buffer_id:
            mod = parser.OFPFlowMod(datapath=datapath, buffer_id=buffer_id,
                                    priority=priority, match=match,
                                    instructions=inst, idle_timeout=idle_timeout)
        else:
            mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                    match=match, instructions=inst,
                                    idle_timeout=idle_timeout)
        
        datapath.send_msg(mod)
    
    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Handle incoming packets
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match['in_port']
        
        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocols(ethernet.ethernet)[0]
        
        dst = eth.dst
        src = eth.src
        
        dpid = datapath.id
        
        # Learn MAC addresses
        ip_pkt = pkt.get_protocol(ipv4.ipv4)
        if ip_pkt:
            src_ip = ip_pkt.src
            self.mac_to_ip[src] = src_ip
            self.ip_to_mac[src_ip] = src
        
        # Check if source IP is blocked
        if ip_pkt and ip_pkt.src in BLOCKED_IPS:
            self.logger.warning(f"🚫 Dropped packet from blocked IP: {ip_pkt.src}")
            return
        
        # Simple learning switch logic
        self.mac_to_port = getattr(self, 'mac_to_port', {})
        self.mac_to_port.setdefault(dpid, {})
        
        self.mac_to_port[dpid][src] = in_port
        
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD
        
        actions = [parser.OFPActionOutput(out_port)]
        
        # Install flow to avoid packet_in next time
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            
            # Verify if we have a valid buffer_id
            if msg.buffer_id != ofproto.OFP_NO_BUFFER:
                self.add_flow(datapath, 1, match, actions, msg.buffer_id)
                return
            else:
                self.add_flow(datapath, 1, match, actions)
        
        data = None
        if msg.buffer_id == ofproto.OFP_NO_BUFFER:
            data = msg.data
        
        out = parser.OFPPacketOut(datapath=datapath, buffer_id=msg.buffer_id,
                                  in_port=in_port, actions=actions, data=data)
        datapath.send_msg(out)
    
    def block_ip(self, ip_address):
        """
        Block an IP address by installing drop rules on all switches
        
        Args:
            ip_address: IP address to block
        """
        if ip_address in BLOCKED_IPS:
            self.logger.warning(f"⚠️  IP {ip_address} is already blocked")
            return False
        
        BLOCKED_IPS.add(ip_address)
        
        # Install drop rules on all connected switches
        for dpid, datapath in self.datapaths.items():
            ofproto = datapath.ofproto
            parser = datapath.ofproto_parser
            
            # Block packets FROM this IP
            match = parser.OFPMatch(eth_type=0x0800, ipv4_src=ip_address)
            actions = []  # Empty actions = drop
            self.add_flow(datapath, 100, match, actions)
            
            # Block packets TO this IP
            match = parser.OFPMatch(eth_type=0x0800, ipv4_dst=ip_address)
            actions = []  # Empty actions = drop
            self.add_flow(datapath, 100, match, actions)
            
            self.logger.info(f"🚫 Blocked IP {ip_address} on switch {dpid}")
        
        return True
    
    def unblock_ip(self, ip_address):
        """
        Unblock an IP address (for testing purposes)
        
        Args:
            ip_address: IP address to unblock
        """
        if ip_address in BLOCKED_IPS:
            BLOCKED_IPS.remove(ip_address)
            self.logger.info(f"✅ Unblocked IP {ip_address}")
            # Note: Flow entries remain until they timeout or switch is restarted
            return True
        return False
