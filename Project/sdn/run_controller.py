#!/usr/bin/env python3
"""
Simple launcher for the os-ken SDN controller
"""

import sys
import os

# Get the absolute path to the controller file
controller_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ryu_blocker.py')

if __name__ == '__main__':
    # Import os-ken
    from os_ken import cfg
    from os_ken.base.app_manager import AppManager
    from os_ken.lib import hub
    
    # Configure
    cfg.CONF(project='os-ken', args=sys.argv[1:])
    
    print("🚀 SDN Controller Starting...")
    print("📡 REST API will be available at http://127.0.0.1:8080")
    print("Press Ctrl+C to stop")
    print("")
    
    # Create and run app manager
    app_mgr = AppManager.get_instance()
    app_mgr.load_apps([controller_path])
    contexts = app_mgr.create_contexts()
    services = app_mgr.instantiate_apps(**contexts)
    
    # Get all threads from the app manager
    threads = []
    for app in app_mgr.applications.values():
        threads.append(app.threads[0] if hasattr(app, 'threads') and app.threads else None)
    
    # Filter out None values
    threads = [t for t in threads if t is not None]
    
    try:
        # Keep the main thread alive
        if threads:
            hub.joinall(threads)
        else:
            # If no threads, just sleep forever
            while True:
                hub.sleep(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down controller...")
        app_mgr.close()
