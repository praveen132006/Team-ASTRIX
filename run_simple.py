#!/usr/bin/env python
"""
Simple run script for the AI Forensic Analysis System
"""

import os
import sys
import time
import webbrowser
from pathlib import Path

print("AI Forensic Analysis System - Simple Start Script")
print("=" * 50)

# Ensure required directories exist
for directory in ['uploads', 'debug_output', 'models']:
    Path(directory).mkdir(exist_ok=True)
    print(f"✓ Directory '{directory}' is ready")

# Ensure .env file exists
if not os.path.exists('.env'):
    with open('.env', 'w') as f:
        f.write('DEEPAI_API_KEY=your_api_key_here\n')
        f.write('DEBUG=True\n')
        f.write('PORT=5000\n')
    print("✓ Created .env file with default settings")
else:
    print("✓ .env file already exists")

print("\nStarting the application...")

# Open browser after a short delay
def open_browser():
    print("Opening browser...")
    time.sleep(2)
    webbrowser.open('http://localhost:5000')

import threading
browser_thread = threading.Thread(target=open_browser)
browser_thread.daemon = True
browser_thread.start()

# Start Flask application
try:
    from app import app
    print("✓ Application imported successfully")
    print("✓ Server starting at http://localhost:5000")
    
    # For development, enable debug mode
    app.run(debug=True, use_reloader=False)
except Exception as e:
    print(f"Error starting application: {e}")
    sys.exit(1) 