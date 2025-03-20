#!/usr/bin/env python
"""
Run script for the AI Forensic Analysis System
This script helps users set up and run the application.
"""

import os
import sys
import argparse
import subprocess
import platform
import webbrowser
import time
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    required_version = (3, 8)
    current_version = sys.version_info
    
    if current_version < required_version:
        print(f"Error: Python {required_version[0]}.{required_version[1]} or higher is required.")
        print(f"Current Python version: {current_version[0]}.{current_version[1]}.{current_version[2]}")
        sys.exit(1)

def check_dependencies():
    """Check if required directories exist and create them if needed."""
    directories = ['uploads', 'debug_output', 'models']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✓ Directory '{directory}' is ready")

def create_env_file():
    """Create a .env file if it doesn't exist."""
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write('DEEPAI_API_KEY=your_api_key_here\n')
            f.write('DEBUG=False\n')
            f.write('PORT=5000\n')
        print("✓ Created .env file with default settings")
    else:
        print("✓ .env file already exists")

def setup_virtual_env(force_recreate=False):
    """Set up a virtual environment if it doesn't exist or force recreation."""
    if os.path.exists('venv') and force_recreate:
        print("Removing existing virtual environment...")
        try:
            shutil.rmtree('venv')
            print("✓ Removed old virtual environment")
        except Exception as e:
            print(f"Error removing virtual environment: {e}")
            sys.exit(1)
    
    if not os.path.exists('venv') or force_recreate:
        print("Setting up virtual environment...")
        try:
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            print("✓ Virtual environment created")
            
            # Upgrade pip in the new environment
            if platform.system() == 'Windows':
                pip_path = os.path.join('venv', 'Scripts', 'python')
            else:
                pip_path = os.path.join('venv', 'bin', 'python')
            
            subprocess.run([pip_path, '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
            print("✓ Pip upgraded in virtual environment")
        except subprocess.CalledProcessError:
            print("Error: Failed to create virtual environment.")
            sys.exit(1)
    else:
        print("✓ Virtual environment already exists")

def install_requirements():
    """Install required packages from requirements.txt."""
    print("Installing requirements...")
    
    # Determine the pip command based on the platform
    if platform.system() == 'Windows':
        pip_path = os.path.join('venv', 'Scripts', 'python')
        pip_args = ['-m', 'pip', 'install', '-r', 'requirements.txt']
    else:
        pip_path = os.path.join('venv', 'bin', 'python')
        pip_args = ['-m', 'pip', 'install', '-r', 'requirements.txt']
    
    try:
        subprocess.run([pip_path] + pip_args, check=True)
        print("✓ Requirements installed")
    except subprocess.CalledProcessError:
        print("Error: Failed to install requirements.")
        sys.exit(1)

def run_application(use_docker=False, debug=False):
    """Run the application using Docker or directly."""
    if use_docker:
        print("Starting the application with Docker...")
        try:
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            print("✓ Application started with Docker")
            print("✓ Open http://localhost:5000 in your browser")
            # Wait a bit for Docker to start up
            time.sleep(5)
            webbrowser.open('http://localhost:5000')
        except subprocess.CalledProcessError:
            print("Error: Failed to start Docker containers.")
            sys.exit(1)
        except FileNotFoundError:
            print("Error: Docker Compose not found. Please install Docker and Docker Compose.")
            sys.exit(1)
    else:
        print("Starting the application directly...")
        
        # Determine the python command based on the platform
        if platform.system() == 'Windows':
            python_path = os.path.join('venv', 'Scripts', 'python')
        else:
            python_path = os.path.join('venv', 'bin', 'python')
        
        env = os.environ.copy()
        if debug:
            env['DEBUG'] = 'True'
        
        try:
            # Open browser after a short delay
            def open_browser():
                time.sleep(2)
                webbrowser.open('http://localhost:5000')
            
            import threading
            browser_thread = threading.Thread(target=open_browser)
            browser_thread.daemon = True
            browser_thread.start()
            
            # Run the application
            subprocess.run([python_path, 'app.py'], env=env, check=True)
        except KeyboardInterrupt:
            print("\nApplication stopped")
        except subprocess.CalledProcessError:
            print("Error: Failed to start the application.")
            sys.exit(1)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run the AI Forensic Analysis System')
    parser.add_argument('--docker', action='store_true', help='Run with Docker')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--setup', action='store_true', help='Set up the environment without running')
    parser.add_argument('--force-recreate-venv', action='store_true', help='Force recreate the virtual environment')
    args = parser.parse_args()
    
    # Check Python version
    check_python_version()
    
    # Check and create required directories
    check_dependencies()
    
    # Create .env file if it doesn't exist
    create_env_file()
    
    if not args.docker:
        # Set up virtual environment
        setup_virtual_env(force_recreate=args.force_recreate_venv)
        
        # Install requirements
        install_requirements()
    
    if not args.setup:
        # Run the application
        run_application(use_docker=args.docker, debug=args.debug)

if __name__ == '__main__':
    main() 