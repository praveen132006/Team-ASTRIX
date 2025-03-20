#!/bin/bash

echo "ForensicAI by Team ASTRIX - Installation and Setup"
echo "================================================="
echo

echo "Checking Python installation..."
if command -v python3 &>/dev/null; then
    python3 --version
else
    echo "Python 3 is not installed or not in PATH. Please install Python 3.8 or higher."
    echo "Visit https://www.python.org/downloads/ to download and install Python."
    exit 1
fi

echo
echo "Creating required directories..."
mkdir -p uploads debug_output models
echo "Directories created successfully."

echo
echo "Installing required packages..."
python3 -m pip install opencv-python numpy pillow flask
if [ $? -ne 0 ]; then
    echo "There was an issue installing packages. Trying with --user flag..."
    python3 -m pip install --user opencv-python numpy pillow flask
fi

echo
echo "Installation complete!"
echo
echo "To run the application, you can:"
echo "1. Use this script again"
echo "2. Or run the command: python3 simple_app.py"
echo
read -p "Would you like to start the application now? (y/n) " choice

if [[ $choice =~ ^[Yy]$ ]]; then
    echo
    echo "Starting ForensicAI by Team ASTRIX..."
    echo "Access the application at http://localhost:5000"
    echo "Press Ctrl+C to stop the server when done."
    echo
    python3 simple_app.py
else
    echo
    echo "To start the application later, run: python3 simple_app.py"
    echo
fi 