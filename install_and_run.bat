@echo off
echo ForensicAI by Team ASTRIX - Installation and Setup
echo ================================================
echo.

echo Checking Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo Python is not installed or not in PATH. Please install Python 3.8 or higher.
    echo Visit https://www.python.org/downloads/ to download and install Python.
    pause
    exit /b 1
)

echo.
echo Creating required directories...
if not exist uploads mkdir uploads
if not exist debug_output mkdir debug_output
if not exist models mkdir models
echo Directories created successfully.

echo.
echo Installing required packages...
pip install opencv-python numpy pillow flask
if %ERRORLEVEL% NEQ 0 (
    echo There was an issue installing packages. Trying with --user flag...
    pip install --user opencv-python numpy pillow flask
)

echo.
echo Installation complete!
echo.
echo To run the application, you can:
echo 1. Use this script again
echo 2. Or run the command: python simple_app.py
echo.
echo Would you like to start the application now? (Y/N)
set /p choice=

if /i "%choice%"=="Y" (
    echo.
    echo Starting ForensicAI by Team ASTRIX...
    echo Access the application at http://localhost:5000
    echo Press Ctrl+C to stop the server when done.
    echo.
    python simple_app.py
) else (
    echo.
    echo To start the application later, run: python simple_app.py
    echo.
    pause
) 