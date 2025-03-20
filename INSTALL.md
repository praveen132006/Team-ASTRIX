# Installation Guide for ForensicAI by Team ASTRIX

This guide provides step-by-step instructions to install and run the ForensicAI application.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git

## Quick Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/praveen132006/Team-ASTRIX.git
cd Team-ASTRIX
```

### Step 2: Install Required Packages

```bash
pip install opencv-python numpy pillow flask
```

### Step 3: Run the Application

```bash
python simple_app.py
```

### Step 4: Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## Troubleshooting

If you encounter any issues with packages:

1. Make sure you have the latest version of pip:
```bash
pip install --upgrade pip
```

2. Install packages one by one if there are dependency issues:
```bash
pip install flask
pip install opencv-python
pip install numpy
pip install pillow
```

3. If you still encounter issues, try running the simple version with minimal dependencies:
```bash
python simple_app.py
```

## Full Installation with Virtual Environment

For a more isolated setup:

### Step 1: Clone the Repository

```bash
git clone https://github.com/praveen132006/Team-ASTRIX.git
cd Team-ASTRIX
```

### Step 2: Create a Virtual Environment

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
python simple_app.py
```

### Step 5: Access the Application

Open your web browser and navigate to:
```
http://localhost:5000
```

## Docker Installation (Optional)

If you have Docker installed:

```bash
docker-compose up -d
```

Then access the application at http://localhost:5000 