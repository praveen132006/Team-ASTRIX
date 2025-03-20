# AI Forensic Analysis System by Team ASTRIX

A powerful tool for detecting AI-generated content using multiple forensic analysis techniques.

![ForensicAI Screenshot](static/images/screenshot.jpg)

## Overview

The AI Forensic Analysis System is designed to differentiate between authentic (human-created) media and AI-generated or manipulated content. It employs multiple forensic analysis techniques to detect indicators of AI generation, manipulation, or deepfakes in images.

With the rapid advancement of AI image generation technologies like DALL-E, Midjourney, and Stable Diffusion, the ability to detect AI-generated images has become increasingly important. This system provides comprehensive analysis and visualization tools to identify AI artifacts and manipulation markers.

## Features

### Image Analysis

- **Pattern & Frequency Analysis**: Reveals artifacts invisible to the human eye but detectable in the frequency domain, especially effective for DALL-E images.
- **Noise Analysis**: Identifies unnatural noise patterns characteristic of AI-generated images.
- **Facial Feature Analysis**: Detects inconsistencies in facial features, eyes, proportions, and textures that reveal AI-generated faces.
- **Pixel Quantization Detection**: Analyzes pixel value distributions to find quantization artifacts common in AI-generated images.
- **Aspect Ratio Analysis**: Examines image dimensions for standard ratios used by AI generators.
- **Symmetry Detection**: Identifies unnatural symmetry that appears in AI-generated content.

## Installation Instructions

### Prerequisites

- Python 3.8+ 
- pip (Python package manager)
- Git

### Option 1: Quick Setup (Recommended)

1. Clone the repository:
   ```
   git clone https://github.com/praveen132006/Team-ASTRIX.git
   cd Team-ASTRIX
   ```

2. Run the setup script which will create a virtual environment and install dependencies:
   
   **On Windows:**
   ```
   python run.py --setup
   ```
   
   **On macOS/Linux:**
   ```
   python3 run.py --setup
   ```

3. Run the application:
   
   **On Windows:**
   ```
   python run.py
   ```
   
   **On macOS/Linux:**
   ```
   python3 run.py
   ```

4. Access the web interface at: http://localhost:5000

### Option 2: Manual Setup

1. Clone the repository:
   ```
   git clone https://github.com/praveen132006/Team-ASTRIX.git
   cd Team-ASTRIX
   ```

2. Create a virtual environment:
   
   **On Windows:**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
   
   **On macOS/Linux:**
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python simple_app.py
   ```

5. Access the web interface at: http://localhost:5000

### Option 3: Running the Simple Version

If you encounter issues with dependencies or just want to test the application quickly:

1. After cloning the repository and activating the virtual environment:
   ```
   python simple_app.py
   ```

2. This will run the streamlined version of the application with core AI detection features.

## Usage

1. Open the application in your browser at http://localhost:5000
2. Upload an image you want to analyze (supports JPG, PNG, GIF, WEBP)
3. The system will analyze the image and provide:
   - AI generation probability score
   - Classification (AI-generated or authentic)
   - Detailed scores for various detection metrics
   - Visualizations of detected patterns
   - Analysis notes explaining the findings

## Project Information

Developed by Team ASTRIX for the Code Craft Chase Hackathon:

- Sanjeev Ram S - RA2411042010062
- Vishal Adhityaa S K - RA2411042010057
- Praveen M - RA2411042010026

## Repository

GitHub: [https://github.com/praveen132006/Team-ASTRIX](https://github.com/praveen132006/Team-ASTRIX)

## License

This project is licensed under the MIT License - see the LICENSE file for details. 