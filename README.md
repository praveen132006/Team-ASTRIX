# AI Forensic Analysis System

A powerful tool for detecting AI-generated content using multiple forensic analysis techniques.

![ForensicAI Screenshot](static/images/screenshot.jpg)

## Overview

The AI Forensic Analysis System is designed to differentiate between authentic (human-created) media and AI-generated or manipulated content. It employs multiple forensic analysis techniques to detect indicators of AI generation, manipulation, or deepfakes in images, videos, and text.

With the rapid advancement of AI image generation technologies like DALL-E, Midjourney, and Stable Diffusion, the ability to detect AI-generated images has become increasingly important. This system provides comprehensive analysis and visualization tools to identify AI artifacts and manipulation markers.

## Features

### Image Analysis

- **Error Level Analysis (ELA)**: Detects inconsistent compression artifacts that typically appear in manipulated areas.
- **Noise Analysis**: Identifies unnatural noise patterns characteristic of AI-generated images.
- **Frequency Domain Analysis**: Reveals artifacts invisible to the human eye but detectable in the frequency domain.
- **Facial Analysis**: Detects inconsistencies in facial features, textures, and geometry that often appear in AI-generated faces.
- **Metadata Verification**: Examines image metadata for inconsistencies or missing information typical of AI-generated images.
- **CNN Classification**: Uses a deep learning model trained specifically to detect AI-generated imagery.

### Text Analysis

- **Pattern Matching**: Detects common linguistic patterns used by AI text generators.
- **Complexity Analysis**: Analyzes text complexity and structure.
- **Repetition Detection**: Identifies unnatural repetition patterns.
- **Filler Word Analysis**: Detects overuse of filler words common in AI-generated text.

## Core Components

1. **DeepfakeDetector**: Main module for analyzing images and videos.
2. **ELAAnalyzer**: Specialized module for Error Level Analysis.
3. **NoiseAnalyzer**: Analyzes noise patterns and distribution.
4. **CNNClassifier**: Deep learning-based AI image detection.
5. **MetadataAnalyzer**: Extracts and analyzes metadata from files.
6. **TextAnalyzer**: Specialized module for detecting AI-generated text.

## Technical Architecture

The system is built with a modular architecture, allowing for easy extension and improvement:

- **Backend**: Flask-based Python application handling analysis requests
- **Analysis Modules**: Specialized Python modules for different forensic techniques
- **Web Interface**: Modern Bootstrap-based UI for easy interaction
- **Visualization**: Interactive visualizations of analysis results

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/forensic-ai.git
   cd forensic-ai
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Access the web interface at: http://localhost:5000

## Usage

### Web Interface

1. Open the application in your browser.
2. Choose the type of content to analyze (image, video, or text).
3. Upload or paste your content.
4. Click "Analyze" and wait for the results.
5. Review the comprehensive analysis, including:
   - Manipulation probability score
   - Forensic markers detected
   - Visualizations of suspicious areas
   - Detailed recommendations

### API Usage

The system also provides a RESTful API for programmatic access:

```python
import requests

# Analyze an image
response = requests.post(
    "http://localhost:5000/analyze/image",
    files={"file": open("image.jpg", "rb")}
)

# Analyze text
response = requests.post(
    "http://localhost:5000/analyze/text",
    json={"text": "Text to analyze for AI generation"}
)

result = response.json()
print(f"AI Generation Probability: {result['manipulation_probability']}")
```

## Development

### Adding New Analysis Techniques

1. Create a new analyzer module in the `modules/` directory.
2. Implement the required analysis methods.
3. Integrate the new analyzer into the main `DeepfakeDetector` class.
4. Update the UI to display the new analysis results.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- OpenCV for computer vision functionality
- TensorFlow for deep learning capabilities
- Flask for the web framework
- Bootstrap for UI components
- Chart.js for visualization
- Various research papers on digital image forensics and deepfake detection 