from flask import Flask, request, jsonify, render_template, send_from_directory, make_response
from werkzeug.utils import secure_filename
import os
import json
import time
import logging
from datetime import datetime
from dotenv import load_dotenv
from modules.deepfake_detector import DeepfakeDetector
from modules.text_analyzer import TextAnalyzer
from modules.metadata_analyzer import MetadataAnalyzer
from modules.ela_analyzer import ELAAnalyzer
from modules.noise_analyzer import NoiseAnalyzer
from modules.cnn_classifier import CNNClassifier

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("forensic_ai.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG_OUTPUT'] = 'debug_output'
app.config['API_VERSION'] = 'v1'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp', 'tiff'}

# Initialize analyzers
deepfake_detector = DeepfakeDetector()
text_analyzer = TextAnalyzer()
metadata_analyzer = MetadataAnalyzer()
ela_analyzer = ELAAnalyzer()
noise_analyzer = NoiseAnalyzer()
cnn_classifier = CNNClassifier()

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEBUG_OUTPUT'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/robots.txt')
def robots():
    """Serve robots.txt file"""
    return send_from_directory(app.static_folder, 'robots.txt')

@app.route('/analyze/image', methods=['POST'])
def analyze_image_redirect():
    """Redirect to the versioned API endpoint for backward compatibility"""
    return analyze_image_v1()

@app.route(f"/api/{app.config['API_VERSION']}/analyze/image", methods=['POST'])
def analyze_image_v1():
    """Analyze an image using AI forensic techniques"""
    start_time = time.time()
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'File type not supported'}), 415
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Processing image: {filename}")
            
            # Analyze the image with comprehensive forensic detection
            deepfake_result = deepfake_detector.analyze_image(filepath)
            metadata_result = metadata_analyzer.analyze_file(filepath)
            
            # Store paths to visualizations for the frontend
            visualizations = deepfake_result.get('visualizations', {})
            
            # Generate a unique folder for debug files from this analysis
            debug_id = f"{os.path.splitext(filename)[0].replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            debug_folder = f"debug_{debug_id}"
            os.makedirs(os.path.join(app.config['DEBUG_OUTPUT'], debug_folder), exist_ok=True)
            
            # Move visualization files to the designated folder for easier access
            for viz_type, viz_path in visualizations.items():
                if viz_path and os.path.exists(viz_path):
                    new_path = os.path.join(app.config['DEBUG_OUTPUT'], debug_folder, f"{viz_type}.jpg")
                    os.rename(viz_path, new_path)
                    visualizations[viz_type] = f"/debug/{debug_folder}/{viz_type}.jpg"
            
            # Create URLs for accessing debug visualizations
            deepfake_result['visualizations'] = visualizations
            
            # Add overall classification result
            manipulation_probability = deepfake_result.get('manipulation_probability', 0)
            deepfake_result['classification'] = 'AI-generated' if manipulation_probability > 0.5 else 'Authentic'
            deepfake_result['risk_level'] = 'High' if manipulation_probability > 0.7 else 'Medium' if manipulation_probability > 0.4 else 'Low'
            
            # Clean up original upload
            os.remove(filepath)
            
            # Calculate processing time
            process_time = time.time() - start_time
            logger.info(f"Image analysis completed in {process_time:.2f} seconds")
            
            return jsonify({
                'deepfake_analysis': deepfake_result,
                'metadata_analysis': metadata_result,
                'process_time': f"{process_time:.2f}s"
            })
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500

@app.route('/analyze/text', methods=['POST'])
def analyze_text_redirect():
    """Redirect to the versioned API endpoint for backward compatibility"""
    return analyze_text_v1()

@app.route(f"/api/{app.config['API_VERSION']}/analyze/text", methods=['POST'])
def analyze_text_v1():
    """Analyze text for AI generation markers"""
    try:
        if not request.json or 'text' not in request.json:
            return jsonify({'error': 'No text provided'}), 400
        
        text = request.json['text']
        if not text or len(text.strip()) == 0:
            return jsonify({'error': 'Empty text provided'}), 400
        
        logger.info(f"Analyzing text of length: {len(text)}")
        result = text_analyzer.analyze_text(text)
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}", exc_info=True)
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500

@app.route('/analyze/video', methods=['POST'])
def analyze_video_redirect():
    """Redirect to the versioned API endpoint for backward compatibility"""
    return analyze_video_v1()

@app.route(f"/api/{app.config['API_VERSION']}/analyze/video", methods=['POST'])
def analyze_video_v1():
    """Analyze a video for deepfake markers"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            logger.info(f"Processing video: {filename}")
            
            # Analyze the video
            deepfake_result = deepfake_detector.analyze_video(filepath)
            metadata_result = metadata_analyzer.analyze_file(filepath)
            
            # Clean up
            os.remove(filepath)
            
            return jsonify({
                'deepfake_analysis': deepfake_result,
                'metadata_analysis': metadata_result
            })
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
        return jsonify({'error': f"Analysis failed: {str(e)}"}), 500

@app.route('/debug/<path:filename>')
def debug_files(filename):
    """Serve debug output files like visualizations."""
    return send_from_directory(app.config['DEBUG_OUTPUT'], filename)

@app.route('/api/documentation')
def api_docs():
    """Serve API documentation."""
    return render_template('api_docs.html')

@app.route('/about')
def about():
    """About page explaining the AI forensic system."""
    return render_template('about.html')

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors with a custom page"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors with a custom page"""
    return render_template('500.html'), 500

@app.after_request
def add_header(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    return response

if __name__ == '__main__':
    app.run(debug=os.environ.get('DEBUG', 'False').lower() == 'true', host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 