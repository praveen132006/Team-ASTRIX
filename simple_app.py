from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from werkzeug.utils import secure_filename
import os
import time
import json
import random
import shutil
import uuid
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import base64
from flask import Response

# Helper function to convert NumPy types to Python native types for JSON serialization
def convert_to_json_serializable(obj):
    """Converts NumPy types to standard Python types for JSON serialization."""
    
    # Handle NumPy arrays
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle NumPy scalars
    # Note: np.float_ was removed in NumPy 2.0, using specific types instead
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    
    # Handle dictionaries containing NumPy types
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    
    # Handle lists containing NumPy types
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    
    # Return other types as is
    return obj

# Create required directories
for directory in ['uploads', 'debug_output', 'models']:
    Path(directory).mkdir(exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max upload size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DEBUG_OUTPUT'] = 'debug_output'

# Custom JSON encoder for NumPy types
class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        return convert_to_json_serializable(obj)

# Configure Flask to use the custom JSON encoder
app.json_encoder = NumpyJSONEncoder

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DEBUG_OUTPUT'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except Exception:
        return "About page - coming soon"

@app.route('/api/documentation')
def api_docs():
    """Serve API documentation."""
    try:
        return render_template('api_docs.html')
    except Exception:
        return "API Documentation - coming soon"

@app.route('/debug/<path:filename>')
def debug_files(filename):
    """Serve debug output files like visualizations."""
    try:
        return send_from_directory(app.config['DEBUG_OUTPUT'], filename)
    except Exception as e:
        print(f"Error serving debug file: {e}")
        return jsonify({"error": "File not found"}), 404

def analyze_image_features(image_path):
    """Perform basic image analysis to detect AI generation patterns."""
    try:
        print(f"Starting analysis of image: {image_path}")
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
            return {
                'manipulation_probability': 0.5,
                'error': f'Image file not found: {image_path}',
                'width': 1024,
                'height': 768
            }
        
        # Check file size
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            print(f"Error: Image file is empty: {image_path}")
            return {
                'manipulation_probability': 0.5,
                'error': 'Image file is empty',
                'width': 1024,
                'height': 768
            }
        print(f"File size: {file_size} bytes")
        
        # Try to open with PIL first to check if it's a valid image
        try:
            pil_image = Image.open(image_path)
            pil_image.verify()  # Verify it's a valid image
            print(f"Valid image format: {pil_image.format}, Size: {pil_image.size}")
        except Exception as pil_error:
            print(f"PIL Error: {pil_error}")
            return {
                'manipulation_probability': 0.5,
                'error': f'Invalid image format: {str(pil_error)}',
                'width': 1024,
                'height': 768
            }
        
        # Load the image with OpenCV
        img = cv2.imread(image_path)
        if img is None:
            print(f"OpenCV Error: Failed to load image {image_path}")
            return {
                'manipulation_probability': 0.5,
                'error': 'Failed to load image with OpenCV',
                'width': 1024,
                'height': 768
            }
        
        # Get image dimensions
        height, width = img.shape[:2]
        print(f"Image dimensions: {width}x{height}")
        
        # Convert to RGB for better analysis
        try:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as color_error:
            print(f"Color conversion error: {color_error}")
            # Continue with the original image
            img_rgb = img
        
        try:
            # 1. Noise pattern analysis (AI-generated images often have less random noise)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            noise_map = cv2.Laplacian(gray, cv2.CV_64F)
            noise_std = float(np.std(noise_map))  # Convert to Python float
            
            # MODIFIED: Lowered threshold to be more sensitive to noise patterns in AI images
            # AI images typically have less natural noise
            noise_score = float(min(1.0, max(0.0, noise_std / 6.0)))  # Reduced from 8.0 to 6.0
            print(f"Noise analysis complete: {noise_std:.2f}, score: {noise_score:.3f}")
        except Exception as noise_error:
            print(f"Noise analysis error: {noise_error}")
            noise_score = 0.5
            noise_map = np.zeros_like(img[:,:,0], dtype=np.float64)
        
        try:
            # 2. Check for unnatural symmetry in the image
            height, width = gray.shape
            mid_point = width // 2
            left_half = gray[:, :mid_point]
            right_half = cv2.flip(gray[:, mid_point:], 1)
            
            # Handle cases where image dimensions are odd
            if left_half.shape[1] != right_half.shape[1]:
                right_half = right_half[:, :left_half.shape[1]]
                
            symmetry_diff = np.abs(left_half.astype(np.float32) - right_half.astype(np.float32))
            # MODIFIED: Lowered threshold to detect subtle symmetry patterns in AI images
            symmetry_score = float(min(1.0, np.mean(symmetry_diff) / 20.0))  # Reduced from 25.0 to 20.0
            print(f"Symmetry analysis complete, score: {symmetry_score:.3f}")
        except Exception as symmetry_error:
            print(f"Symmetry analysis error: {symmetry_error}")
            symmetry_score = 0.5
        
        try:
            # 3. Color distribution (AI images often have smoother color gradients)
            r_std = float(np.std(img_rgb[:,:,0]))
            g_std = float(np.std(img_rgb[:,:,1]))
            b_std = float(np.std(img_rgb[:,:,2]))
            color_std = (r_std + g_std + b_std) / 3.0
            
            # MODIFIED: Lowered threshold for color distribution to detect smooth gradients in AI images
            color_score = float(min(1.0, max(0.0, color_std / 60.0)))  # Reduced from 65.0 to 60.0
            print(f"Color analysis complete, std: {color_std:.2f}, score: {color_score:.3f}")
        except Exception as color_error:
            print(f"Color analysis error: {color_error}")
            color_score = 0.5
        
        try:
            # 4. Edge consistency check (AI images may have too perfect edges)
            edges = cv2.Canny(img, 100, 200)
            # Adjust edge threshold
            edge_density = float(np.mean(edges > 0))  # Convert to Python float
            # MODIFIED: Adjusted edge score calculation to be more sensitive
            edge_score = float(min(1.0, max(0.0, edge_density * 3.5)))  # Reduced from 4.0 to 3.5
            print(f"Edge analysis complete, density: {edge_density:.3f}, score: {edge_score:.3f}")
        except Exception as edge_error:
            print(f"Edge analysis error: {edge_error}")
            edge_score = 0.5
            edges = np.zeros_like(gray)
        
        try:
            # 5. Check for unnatural pixel distribution (histogram analysis)
            hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
            hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
            hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
            
            # Avoid division by zero
            r_mean = float(np.mean(hist_r) or 1.0)  # Convert to Python float
            g_mean = float(np.mean(hist_g) or 1.0)  # Convert to Python float
            b_mean = float(np.mean(hist_b) or 1.0)  # Convert to Python float
            
            r_std = float(np.std(hist_r))  # Convert to Python float
            g_std = float(np.std(hist_g))  # Convert to Python float
            b_std = float(np.std(hist_b))  # Convert to Python float
            
            hist_smoothness = (r_std / r_mean + g_std / g_mean + b_std / b_mean) / 3.0
            # MODIFIED: Lowered threshold to better detect smooth histograms in AI images
            hist_score = float(min(1.0, max(0.0, hist_smoothness / 3.5)))  # Reduced from 4.0 to 3.5
            print(f"Histogram analysis complete, smoothness: {hist_smoothness:.3f}, score: {hist_score:.3f}")
        except Exception as hist_error:
            print(f"Histogram analysis error: {hist_error}")
            hist_score = 0.5
            hist_r = hist_g = hist_b = np.ones((256, 1))
        
        # ADDED: Check for perfect aspect ratios common in AI-generated images
        try:
            # DALL-E and other AI models often use specific aspect ratios
            aspect_ratio = width / height
            common_ratios = [1.0, 4/3, 16/9, 3/2]  # Common AI generation ratios
            
            # Check how close the image is to common AI aspect ratios
            ratio_diffs = [abs(aspect_ratio - r) for r in common_ratios]
            min_ratio_diff = min(ratio_diffs)
            
            # If very close to a common ratio, this might be AI-generated
            aspect_ratio_score = float(min(1.0, max(0.0, min_ratio_diff * 5.0)))
            print(f"Aspect ratio analysis: {aspect_ratio:.3f}, score: {aspect_ratio_score:.3f}")
        except Exception as ratio_error:
            print(f"Aspect ratio analysis error: {ratio_error}")
            aspect_ratio_score = 0.5
        
        # NEW: Test for regular pattern detection (DALL-E images often have grid-like patterns)
        try:
            # Resize image to standard size for pattern analysis
            resized = cv2.resize(gray, (256, 256))
            
            # Apply Fourier Transform to detect grid patterns in the frequency domain
            f_transform = np.fft.fft2(resized)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Analyze central portion excluding DC component (very low frequencies)
            # DALL-E images often have strong regular patterns visible in the Fourier domain
            center_region = magnitude_spectrum[108:148, 108:148]  # Center 40x40 region
            peak_val = float(np.max(center_region))
            mean_val = float(np.mean(center_region))
            
            # High peak-to-mean ratio suggests strong periodic patterns
            pattern_ratio = peak_val / (mean_val if mean_val > 0 else 1.0)
            
            # Scale to a score (lower means more likely AI-generated)
            pattern_score = float(min(1.0, max(0.0, 3.0 / pattern_ratio)))
            print(f"Pattern analysis complete, ratio: {pattern_ratio:.3f}, score: {pattern_score:.3f}")
            
            # Also check for pixel grid artifacts (common in DALL-E images)
            # Detect grid-like patterns by looking at every 8th pixel correlation
            h_diff = np.mean(np.abs(np.diff(resized[:, ::8], axis=1)))
            v_diff = np.mean(np.abs(np.diff(resized[::8, :], axis=0)))
            
            grid_ratio = (h_diff + v_diff) / 2.0
            grid_score = float(min(1.0, max(0.0, grid_ratio / 15.0)))
            
            # Combine both pattern detection methods
            combined_pattern_score = float((pattern_score + grid_score) / 2.0)
            print(f"Grid analysis, score: {grid_score:.3f}, combined: {combined_pattern_score:.3f}")
            
            # Create pattern visualization
            pattern_vis = np.log(np.abs(f_shift) + 1)
            pattern_vis = cv2.normalize(pattern_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            pattern_vis = cv2.applyColorMap(pattern_vis, cv2.COLORMAP_JET)
            
            pattern_vis_path = os.path.join(os.path.dirname(image_path), 'pattern.jpg')
            cv2.imwrite(pattern_vis_path, pattern_vis)
            
        except Exception as pattern_error:
            print(f"Pattern analysis error: {pattern_error}")
            combined_pattern_score = 0.5
            pattern_vis_path = None
        
        # NEW: Check pixel value distribution for quantization artifacts
        try:
            # DALL-E images often have specific quantization patterns
            pixel_values = gray.flatten()
            hist_pixel, _ = np.histogram(pixel_values, bins=256, range=(0, 256))
            
            # Check for unnatural peaks in the histogram (quantization artifacts)
            peaks = []
            for i in range(1, 255):
                if hist_pixel[i] > hist_pixel[i-1] and hist_pixel[i] > hist_pixel[i+1]:
                    peaks.append((i, hist_pixel[i]))
            
            # Sort peaks by height
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            # Calculate ratio of highest peaks to average
            peak_avg_ratio = 0
            if len(peaks) > 5:
                top_peaks = sum(p[1] for p in peaks[:5])
                avg_value = np.mean(hist_pixel)
                if avg_value > 0:
                    peak_avg_ratio = top_peaks / (5 * avg_value)
            
            # AI images often have more pronounced peaks at specific values
            quantization_score = float(min(1.0, max(0.0, 1.0 / peak_avg_ratio if peak_avg_ratio > 1.0 else 1.0)))
            print(f"Quantization analysis, peaks: {len(peaks)}, score: {quantization_score:.3f}")
        except Exception as quant_error:
            print(f"Quantization analysis error: {quant_error}")
            quantization_score = 0.5
        
        # Calculate an overall manipulation probability with adjusted weights
        # Lower values in these metrics often indicate AI-generated content
        ai_indicators = [
            (1.0 - noise_score) * 1.8,         # Noise analysis
            (1.0 - symmetry_score) * 1.3,      # Symmetry analysis
            (1.0 - color_score) * 1.5,         # Color distribution
            (1.0 - edge_score) * 1.2,          # Edge patterns
            (1.0 - hist_score) * 1.4,          # Histogram analysis
            (1.0 - aspect_ratio_score) * 1.2,  # Aspect ratio analysis
            (1.0 - combined_pattern_score) * 2.0,  # NEW: Pattern detection (higher weight)
            (1.0 - quantization_score) * 1.6   # NEW: Quantization analysis
        ]
        
        # Normalize to get probability between 0 and 1
        weights_sum = 1.8 + 1.3 + 1.5 + 1.2 + 1.4 + 1.2 + 2.0 + 1.6  # Updated with new weights
        manipulation_probability = float(sum(ai_indicators) / weights_sum)  # Convert to Python float
        
        # INCREASED bias toward AI detection for more sensitive results
        # This helps with detecting more subtle AI generations, especially from DALL-E
        manipulation_probability = float(min(1.0, manipulation_probability * 1.25))  # Increased to 1.25
        
        # Print some debug info
        print(f"Analysis scores for {os.path.basename(image_path)}:")
        print(f"  Noise score: {noise_score:.3f} (lower in AI images)")
        print(f"  Symmetry score: {symmetry_score:.3f} (lower in AI images)")
        print(f"  Color score: {color_score:.3f} (lower in AI images)")
        print(f"  Edge score: {edge_score:.3f} (lower in AI images)")
        print(f"  Histogram score: {hist_score:.3f} (lower in AI images)")
        print(f"  Aspect ratio score: {aspect_ratio_score:.3f} (lower in AI images)")
        print(f"  Pattern score: {combined_pattern_score:.3f} (lower in AI images)")
        print(f"  Quantization score: {quantization_score:.3f} (lower in AI images)")
        print(f"  Final manipulation probability: {manipulation_probability:.3f}")
        
        # Create visualization images
        debug_path = os.path.dirname(image_path)
        visualizations = {}
        
        try:
            # Create noise visualization
            noise_vis = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            noise_vis = cv2.applyColorMap(noise_vis, cv2.COLORMAP_JET)
            noise_vis_path = os.path.join(debug_path, 'noise.jpg')
            cv2.imwrite(noise_vis_path, noise_vis)
            visualizations['ela'] = noise_vis_path
            print(f"Created noise visualization: {noise_vis_path}")
        except Exception as viz_error:
            print(f"Error creating noise visualization: {viz_error}")
        
        try:
            # Create edge visualization
            edge_vis_path = os.path.join(debug_path, 'frequency.jpg')
            cv2.imwrite(edge_vis_path, edges)
            visualizations['noise'] = edge_vis_path
            print(f"Created edge visualization: {edge_vis_path}")
        except Exception as viz_error:
            print(f"Error creating edge visualization: {viz_error}")
        
        try:
            # Create symmetry visualization
            symmetry_vis = np.zeros_like(img)
            half_width = width // 2
            symmetry_vis[:, :half_width] = img[:, :half_width]
            flipped = cv2.flip(img[:, half_width:], 1)
            
            # Handle cases where dimensions don't match exactly
            if symmetry_vis[:, half_width:].shape[1] > flipped.shape[1]:
                symmetry_vis[:, half_width:half_width+flipped.shape[1]] = flipped
            elif symmetry_vis[:, half_width:].shape[1] < flipped.shape[1]:
                symmetry_vis[:, half_width:] = flipped[:, :symmetry_vis[:, half_width:].shape[1]]
            else:
                symmetry_vis[:, half_width:] = flipped
                
            symmetry_vis_path = os.path.join(debug_path, 'ela.jpg')
            cv2.imwrite(symmetry_vis_path, symmetry_vis)
            visualizations['frequency'] = symmetry_vis_path
            print(f"Created symmetry visualization: {symmetry_vis_path}")
        except Exception as viz_error:
            print(f"Error creating symmetry visualization: {viz_error}")
        
        try:
            # Add pattern visualization if available
            if pattern_vis_path and os.path.exists(pattern_vis_path):
                visualizations['pattern'] = pattern_vis_path
                print(f"Added pattern visualization: {pattern_vis_path}")
        except Exception as viz_error:
            print(f"Error adding pattern visualization: {viz_error}")
        
        try:
            # Create face detection placeholder
            face_vis_path = os.path.join(debug_path, 'face_detection.jpg')
            cv2.imwrite(face_vis_path, img)
            visualizations['face_detection'] = face_vis_path
            print(f"Created face detection visualization: {face_vis_path}")
        except Exception as viz_error:
            print(f"Error creating face detection visualization: {viz_error}")
        
        try:
            # Histogram visualization
            hist_img = np.zeros((256, 256, 3), dtype=np.uint8)
            hist_img.fill(255)  # White background
            
            # Normalize histograms - convert to float64 to avoid overflow issues
            hist_r_norm = cv2.normalize(hist_r.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX)
            hist_g_norm = cv2.normalize(hist_g.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX)
            hist_b_norm = cv2.normalize(hist_b.astype(np.float64), None, 0, 255, cv2.NORM_MINMAX)
            
            # Draw histogram lines
            for i in range(1, 256):
                try:
                    r_val = int(hist_r_norm[i])
                    g_val = int(hist_g_norm[i])
                    b_val = int(hist_b_norm[i])
                    
                    r_prev = int(hist_r_norm[i-1])
                    g_prev = int(hist_g_norm[i-1])
                    b_prev = int(hist_b_norm[i-1])
                    
                    # Draw lines (point-to-point)
                    cv2.line(hist_img, (i-1, 255 - r_prev), (i, 255 - r_val), (0, 0, 255), 1)
                    cv2.line(hist_img, (i-1, 255 - g_prev), (i, 255 - g_val), (0, 255, 0), 1)
                    cv2.line(hist_img, (i-1, 255 - b_prev), (i, 255 - b_val), (255, 0, 0), 1)
                except Exception as line_error:
                    print(f"Error drawing histogram line at index {i}: {line_error}")
            
            hist_vis_path = os.path.join(debug_path, 'histogram.jpg')
            cv2.imwrite(hist_vis_path, hist_img)
            visualizations['histogram'] = hist_vis_path
            print(f"Created histogram visualization: {hist_vis_path}")
        except Exception as viz_error:
            print(f"Error creating histogram visualization: {viz_error}")
        
        # Ensure all values are Python native types, not NumPy types
        result = {
            'manipulation_probability': float(manipulation_probability),
            'noise_score': float(noise_score),
            'symmetry_score': float(symmetry_score),
            'color_score': float(color_score),
            'edge_score': float(edge_score),
            'hist_score': float(hist_score),
            'aspect_ratio_score': float(aspect_ratio_score),
            'pattern_score': float(combined_pattern_score),  # New score
            'quantization_score': float(quantization_score), # New score
            'width': int(width),
            'height': int(height),
            'visualizations': visualizations
        }
        
        return result
    except Exception as e:
        print(f"Critical error in image analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'manipulation_probability': 0.5,
            'error': f"Analysis failed: {str(e)}",
            'width': 1024,
            'height': 768,
            'visualizations': {}
        }

@app.route('/analyze/image', methods=['POST'])
def analyze_image():
    """Handle image upload and analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        try:
            # Create a unique directory name with a shorter name to avoid path length issues
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            unique_id = str(uuid.uuid4())[:8]  # Use just first 8 chars of UUID
            debug_dir = f"debug_{unique_id}_{timestamp}"
            debug_path = os.path.join(app.config['DEBUG_OUTPUT'], debug_dir)
            os.makedirs(debug_path, exist_ok=True)
            
            # Generate a shorter filename while preserving extension
            original_filename = secure_filename(file.filename)
            filename_parts = os.path.splitext(original_filename)
            short_name = f"{filename_parts[0][:20]}{filename_parts[1]}"  # Limit base name to 20 chars
            filename = short_name
            
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Copy file to debug directory instead of moving it
            debug_filepath = os.path.join(debug_path, filename)
            shutil.copy2(filepath, debug_filepath)
            
            # Analyze the image
            print(f"Starting analysis for file: {original_filename}")
            try:
                analysis_result = analyze_image_features(debug_filepath)
            except Exception as analysis_error:
                print(f"Error during image analysis: {str(analysis_error)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'Error during image analysis: {str(analysis_error)}',
                    'traceback': traceback.format_exc()
                }), 500
            
            # Convert NumPy types to Python native types for JSON serialization
            try:
                analysis_result = convert_to_json_serializable(analysis_result)
            except Exception as conversion_error:
                print(f"Error converting analysis results to JSON: {str(conversion_error)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'Error processing analysis results: {str(conversion_error)}',
                    'traceback': traceback.format_exc()
                }), 500
            
            # Determine if the image is likely AI-generated
            manipulation_probability = analysis_result.get('manipulation_probability', 0)
            print(f"Analysis completed with manipulation probability: {manipulation_probability}")
            
            # Add classification
            is_ai_generated = manipulation_probability > 0.5
            classification = "AI-generated" if is_ai_generated else "Authentic"
            
            # Determine confidence level
            confidence = abs(manipulation_probability - 0.5) * 2  # Scale to 0-1
            confidence_level = "High" if confidence > 0.7 else "Medium" if confidence > 0.4 else "Low"
            
            # Add notes based on individual scores
            notes = []
            
            if 'noise_score' in analysis_result:
                noise_score = analysis_result['noise_score']
                if noise_score < 0.4:
                    notes.append("Low noise patterns detected (typical of AI-generated images)")
                elif noise_score > 0.7:
                    notes.append("Natural noise patterns detected (typical of authentic images)")
            
            if 'symmetry_score' in analysis_result:
                symmetry_score = analysis_result['symmetry_score']
                if symmetry_score < 0.4:
                    notes.append("Unnatural symmetry detected (typical of AI-generated images)")
                elif symmetry_score > 0.7:
                    notes.append("Natural asymmetry detected (typical of authentic images)")
            
            if 'color_score' in analysis_result:
                color_score = analysis_result['color_score']
                if color_score < 0.4:
                    notes.append("Smooth color distribution (typical of AI-generated images)")
                elif color_score > 0.7:
                    notes.append("Natural color variation (typical of authentic images)")
            
            if 'edge_score' in analysis_result:
                edge_score = analysis_result['edge_score']
                if edge_score < 0.4:
                    notes.append("Unnatural edge patterns (typical of AI-generated images)")
                elif edge_score > 0.7:
                    notes.append("Natural edge patterns (typical of authentic images)")
            
            if 'hist_score' in analysis_result:
                hist_score = analysis_result['hist_score']
                if hist_score < 0.4:
                    notes.append("Unusual histogram distribution (typical of AI-generated images)")
                elif hist_score > 0.7:
                    notes.append("Natural histogram distribution (typical of authentic images)")
            
            # Add note about aspect ratio if available
            if 'aspect_ratio_score' in analysis_result:
                aspect_ratio_score = analysis_result['aspect_ratio_score']
                if aspect_ratio_score < 0.2:
                    notes.append("Image has a common AI generation aspect ratio (highly suspicious)")
                elif aspect_ratio_score < 0.4:
                    notes.append("Image aspect ratio matches known AI generator preferences")
            
            # Add notes about pattern analysis (new)
            if 'pattern_score' in analysis_result:
                pattern_score = analysis_result['pattern_score']
                if pattern_score < 0.3:
                    notes.append("Strong grid-like patterns detected (very characteristic of DALL-E images)")
                elif pattern_score < 0.5:
                    notes.append("Suspicious regular patterns detected in frequency domain (common in AI-generated images)")
                elif pattern_score > 0.8:
                    notes.append("No artificial patterns detected (typical of authentic images)")
            
            # Add notes about quantization analysis (new)
            if 'quantization_score' in analysis_result:
                quantization_score = analysis_result['quantization_score']
                if quantization_score < 0.3:
                    notes.append("Suspicious pixel value quantization detected (typical of DALL-E and other AI models)")
                elif quantization_score < 0.5:
                    notes.append("Some pixel value artifacts detected (possibly AI-generated)")
                elif quantization_score > 0.8:
                    notes.append("Natural pixel value distribution (typical of authentic images)")
            
            # Base64 encode image for preview
            try:
                with open(debug_filepath, "rb") as img_file:
                    image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            except Exception as img_error:
                print(f"Error encoding image: {str(img_error)}")
                image_base64 = ""
            
            # Create result with all analysis data
            result = {
                'filename': original_filename,  # Original filename for reference
                'manipulation_probability': float(manipulation_probability),  # Ensure it's a Python float
                'classification': classification,
                'confidence_level': confidence_level,
                'confidence': float(confidence),  # Ensure it's a Python float
                'analysis_notes': notes,
                'image_preview': f"data:image/jpeg;base64,{image_base64}" if image_base64 else None,
                'width': int(analysis_result.get('width', 0)),  # Ensure it's a Python int
                'height': int(analysis_result.get('height', 0)),  # Ensure it's a Python int
                'debug_path': debug_path
            }
            
            # Add visualizations if available
            if 'visualizations' in analysis_result:
                visualizations = {}
                for viz_type, viz_path in analysis_result['visualizations'].items():
                    if viz_path and os.path.exists(viz_path):
                        # Create URL path to access the visualization
                        try:
                            rel_path = os.path.relpath(viz_path, app.config['DEBUG_OUTPUT'])
                            visualizations[viz_type] = f"/debug/{rel_path}"
                        except Exception as viz_error:
                            print(f"Error processing visualization path {viz_path}: {str(viz_error)}")
                
                result['visualizations'] = visualizations
            
            # Add detailed scores if available
            if 'noise_score' in analysis_result:
                result['detailed_scores'] = {
                    'noise_score': float(analysis_result.get('noise_score', 0)),
                    'symmetry_score': float(analysis_result.get('symmetry_score', 0)),
                    'color_score': float(analysis_result.get('color_score', 0)),
                    'edge_score': float(analysis_result.get('edge_score', 0)),
                    'hist_score': float(analysis_result.get('hist_score', 0)),
                    'aspect_ratio_score': float(analysis_result.get('aspect_ratio_score', 0.5)),
                    'pattern_score': float(analysis_result.get('pattern_score', 0.5)),
                    'quantization_score': float(analysis_result.get('quantization_score', 0.5))
                }
            
            # Special DALL-E detection note
            dall_e_indicators_count = 0
            
            # Check for specific DALL-E indicators
            if analysis_result.get('pattern_score', 1.0) < 0.4:  # Strong grid patterns
                dall_e_indicators_count += 1
            if analysis_result.get('quantization_score', 1.0) < 0.4:  # Quantization artifacts
                dall_e_indicators_count += 1
            if analysis_result.get('aspect_ratio_score', 1.0) < 0.3:  # Common aspect ratio
                dall_e_indicators_count += 1
            if analysis_result.get('noise_score', 1.0) < 0.3:  # Very smooth noise profile
                dall_e_indicators_count += 1
            
            # If multiple DALL-E indicators found, add special note
            if dall_e_indicators_count >= 2:
                if 'analysis_notes' in result:
                    result['analysis_notes'].insert(0, "Multiple indicators suggest this may be created by DALL-E or similar AI model")
                
                # Increase confidence for DALL-E detection
                if manipulation_probability > 0.4:  # If already leaning toward AI
                    result['manipulation_probability'] = min(0.95, manipulation_probability * 1.2)
                    result['classification'] = "AI-generated (DALL-E likely)"
            
            # Clean up original upload file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except Exception as clean_error:
                    print(f"Error removing original file: {str(clean_error)}")
            
            # Use the custom JSON encoder to handle NumPy types 
            try:
                response_json = json.dumps(result, cls=NumpyJSONEncoder)
                return Response(response_json, mimetype='application/json')
            except Exception as json_error:
                print(f"Error creating JSON response: {str(json_error)}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'error': f'Error creating response: {str(json_error)}',
                    'traceback': traceback.format_exc()
                }), 500
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"General error in analyze_image route: {str(e)}")
            return jsonify({
                'error': f'Analysis failed: {str(e)}',
                'traceback': traceback.format_exc()
            }), 500
    
    return jsonify({'error': 'Invalid file'}), 400

@app.route('/analyze/text', methods=['POST'])
def analyze_text():
    """Mock text analysis function."""
    if not request.json or 'text' not in request.json:
        return jsonify({'error': 'No text provided'}), 400
    
    text = request.json['text']
    if not text or len(text.strip()) == 0:
        return jsonify({'error': 'Empty text provided'}), 400
    
    ai_probability = random.uniform(0.3, 0.9)
    
    return jsonify({
        'ai_generated_probability': ai_probability,
        'classification': 'AI-generated' if ai_probability > 0.5 else 'Human-written',
        'confidence': random.uniform(0.6, 0.95),
        'analysis_details': {
            'pattern_matches': {
                r'\b(furthermore|moreover|additionally)\b': random.randint(0, 3),
                r'\b(in conclusion|to summarize|in summary)\b': random.randint(0, 2),
                r'\b(it is worth noting|it should be noted)\b': random.randint(0, 2),
            },
            'complexity_score': random.uniform(0.3, 0.8),
            'repetition_score': random.uniform(0.2, 0.6),
            'filler_word_score': random.uniform(0.1, 0.7)
        },
        'analysis_method': 'linguistic_analysis',
        'analysis_complete': True
    })

@app.route('/analyze/video', methods=['POST'])
def analyze_video():
    """Mock video analysis function."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the file temporarily
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    # Create a mock analysis response
    manipulation_probability = random.uniform(0.3, 0.9)
    
    # Clean up
    os.remove(filepath)
    
    return jsonify({
        'deepfake_analysis': {
            'manipulation_probability': manipulation_probability,
            'classification': 'AI-generated' if manipulation_probability > 0.5 else 'Authentic',
            'risk_level': 'High' if manipulation_probability > 0.7 else 'Medium' if manipulation_probability > 0.4 else 'Low',
            'frames_analyzed': random.randint(50, 200),
            'suspicious_frames': random.randint(5, 50),
            'analysis_complete': True
        },
        'metadata_analysis': {
            'file_info': {
                'file_size': 15000000,
                'created_time': datetime.now().isoformat(),
                'modified_time': datetime.now().isoformat(),
                'accessed_time': datetime.now().isoformat(),
                'sha256': 'mock_hash_value'
            },
            'mime_type': 'video/mp4',
            'analysis_complete': True
        }
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    try:
        return render_template('404.html'), 404
    except Exception:
        return "Page not found", 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors."""
    try:
        return render_template('500.html'), 500
    except Exception:
        return "Server error", 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 