import cv2
import numpy as np
from PIL import Image
import requests
import os
from dotenv import load_dotenv
import urllib.request
from modules.ela_analyzer import ELAAnalyzer
from modules.noise_analyzer import NoiseAnalyzer
from modules.cnn_classifier import CNNClassifier

class DeepfakeDetector:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Define analysis parameters
        self.ela_quality = 90
        self.temp_path = "temp_ela.jpg"
        
        # Model paths
        self.model_path = "models/face_detector.caffemodel"
        self.config_path = "models/face_detector.prototxt"
        
        # Analysis thresholds
        self.frequency_threshold = 0.6
        self.noise_threshold = 0.4
        self.compression_threshold = 0.5
        self.artifact_threshold = 0.45
        
        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)
        os.makedirs("debug_output", exist_ok=True)
        
        # Download pre-trained models if they don't exist
        self._download_models()
        
        # Load face detector model
        self.face_detector = cv2.dnn.readNet(self.config_path, self.model_path)
        
        # Initialize analyzer modules
        self.ela_analyzer = ELAAnalyzer(quality=self.ela_quality)
        self.noise_analyzer = NoiseAnalyzer()
        self.cnn_classifier = CNNClassifier()
        
    def _download_models(self):
        """Download required models if they don't exist."""
        models = {
            'face_detector.prototxt': 'https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt',
            'face_detector.caffemodel': 'https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel'
        }
        
        for filename, url in models.items():
            filepath = os.path.join("models", filename)
            if not os.path.exists(filepath):
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
        
    def analyze_image(self, image_path):
        """Enhanced analysis of an image for potential AI generation or manipulation."""
        try:
            # Load and prepare image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")
            
            print(f"[DEBUG] Analyzing image: {image_path}")
            height, width = img.shape[:2]
            
            # Run ELA analysis
            ela_results = self.ela_analyzer.analyze_image(image_path)
            
            # Run noise analysis
            noise_results = self.noise_analyzer.analyze_image(image_path)
            
            # Run CNN classifier
            cnn_results = self.cnn_classifier.predict(image_path)
            
            # Initialize analysis components
            analysis_results = {
                'facial_analysis': self._analyze_faces(img),
                'frequency_analysis': self._analyze_frequency_domain(img),
                'noise_analysis': noise_results,
                'ela_analysis': ela_results,
                'cnn_analysis': cnn_results,
                'compression_analysis': self._analyze_compression(img),
                'metadata_analysis': self._analyze_metadata(image_path),
                'artifact_analysis': self._detect_ai_artifacts(img)
            }
            
            # Save debug visualizations
            self._save_debug_visualizations(img, analysis_results)
            
            # Calculate final scores with more weight on new analysis methods
            manipulation_score = self._calculate_final_score(analysis_results)
            confidence_score = self._calculate_confidence(analysis_results)
            
            # Generate detailed report
            return {
                'manipulation_probability': float(manipulation_score),
                'confidence_score': float(confidence_score),
                'analysis_details': {
                    'faces_detected': analysis_results['facial_analysis']['num_faces'],
                    'face_analysis_scores': analysis_results['facial_analysis']['face_scores'],
                    'frequency_anomalies': analysis_results['frequency_analysis']['anomalies'],
                    'noise_inconsistency': analysis_results['noise_analysis']['noise_inconsistency'],
                    'ela_score': analysis_results['ela_analysis']['ela_score'],
                    'ai_generated_probability': analysis_results['cnn_analysis'].get('ai_generated_probability', 0),
                    'compression_artifacts': analysis_results['compression_analysis']['artifact_score'],
                    'ai_artifacts': analysis_results['artifact_analysis']['detected_patterns'],
                    'metadata_inconsistencies': analysis_results['metadata_analysis']['inconsistencies']
                },
                'visualizations': {
                    'ela_heatmap': ela_results.get('ela_heatmap', None),
                    'noise_map': noise_results.get('visualization_paths', {}).get('combined', None),
                    'cnn_heatmap': cnn_results.get('visualization_path', None),
                    'face_detection': 'debug_output/face_detection.jpg'
                },
                'forensic_markers': self._identify_forensic_markers(analysis_results),
                'recommendations': self._generate_detailed_recommendations(analysis_results),
                'analysis_complete': True
            }
            
        except Exception as e:
            print(f"[ERROR] Analysis failed: {str(e)}")
            return {'error': str(e), 'analysis_complete': False}

    def _analyze_faces(self, img):
        """Enhanced facial analysis with advanced detection of AI artifacts."""
        try:
            height, width = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            faces_analyzed = []
            face_boxes = []
            face_features = []
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.3:
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width))
                    y2 = max(0, min(y2, height))
                    
                    if x2 > x1 and y2 > y1:  # Ensure valid face region
                        face = img[y1:y2, x1:x2]
                        if face.size > 0 and face.shape[0] > 0 and face.shape[1] > 0:
                            # Analyze face region
                            face_analysis = {
                                'texture': self._analyze_facial_texture(face),
                                'color': self._analyze_color_consistency(face),
                                'geometry': self._analyze_facial_geometry(face),
                                'landmarks': self._analyze_facial_landmarks(face),
                                'symmetry': self._analyze_facial_symmetry(face)
                            }
                            
                            # Calculate face manipulation score
                            face_score = self._calculate_face_score(face_analysis)
                            faces_analyzed.append(float(face_score))
                            face_boxes.append((int(x1), int(y1), int(x2), int(y2)))
                            face_features.append(face_analysis)
            
            return {
                'num_faces': len(faces_analyzed),
                'face_scores': faces_analyzed,
                'face_boxes': face_boxes,
                'face_features': face_features
            }
        except Exception as e:
            print(f"[ERROR] Error in face analysis: {str(e)}")
            return {
                'num_faces': 0,
                'face_scores': [],
                'face_boxes': [],
                'face_features': []
            }

    def _analyze_frequency_domain(self, img):
        """Analyze frequency domain for AI generation artifacts."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply DFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
        
        # Analyze frequency patterns
        anomalies = []
        
        # Check for regular patterns indicating AI generation
        freq_std = np.std(magnitude_spectrum)
        freq_mean = np.mean(magnitude_spectrum)
        freq_peaks = self._detect_frequency_peaks(magnitude_spectrum)
        
        if freq_std < self.frequency_threshold:
            anomalies.append('Unusually regular frequency patterns')
        
        # Save frequency visualization
        cv2.imwrite('debug_output/frequency_analysis.jpg', magnitude_spectrum)
        
        return {
            'anomalies': anomalies,
            'frequency_std': float(freq_std),
            'frequency_mean': float(freq_mean),
            'frequency_peaks': freq_peaks
        }

    def _analyze_noise_patterns(self, img):
        """Advanced noise pattern analysis for AI detection."""
        # Extract noise using multiple methods
        noise_layers = []
        
        # Method 1: Wavelet decomposition
        noise_wavelet = self._extract_noise_wavelet(img)
        
        # Method 2: High-pass filter
        noise_highpass = self._extract_noise_highpass(img)
        
        # Method 3: Local variance analysis
        noise_local = self._analyze_local_noise(img)
        
        # Combine noise analyses
        noise_inconsistency = self._calculate_noise_inconsistency([
            noise_wavelet, noise_highpass, noise_local
        ])
        
        return {
            'inconsistency': float(noise_inconsistency),
            'noise_patterns': {
                'wavelet': float(np.mean(noise_wavelet)),
                'highpass': float(np.mean(noise_highpass)),
                'local': float(np.mean(noise_local))
            }
        }

    def _detect_ai_artifacts(self, img):
        """Detect specific artifacts common in AI-generated images."""
        artifacts = []
        
        # Check for repeating patterns
        if self._detect_repeating_patterns(img):
            artifacts.append('Repeating texture patterns')
        
        # Check for unnatural color transitions
        if self._detect_color_artifacts(img):
            artifacts.append('Unnatural color transitions')
        
        # Check for inconsistent lighting
        if self._detect_lighting_inconsistencies(img):
            artifacts.append('Inconsistent lighting patterns')
        
        # Check for geometric irregularities
        if self._detect_geometric_artifacts(img):
            artifacts.append('Geometric irregularities')
        
        return {
            'detected_patterns': artifacts,
            'artifact_score': len(artifacts) / 4.0
        }

    def _analyze_compression(self, img):
        """Analyze compression artifacts in the image."""
        try:
            # Detect compression artifacts
            compression_score = self._detect_compression_artifacts(img)
            
            # Analyze compression patterns
            artifacts = []
            if compression_score > self.compression_threshold:
                artifacts.append("High compression artifacts detected")
            
            return {
                'artifact_score': float(compression_score / 1000.0),  # Normalize to 0-1
                'artifacts': artifacts
            }
        except Exception as e:
            print(f"[ERROR] Error in compression analysis: {str(e)}")
            return {
                'artifact_score': 0.0,
                'artifacts': []
            }

    def _calculate_final_score(self, analysis_results):
        """Calculate final manipulation probability with weighted analysis including new modules."""
        try:
            weights = {
                'facial': 0.20,
                'frequency': 0.10,
                'noise': 0.15,
                'ela': 0.15,
                'cnn': 0.20,
                'compression': 0.10,
                'artifacts': 0.05,
                'metadata': 0.05
            }
            
            # Extract and validate scores
            scores = {
                'facial': float(np.mean(analysis_results['facial_analysis']['face_scores'])) if analysis_results['facial_analysis']['face_scores'] else 0.5,
                'frequency': float(len(analysis_results['frequency_analysis']['anomalies'])) / 5.0,
                'noise': float(analysis_results['noise_analysis']['noise_inconsistency']),
                'ela': float(analysis_results['ela_analysis']['ela_score']),
                'cnn': float(analysis_results['cnn_analysis'].get('ai_generated_probability', 0.5)),
                'compression': float(analysis_results['compression_analysis']['artifact_score']),
                'artifacts': float(analysis_results['artifact_analysis']['artifact_score']),
                'metadata': float(len(analysis_results['metadata_analysis']['inconsistencies'])) / 5.0
            }
            
            # Calculate weighted score
            final_score = sum(float(scores[k]) * weights[k] for k in weights)
            
            # Boost score if multiple strong indicators
            strong_indicators = sum(1 for score in scores.values() if float(score) > 0.7)
            if strong_indicators >= 3:
                final_score = min(1.0, final_score * 1.2)
            
            return float(final_score)
            
        except Exception as e:
            print(f"[ERROR] Error in final score calculation: {str(e)}")
            return 0.5  # Return neutral score on error

    def _identify_forensic_markers(self, analysis_results):
        """Identify specific forensic markers indicating manipulation."""
        markers = []
        
        # Face-related markers
        if analysis_results['facial_analysis']['num_faces'] > 0:
            face_scores = analysis_results['facial_analysis']['face_scores']
            if any(score > 0.8 for score in face_scores):
                markers.append('High confidence facial manipulation detected')
            if any(score > 0.6 for score in face_scores):
                markers.append('Moderate facial anomalies present')
        
        # Frequency domain markers
        if analysis_results['frequency_analysis']['anomalies']:
            markers.extend(analysis_results['frequency_analysis']['anomalies'])
        
        # Noise pattern markers
        if analysis_results['noise_analysis']['noise_inconsistency'] > self.noise_threshold:
            markers.append('Inconsistent noise patterns detected')
        
        # ELA markers
        if analysis_results['ela_analysis']['ela_score'] > 0.6:
            markers.append('Error Level Analysis indicates potential manipulation')
        if analysis_results['ela_analysis'].get('high_ela_regions', []):
            markers.append('High ELA regions detected, suggesting tampering')
        
        # CNN markers
        if analysis_results['cnn_analysis'].get('ai_generated_probability', 0) > 0.7:
            markers.append('CNN classifier detected AI generation patterns with high confidence')
        
        # AI artifact markers
        markers.extend(analysis_results['artifact_analysis']['detected_patterns'])
        
        return markers

    def _generate_detailed_recommendations(self, analysis_results):
        """Generate detailed recommendations based on analysis results."""
        recommendations = []
        
        # Face-specific recommendations
        if analysis_results['facial_analysis']['num_faces'] > 0:
            face_scores = analysis_results['facial_analysis']['face_scores']
            max_face_score = max(face_scores)
            if max_face_score > 0.8:
                recommendations.append("High probability of facial manipulation detected. Recommend thorough verification of source material.")
            elif max_face_score > 0.6:
                recommendations.append("Moderate facial anomalies detected. Suggest cross-referencing with original sources.")
        
        # Frequency analysis recommendations
        if analysis_results['frequency_analysis']['anomalies']:
            recommendations.append("Unusual frequency patterns detected, suggesting potential AI generation.")
        
        # Noise analysis recommendations
        if analysis_results['noise_analysis']['noise_inconsistency'] > self.noise_threshold:
            recommendations.append("Inconsistent noise patterns indicate possible image manipulation.")
        
        # ELA recommendations
        if analysis_results['ela_analysis']['ela_score'] > 0.6:
            recommendations.append("Error Level Analysis shows abnormal compression artifacts, suggesting editing.")
        
        # CNN recommendations
        if analysis_results['cnn_analysis'].get('ai_generated_probability', 0) > 0.7:
            recommendations.append("Neural network analysis strongly indicates AI generation. Verify authenticity.")
        
        # Artifact recommendations
        if analysis_results['artifact_analysis']['detected_patterns']:
            recommendations.append("AI generation artifacts detected. Review image carefully for authenticity.")
        
        # Metadata recommendations
        if analysis_results['metadata_analysis']['inconsistencies']:
            recommendations.append("Metadata inconsistencies found. Verify image origin and history.")
        
        # Overall recommendation based on final score
        final_score = self._calculate_final_score(analysis_results)
        if final_score > 0.8:
            recommendations.append("HIGH RISK: This image shows strong indicators of AI generation or manipulation. Do not trust without thorough verification.")
        elif final_score > 0.5:
            recommendations.append("MEDIUM RISK: This image shows some signs of potential manipulation. Use caution and verify source.")
        else:
            recommendations.append("LOW RISK: This image appears to be authentic based on our analysis, but always verify sensitive content.")
        
        return " ".join(recommendations)

    def _save_debug_visualizations(self, img, analysis_results):
        """Save visualization of analysis results for debugging."""
        # Create debug directory if it doesn't exist
        os.makedirs('debug_output', exist_ok=True)
        
        # Save annotated image with face detections
        debug_img = img.copy()
        for box in analysis_results['facial_analysis']['face_boxes']:
            x1, y1, x2, y2 = box
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite('debug_output/face_detection.jpg', debug_img)
        
        # Note: The other visualizations are already saved by their respective analyzers
        # ELA heatmap is saved by ela_analyzer
        # Noise map is saved by noise_analyzer
        # CNN heatmap is saved by cnn_classifier

    # Helper methods for specific analyses
    def _analyze_facial_texture(self, face):
        """Analyze facial texture patterns for inconsistencies."""
        try:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            lbp = self._compute_lbp(gray)
            glcm_score = self._compute_glcm(gray)
            
            # Normalize scores
            lbp_score = float(np.mean(lbp) / 255.0)
            glcm_score = float(min(1.0, glcm_score / 100.0))  # Normalize GLCM score
            
            return {
                'lbp_score': lbp_score,
                'glcm_score': glcm_score,
                'texture_score': float((lbp_score + glcm_score) / 2.0)  # Combined score
            }
        except Exception as e:
            print(f"[ERROR] Error in facial texture analysis: {str(e)}")
            return {
                'lbp_score': 0.5,
                'glcm_score': 0.5,
                'texture_score': 0.5
            }

    def _analyze_facial_geometry(self, face):
        """Analyze facial geometric properties for irregularities."""
        # Implement facial geometry analysis
        # This is a placeholder for the actual implementation
        return {'geometry_score': 0.5}

    def _analyze_facial_landmarks(self, face):
        """Analyze facial landmark positions and relationships."""
        # Implement facial landmark analysis
        # This is a placeholder for the actual implementation
        return {'landmark_score': 0.5}

    def _analyze_facial_symmetry(self, face):
        """Analyze facial symmetry for unnatural patterns."""
        try:
            height, width = face.shape[:2]
            mid = width // 2
            
            # Ensure even width for proper comparison
            if width % 2 != 0:
                width = width - 1
                face = face[:, :width]
                mid = width // 2
            
            # Split face into left and right halves
            left_side = face[:, :mid]
            right_side = face[:, mid:]
            
            # Flip right side for comparison
            right_side_flipped = cv2.flip(right_side, 1)
            
            # Ensure both sides have the same size
            min_height = min(left_side.shape[0], right_side_flipped.shape[0])
            min_width = min(left_side.shape[1], right_side_flipped.shape[1])
            
            left_side = left_side[:min_height, :min_width]
            right_side_flipped = right_side_flipped[:min_height, :min_width]
            
            # Calculate symmetry score
            symmetry_diff = cv2.absdiff(left_side, right_side_flipped)
            symmetry_score = 1.0 - (np.mean(symmetry_diff) / 255.0)
            
            return {'symmetry_score': float(symmetry_score)}
        except Exception as e:
            print(f"[ERROR] Error in facial symmetry analysis: {str(e)}")
            return {'symmetry_score': 0.5}

    def _detect_repeating_patterns(self, img):
        """Detect unnaturally repeating patterns in the image."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply template matching at multiple scales
        scales = [0.5, 1.0, 2.0]
        pattern_scores = []
        
        for scale in scales:
            resized = cv2.resize(gray, None, fx=scale, fy=scale)
            result = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
            pattern_scores.append(np.max(result))
        
        return np.mean(pattern_scores) > self.artifact_threshold

    def _detect_color_artifacts(self, img):
        """Detect unnatural color transitions and patterns."""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate color gradients
        gradient_l = np.gradient(l)
        gradient_a = np.gradient(a)
        gradient_b = np.gradient(b)
        
        # Check for unnatural transitions
        gradient_magnitudes = np.sqrt(
            gradient_l[0]**2 + gradient_l[1]**2 +
            gradient_a[0]**2 + gradient_a[1]**2 +
            gradient_b[0]**2 + gradient_b[1]**2
        )
        
        return np.mean(gradient_magnitudes) > self.artifact_threshold

    def _detect_lighting_inconsistencies(self, img):
        """Detect inconsistent lighting patterns."""
        # Convert to YCrCb
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:,:,0]
        
        # Analyze lighting gradients
        gradient_y = np.gradient(y)
        gradient_magnitude = np.sqrt(gradient_y[0]**2 + gradient_y[1]**2)
        
        return np.std(gradient_magnitude) > self.artifact_threshold

    def _detect_geometric_artifacts(self, img):
        """Detect geometric irregularities common in AI-generated images."""
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Analyze edge patterns
        lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
        if lines is not None:
            return len(lines) > 50  # Threshold for excessive regular lines
        
        return False

    def _compute_lbp(self, gray_img):
        """Compute Local Binary Pattern features."""
        rows, cols = gray_img.shape
        lbp = np.zeros_like(gray_img)
        
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = gray_img[i, j]
                code = 0
                code |= (gray_img[i-1, j-1] >= center) << 7
                code |= (gray_img[i-1, j] >= center) << 6
                code |= (gray_img[i-1, j+1] >= center) << 5
                code |= (gray_img[i, j+1] >= center) << 4
                code |= (gray_img[i+1, j+1] >= center) << 3
                code |= (gray_img[i+1, j] >= center) << 2
                code |= (gray_img[i+1, j-1] >= center) << 1
                code |= (gray_img[i, j-1] >= center) << 0
                lbp[i, j] = code
                
        return lbp

    def _compute_glcm(self, gray_img):
        """Compute Gray-Level Co-occurrence Matrix features."""
        try:
            # Simple GLCM implementation
            # Normalize image to reduce computation
            normalized = cv2.normalize(gray_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Calculate horizontal GLCM
            glcm = np.zeros((256, 256), dtype=np.float32)
            rows, cols = normalized.shape
            
            for i in range(rows):
                for j in range(cols-1):
                    i_val = normalized[i, j]
                    j_val = normalized[i, j+1]
                    glcm[i_val, j_val] += 1
            
            # Normalize GLCM
            if glcm.sum() > 0:
                glcm = glcm / glcm.sum()
            
            # Calculate contrast
            indices = np.indices((256, 256))
            contrast = np.sum(glcm * ((indices[0] - indices[1]) ** 2))
            
            return float(contrast)
        except Exception as e:
            print(f"[ERROR] Error in GLCM computation: {str(e)}")
            return 0.5  # Return neutral value on error

    def _detect_frequency_peaks(self, spectrum):
        """Detect frequency peaks in the spectrum."""
        # Implement frequency peak detection logic
        # This is a placeholder for the actual implementation
        return []

    def _calculate_noise_inconsistency(self, noise_layers):
        """Calculate noise inconsistency based on multiple noise layers."""
        # Implement noise inconsistency calculation logic
        # This is a placeholder for the actual implementation
        return 0.5

    def _extract_noise_wavelet(self, img):
        """Extract noise using wavelet decomposition."""
        # Implement wavelet decomposition logic
        # This is a placeholder for the actual implementation
        return []

    def _extract_noise_highpass(self, img):
        """Extract noise using high-pass filter."""
        # Implement high-pass filter logic
        # This is a placeholder for the actual implementation
        return []

    def _analyze_local_noise(self, img):
        """Analyze local noise patterns."""
        # Implement local noise analysis logic
        # This is a placeholder for the actual implementation
        return []

    def _analyze_color_consistency(self, face_img):
        """Analyze color consistency in face region."""
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate local color consistency
        window_size = 8
        inconsistency = 0
        
        for channel in [l, a, b]:
            # Calculate local variance in sliding windows
            kernel = np.ones((window_size, window_size)) / (window_size * window_size)
            local_mean = cv2.filter2D(channel.astype(float), -1, kernel)
            local_mean_sq = cv2.filter2D(channel.astype(float)**2, -1, kernel)
            local_var = local_mean_sq - local_mean**2
            
            # Higher variance indicates potential inconsistency
            inconsistency += np.mean(local_var)
        
        # Normalize and invert (higher score = more consistent)
        color_score = 1.0 - min(1.0, inconsistency / (3 * 255 * 255))
        return float(color_score)

    def _calculate_face_score(self, face_analysis):
        """Calculate face manipulation score based on analysis results."""
        # Implement face manipulation score calculation logic
        # This is a placeholder for the actual implementation
        return 0.5

    def _calculate_confidence(self, analysis_results):
        """Calculate confidence score for the analysis."""
        # Implement confidence score calculation logic
        # This is a placeholder for the actual implementation
        return 0.5

    def _analyze_metadata(self, image_path):
        """Analyze image metadata for inconsistencies."""
        # Implement metadata analysis logic
        # This is a placeholder for the actual implementation
        return {'inconsistencies': []}

    def analyze_video(self, video_path):
        """Analyze a video for potential manipulation."""
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            total_score = 0
            frame_results = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 10th frame
                if frame_count % 10 == 0:
                    # Save frame temporarily
                    temp_frame_path = f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(temp_frame_path, frame)
                    
                    # Analyze frame
                    result = self.analyze_image(temp_frame_path)
                    frame_results.append(result)
                    total_score += result.get('manipulation_probability', 0)
                    
                    # Clean up
                    os.remove(temp_frame_path)
                
                frame_count += 1
            
            cap.release()
            
            # Calculate average score
            processed_frames = len(frame_results)
            avg_score = total_score / processed_frames if processed_frames > 0 else 0
            
            return {
                'manipulation_probability': float(avg_score),
                'frames_analyzed': frame_count,
                'processed_frames': processed_frames,
                'frame_results': frame_results,
                'analysis_complete': True,
                'analysis_method': 'video_analysis'
            }
        except Exception as e:
            return {
                'error': str(e),
                'analysis_complete': False
            }
    
    def _analyze_noise(self, image):
        """Analyze noise levels in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_sigma = np.std(gray)
        return float(noise_sigma)
    
    def _detect_compression_artifacts(self, image):
        """Detect compression artifacts in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        compression_score = np.var(laplacian)
        return float(compression_score)
    
    def _calculate_manipulation_score(self, ela_score, noise_level, compression_artifacts, face_scores):
        """Calculate overall manipulation probability."""
        try:
            # Ensure all inputs are valid numbers
            ela_score = float(ela_score) if ela_score is not None else 0.0
            noise_level = float(noise_level) if noise_level is not None else 0.0
            compression_artifacts = float(compression_artifacts) if compression_artifacts is not None else 0.0
            
            # Normalize scores with adjusted thresholds and validation
            normalized_ela = min(1.0, max(0.0, ela_score / 50))  # Lowered threshold
            normalized_noise = min(1.0, max(0.0, noise_level / 30))  # Lowered threshold
            normalized_compression = min(1.0, max(0.0, compression_artifacts / 500))  # Lowered threshold
            
            print(f"[DEBUG] Normalized scores:")
            print(f"- ELA: {normalized_ela}")
            print(f"- Noise: {normalized_noise}")
            print(f"- Compression: {normalized_compression}")
            
            # Calculate face manipulation score with higher weight
            face_manipulation = 0.0
            if face_scores and isinstance(face_scores, (list, tuple)):  # Ensure face_scores is a sequence
                # Filter out any invalid scores and ensure they're numbers
                valid_scores = [float(score) for score in face_scores if isinstance(score, (int, float, str)) and not isinstance(score, bool)]
                if valid_scores:
                    face_manipulation = max(valid_scores)
                print(f"[DEBUG] Face manipulation score: {face_manipulation}")
            
            # Updated weights to give more importance to face analysis
            weights = {
                'ela': 0.25,
                'noise': 0.2,
                'compression': 0.15,
                'face': 0.4  # Slightly reduced to prevent over-reliance on face detection
            }
            
            # Calculate weighted average with validation
            manipulation_score = (
                normalized_ela * weights['ela'] +
                normalized_noise * weights['noise'] +
                normalized_compression * weights['compression'] +
                face_manipulation * weights['face']
            )
            
            # Ensure score is valid
            if np.isnan(manipulation_score) or manipulation_score < 0:
                manipulation_score = 0.0
            elif manipulation_score > 1:
                manipulation_score = 1.0
            
            # Boost score if multiple indicators are high (lowered threshold)
            high_indicators = sum(1 for score in [normalized_ela, normalized_noise, normalized_compression, face_manipulation] if score > 0.4)
            if high_indicators >= 2:
                manipulation_score = min(1.0, manipulation_score * 1.3)
                print("[DEBUG] Score boosted due to multiple high indicators")
            
            return float(manipulation_score)
            
        except Exception as e:
            print(f"[ERROR] Error in manipulation score calculation: {str(e)}")
            return 0.0  # Return safe default on error

    def _analyze_face(self, face_img):
        """Analyze a face region for potential manipulation markers."""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            
            # Calculate texture features using LBP
            lbp = self._compute_lbp(gray)
            texture_score = np.mean(lbp) / 255.0  # Normalize to 0-1
            
            # Calculate color consistency
            color_score = self._analyze_color_consistency(face_img)
            
            # Calculate edge coherence
            edge_score = self._analyze_edge_coherence(gray)
            
            # Calculate noise inconsistency
            noise_score = self._analyze_face_noise(gray)
            
            # Calculate quality degradation
            quality_score = self._analyze_quality_degradation(face_img)
            
            # Validate all scores
            scores = {
                'texture': texture_score,
                'color': color_score,
                'edge': edge_score,
                'noise': noise_score,
                'quality': quality_score
            }
            
            # Replace any invalid scores with neutral value
            for key in scores:
                if scores[key] is None or np.isnan(scores[key]):
                    scores[key] = 0.5
                scores[key] = min(1.0, max(0.0, float(scores[key])))
            
            # Combine scores with weights
            weights = {
                'texture': 0.25,
                'color': 0.25,
                'edge': 0.2,
                'noise': 0.15,
                'quality': 0.15
            }
            
            face_score = sum(scores[k] * weights[k] for k in weights)
            
            # Ensure final score is valid
            if np.isnan(face_score) or face_score < 0:
                face_score = 0.5  # Return neutral score on error
            elif face_score > 1:
                face_score = 1.0
                
            return float(face_score)
            
        except Exception as e:
            print(f"[ERROR] Error in face analysis: {str(e)}")
            return 0.5  # Return neutral score on error
    
    def _analyze_edge_coherence(self, gray_img):
        """Analyze edge coherence for manipulation detection."""
        # Calculate gradients using Sobel
        sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        # Analyze local edge direction consistency
        window_size = 16
        coherence_map = np.zeros_like(magnitude)
        
        for i in range(0, gray_img.shape[0] - window_size, window_size//2):
            for j in range(0, gray_img.shape[1] - window_size, window_size//2):
                window_dir = direction[i:i+window_size, j:j+window_size]
                window_mag = magnitude[i:i+window_size, j:j+window_size]
                
                # Calculate direction consistency weighted by magnitude
                dir_std = np.std(window_dir, weights=window_mag)
                coherence_map[i:i+window_size, j:j+window_size] = np.exp(-dir_std)
        
        # Calculate overall coherence score
        edge_score = np.mean(coherence_map)
        return float(edge_score)

    def _analyze_face_noise(self, gray_img):
        """Analyze noise patterns for inconsistencies."""
        # Apply median filter to remove noise
        denoised = cv2.medianBlur(gray_img, 3)
        
        # Calculate noise as difference from denoised image
        noise = gray_img.astype(float) - denoised.astype(float)
        
        # Analyze noise statistics in local windows
        window_size = 16
        noise_consistency = []
        
        for i in range(0, gray_img.shape[0] - window_size, window_size//2):
            for j in range(0, gray_img.shape[1] - window_size, window_size//2):
                window_noise = noise[i:i+window_size, j:j+window_size]
                noise_consistency.append(np.std(window_noise))
        
        # Calculate noise consistency score
        noise_std = np.std(noise_consistency)
        noise_score = 1.0 - min(1.0, noise_std / 50.0)  # Normalize to 0-1
        return float(noise_score)

    def _analyze_quality_degradation(self, face_img):
        """Analyze quality degradation patterns."""
        # Convert to YCrCb color space
        ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        
        # Apply DCT transform to detect compression artifacts
        h, w = y.shape
        h_pad = h - (h % 8)
        w_pad = w - (w % 8)
        y_blocks = y[:h_pad, :w_pad].reshape(h_pad//8, 8, w_pad//8, 8).swapaxes(1, 2)
        
        # Calculate DCT energy distribution
        dct_energy = []
        for i in range(y_blocks.shape[0]):
            for j in range(y_blocks.shape[1]):
                block = y_blocks[i, j].astype(float)
                dct = cv2.dct(block)
                energy = np.sum(np.abs(dct))
                dct_energy.append(energy)
        
        # Analyze energy distribution
        energy_std = np.std(dct_energy)
        quality_score = 1.0 - min(1.0, energy_std / 1000.0)  # Normalize to 0-1
        return float(quality_score) 