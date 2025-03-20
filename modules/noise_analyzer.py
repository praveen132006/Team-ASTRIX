import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pywt
from skimage import restoration

class NoiseAnalyzer:
    def __init__(self):
        """
        Initialize Noise Analyzer for detecting unnatural noise patterns
        in AI-generated images.
        """
        os.makedirs("debug_output/noise", exist_ok=True)
    
    def analyze_image(self, image_path):
        """
        Analyze noise patterns in an image to detect potential AI generation.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing noise analysis results and visualization paths
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")
            
            # Extract noise using different methods
            wavelet_noise = self._extract_noise_wavelet(img)
            highpass_noise = self._extract_noise_highpass(img)
            median_noise = self._extract_noise_median(img)
            
            # Calculate noise statistics
            noise_stats = {
                'wavelet': self._calculate_noise_statistics(wavelet_noise),
                'highpass': self._calculate_noise_statistics(highpass_noise),
                'median': self._calculate_noise_statistics(median_noise)
            }
            
            # Calculate noise inconsistency - a key indicator for AI-generated images
            noise_inconsistency = self._calculate_noise_inconsistency([
                wavelet_noise, highpass_noise, median_noise
            ])
            
            # Calculate local noise variance which often reveals AI generation
            local_noise_map, local_noise_score = self._analyze_local_noise(img)
            
            # Save noise visualizations
            visualization_paths = self._save_noise_visualizations(
                img, wavelet_noise, highpass_noise, median_noise, local_noise_map
            )
            
            # Calculate noise periodicity (AI images often have periodic noise)
            noise_periodicity = self._detect_noise_periodicity(wavelet_noise)
            
            # Calculate overall noise manipulation score
            manipulation_score = self._calculate_noise_manipulation_score(
                noise_inconsistency, local_noise_score, noise_periodicity, noise_stats
            )
            
            return {
                'noise_inconsistency': float(noise_inconsistency),
                'local_noise_score': float(local_noise_score),
                'noise_periodicity': float(noise_periodicity),
                'manipulation_score': float(manipulation_score),
                'noise_statistics': noise_stats,
                'visualization_paths': visualization_paths,
                'analysis_complete': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'analysis_complete': False
            }
    
    def _extract_noise_wavelet(self, img):
        """
        Extract noise using wavelet decomposition.
        This method separates the high-frequency components (noise) from the image.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply wavelet decomposition
        coeffs = pywt.dwt2(gray, 'db8')
        cA, (cH, cV, cD) = coeffs
        
        # Reconstruct image without the high-frequency components
        denoised = pywt.idwt2((cA, (None, None, None)), 'db8')
        
        # Resize denoised image to match original size if needed
        if denoised.shape != gray.shape:
            denoised = cv2.resize(denoised, (gray.shape[1], gray.shape[0]))
        
        # Extract noise
        noise = gray.astype(np.float32) - denoised.astype(np.float32)
        
        return noise
    
    def _extract_noise_highpass(self, img):
        """
        Extract noise using high-pass filtering.
        This method isolates the high-frequency components which contain noise.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (0, 0), 3)
        
        # Subtract the blurred image from the original to get high-frequency components
        noise = gray.astype(np.float32) - blur.astype(np.float32)
        
        return noise
    
    def _extract_noise_median(self, img):
        """
        Extract noise using median filtering.
        This method is effective for extracting impulse and salt-and-pepper noise.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply median filter
        median = cv2.medianBlur(gray, 3)
        
        # Extract noise
        noise = gray.astype(np.float32) - median.astype(np.float32)
        
        return noise
    
    def _analyze_local_noise(self, img, block_size=16):
        """
        Analyze local noise patterns by calculating variance in small blocks.
        AI-generated images often have unnaturally consistent noise patterns.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        height, width = gray.shape
        noise_map = np.zeros((height, width), dtype=np.float32)
        
        # Calculate local variance in sliding windows
        for y in range(0, height - block_size, block_size//2):
            for x in range(0, width - block_size, block_size//2):
                block = gray[y:y+block_size, x:x+block_size].astype(np.float32)
                noise_map[y:y+block_size, x:x+block_size] = np.std(block)
        
        # Calculate inconsistency of local noise
        local_std = np.std(noise_map[noise_map > 0])
        local_mean = np.mean(noise_map[noise_map > 0])
        
        # Normalize to get score between 0 and 1
        # Lower values indicate more consistent noise (potential AI generation)
        local_noise_score = min(1.0, local_std / (local_mean + 1e-6))
        
        return noise_map, local_noise_score
    
    def _calculate_noise_statistics(self, noise):
        """
        Calculate various statistics for noise analysis.
        """
        # Remove potential NaN values
        valid_noise = noise[~np.isnan(noise)]
        
        if len(valid_noise) == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0,
                'entropy': 0.0
            }
        
        # Calculate basic statistics
        mean = float(np.mean(valid_noise))
        std = float(np.std(valid_noise))
        
        # Normalize noise for higher-order statistics
        if std > 0:
            normalized_noise = (valid_noise - mean) / std
        else:
            normalized_noise = valid_noise - mean
        
        # Calculate skewness
        skewness = float(np.mean(normalized_noise**3))
        
        # Calculate kurtosis
        kurtosis = float(np.mean(normalized_noise**4) - 3)  # Excess kurtosis
        
        # Calculate entropy (approximation)
        hist, _ = np.histogram(valid_noise, bins=256, density=True)
        entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        
        return {
            'mean': mean,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'entropy': entropy
        }
    
    def _calculate_noise_inconsistency(self, noise_layers):
        """
        Calculate inconsistency between different noise extraction methods.
        AI-generated images often show inconsistencies between different types of noise.
        """
        # Calculate correlation between different noise layers
        correlations = []
        for i in range(len(noise_layers)):
            for j in range(i+1, len(noise_layers)):
                # Flatten and normalize
                n1 = noise_layers[i].flatten()
                n2 = noise_layers[j].flatten()
                
                # Calculate correlation
                corr = np.corrcoef(n1, n2)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        # Calculate inconsistency score (inverse of average correlation)
        if correlations:
            avg_correlation = np.mean(correlations)
            inconsistency = 1.0 - max(0.0, min(1.0, avg_correlation))
        else:
            inconsistency = 0.5  # Neutral score if correlation can't be calculated
        
        return float(inconsistency)
    
    def _detect_noise_periodicity(self, noise):
        """
        Detect periodic patterns in noise, which can indicate AI generation.
        Uses FFT to detect periodicity.
        """
        # Apply FFT
        fft = np.fft.fft2(noise)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shift)
        
        # Exclude DC component (center of the spectrum)
        center_y, center_x = magnitude.shape[0] // 2, magnitude.shape[1] // 2
        window_size = 5
        magnitude[center_y-window_size:center_y+window_size, center_x-window_size:center_x+window_size] = 0
        
        # Find peaks
        max_magnitude = np.max(magnitude)
        mean_magnitude = np.mean(magnitude)
        
        # Calculate periodicity score (ratio of max to mean)
        if mean_magnitude > 0:
            periodicity = min(1.0, (max_magnitude / mean_magnitude) / 100)
        else:
            periodicity = 0.0
        
        return float(periodicity)
    
    def _calculate_noise_manipulation_score(self, inconsistency, local_score, periodicity, stats):
        """
        Calculate overall noise manipulation score based on multiple indicators.
        """
        # Combine scores with weights
        weights = {
            'inconsistency': 0.35,
            'local': 0.30,
            'periodicity': 0.15,
            'kurtosis': 0.10,
            'entropy': 0.10
        }
        
        # Get kurtosis and entropy scores
        avg_kurtosis = np.mean([abs(stats[key]['kurtosis']) for key in stats])
        kurtosis_score = min(1.0, avg_kurtosis / 5.0)
        
        avg_entropy = np.mean([stats[key]['entropy'] for key in stats])
        entropy_score = max(0.0, min(1.0, 1.0 - (avg_entropy / 8.0)))
        
        # Calculate weighted score
        manipulation_score = (
            inconsistency * weights['inconsistency'] +
            (1.0 - local_score) * weights['local'] +  # Invert since lower local_score means more AI-like
            periodicity * weights['periodicity'] +
            kurtosis_score * weights['kurtosis'] +
            entropy_score * weights['entropy']
        )
        
        return float(manipulation_score)
    
    def _save_noise_visualizations(self, img, wavelet_noise, highpass_noise, median_noise, local_noise_map):
        """
        Save visualizations of the different noise extraction methods.
        """
        # Normalize noise for visualization
        def normalize_for_display(noise):
            return cv2.normalize(noise, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create output paths
        output_paths = {}
        
        # Save wavelet noise
        wavelet_viz = normalize_for_display(wavelet_noise)
        wavelet_path = "debug_output/noise/wavelet_noise.jpg"
        cv2.imwrite(wavelet_path, wavelet_viz)
        output_paths['wavelet'] = wavelet_path
        
        # Save highpass noise
        highpass_viz = normalize_for_display(highpass_noise)
        highpass_path = "debug_output/noise/highpass_noise.jpg"
        cv2.imwrite(highpass_path, highpass_viz)
        output_paths['highpass'] = highpass_path
        
        # Save median noise
        median_viz = normalize_for_display(median_noise)
        median_path = "debug_output/noise/median_noise.jpg"
        cv2.imwrite(median_path, median_viz)
        output_paths['median'] = median_path
        
        # Save local noise map
        local_viz = normalize_for_display(local_noise_map)
        local_path = "debug_output/noise/local_noise_map.jpg"
        cv2.imwrite(local_path, local_viz)
        output_paths['local'] = local_path
        
        # Create combined noise visualization
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Wavelet noise
        axes[0, 1].imshow(wavelet_viz, cmap='gray')
        axes[0, 1].set_title('Wavelet Noise')
        axes[0, 1].axis('off')
        
        # Highpass noise
        axes[1, 0].imshow(highpass_viz, cmap='gray')
        axes[1, 0].set_title('Highpass Noise')
        axes[1, 0].axis('off')
        
        # Local noise variance
        local_viz_color = cv2.applyColorMap(local_viz, cv2.COLORMAP_JET)
        axes[1, 1].imshow(cv2.cvtColor(local_viz_color, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('Local Noise Variance')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        combined_path = "debug_output/noise/combined_noise_analysis.jpg"
        plt.savefig(combined_path)
        plt.close()
        
        output_paths['combined'] = combined_path
        
        return output_paths 