import cv2
import numpy as np
from PIL import Image, ImageChops
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments

class ELAAnalyzer:
    def __init__(self, quality=90, scale=10):
        """
        Initialize ELA Analyzer.
        
        Args:
            quality: JPEG compression quality (0-100)
            scale: Amplification factor for visualization
        """
        self.quality = quality
        self.scale = scale
        os.makedirs("debug_output/ela", exist_ok=True)
    
    def analyze_image(self, image_path):
        """
        Perform Error Level Analysis on an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing ELA results and paths to visualization
        """
        try:
            # Open image with PIL
            original = Image.open(image_path)
            
            # Convert to RGB if needed
            if original.mode != 'RGB':
                original = original.convert('RGB')
            
            # Save a temporary copy with JPEG compression
            temp_path = "debug_output/ela/temp_ela.jpg"
            original.save(temp_path, quality=self.quality, optimize=True)
            
            # Open the temp image
            compressed = Image.open(temp_path)
            
            # Calculate the difference
            ela_image = ImageChops.difference(original, compressed)
            
            # Scale the difference for visualization
            extrema = ela_image.getextrema()
            max_diff = max([ex[1] for ex in extrema])
            if max_diff == 0:
                max_diff = 1  # Avoid division by zero
            
            # Amplify the difference
            ela_image = ImageChops.multiply(ela_image, Image.new('RGB', original.size, (self.scale, self.scale, self.scale)))
            
            # Convert to numpy array for additional analysis
            ela_array = np.array(ela_image)
            
            # Calculate ELA score (normalized mean difference)
            ela_mean = np.mean(ela_array) / (max_diff * self.scale)
            
            # Identify regions with high ELA values (potential manipulation)
            high_ela_regions = self._identify_high_ela_regions(ela_array)
            
            # Save visualization
            ela_path = "debug_output/ela/ela_result.jpg"
            ela_image.save(ela_path)
            
            # Create heatmap for better visualization
            heatmap_path = self._generate_heatmap(ela_array, image_path)
            
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return {
                'ela_score': float(ela_mean),
                'ela_visualization': ela_path,
                'ela_heatmap': heatmap_path,
                'high_ela_regions': high_ela_regions,
                'analysis_complete': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'analysis_complete': False
            }
    
    def _identify_high_ela_regions(self, ela_array, threshold_percentile=95):
        """
        Identify regions with high ELA values that may indicate manipulation.
        
        Args:
            ela_array: NumPy array of the ELA image
            threshold_percentile: Percentile threshold for high values
            
        Returns:
            List of regions with high ELA values
        """
        # Calculate threshold based on percentile
        threshold = np.percentile(ela_array, threshold_percentile)
        
        # Find regions above threshold
        high_values = np.where(ela_array > threshold)
        
        # If no high regions found
        if len(high_values[0]) == 0:
            return []
        
        # Group into contiguous regions
        # For simplicity, we'll just report the bounding boxes
        y_min, y_max = np.min(high_values[0]), np.max(high_values[0])
        x_min, x_max = np.min(high_values[1]), np.max(high_values[1])
        
        high_regions = [{
            'x_min': int(x_min),
            'y_min': int(y_min),
            'x_max': int(x_max),
            'y_max': int(y_max),
            'size': int((x_max - x_min) * (y_max - y_min)),
            'intensity': float(np.mean(ela_array[high_values]))
        }]
        
        return high_regions
    
    def _generate_heatmap(self, ela_array, original_path):
        """
        Generate a heatmap visualization of the ELA results.
        
        Args:
            ela_array: NumPy array of the ELA image
            original_path: Path to the original image
            
        Returns:
            Path to the generated heatmap image
        """
        # Convert to grayscale for heatmap
        ela_gray = cv2.cvtColor(ela_array, cv2.COLOR_RGB2GRAY)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        plt.imshow(ela_gray, cmap='hot')
        plt.colorbar(label='ELA Intensity')
        plt.title('Error Level Analysis Heatmap')
        plt.axis('off')
        
        # Save heatmap
        heatmap_path = "debug_output/ela/ela_heatmap.jpg"
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()
        
        # Create overlay on original image
        original = cv2.imread(original_path)
        
        # Resize ELA to match original if needed
        if original.shape[:2] != ela_gray.shape[:2]:
            ela_gray = cv2.resize(ela_gray, (original.shape[1], original.shape[0]))
        
        # Normalize to 0-255
        ela_normalized = cv2.normalize(ela_gray, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply colormap
        ela_heatmap = cv2.applyColorMap(ela_normalized, cv2.COLORMAP_JET)
        
        # Blend with original image
        overlay = cv2.addWeighted(original, 0.6, ela_heatmap, 0.4, 0)
        
        # Save overlay
        overlay_path = "debug_output/ela/ela_overlay.jpg"
        cv2.imwrite(overlay_path, overlay)
        
        return overlay_path 