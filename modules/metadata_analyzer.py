import exifread
import os
import magic
import hashlib
from datetime import datetime
import cv2
import json

class MetadataAnalyzer:
    def __init__(self):
        self.mime = magic.Magic(mime=True)
        
    def analyze_file(self, file_path):
        """Analyze file metadata and return comprehensive analysis."""
        try:
            # Basic file information
            file_info = self._get_file_info(file_path)
            
            # Get file type specific metadata
            mime_type = self.mime.from_file(file_path)
            
            if 'image' in mime_type:
                metadata = self._analyze_image(file_path)
            elif 'video' in mime_type:
                metadata = self._analyze_video(file_path)
            else:
                metadata = {}
            
            return {
                'file_info': file_info,
                'metadata': metadata,
                'mime_type': mime_type,
                'analysis_complete': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'analysis_complete': False
            }
    
    def _get_file_info(self, file_path):
        """Get basic file information."""
        stat = os.stat(file_path)
        
        # Calculate file hash
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        
        return {
            'file_size': stat.st_size,
            'created_time': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified_time': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'accessed_time': datetime.fromtimestamp(stat.st_atime).isoformat(),
            'sha256': sha256_hash.hexdigest()
        }
    
    def _analyze_image(self, image_path):
        """Extract and analyze image metadata."""
        metadata = {}
        
        # Extract EXIF data
        with open(image_path, 'rb') as image_file:
            tags = exifread.process_file(image_file)
            
            # Convert EXIF tags to serializable format
            for tag, value in tags.items():
                metadata[tag] = str(value)
        
        # Extract image properties using OpenCV
        try:
            img = cv2.imread(image_path)
            if img is not None:
                height, width, channels = img.shape
                metadata.update({
                    'dimensions': {
                        'width': width,
                        'height': height,
                        'channels': channels
                    }
                })
        except Exception as e:
            metadata['cv2_error'] = str(e)
        
        return metadata
    
    def _analyze_video(self, video_path):
        """Extract and analyze video metadata."""
        metadata = {}
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            metadata.update({
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'duration': float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
            })
            
            cap.release()
        except Exception as e:
            metadata['video_analysis_error'] = str(e)
        
        return metadata
    
    def get_metadata_summary(self, file_path):
        """Get a human-readable summary of the file's metadata."""
        analysis = self.analyze_file(file_path)
        
        if not analysis['analysis_complete']:
            return f"Error analyzing file: {analysis.get('error', 'Unknown error')}"
        
        summary = []
        summary.append(f"File Type: {analysis['mime_type']}")
        summary.append(f"File Size: {analysis['file_info']['file_size']} bytes")
        summary.append(f"Last Modified: {analysis['file_info']['modified_time']}")
        
        if 'metadata' in analysis:
            if 'dimensions' in analysis['metadata']:
                dims = analysis['metadata']['dimensions']
                summary.append(f"Dimensions: {dims['width']}x{dims['height']}")
            
            if 'duration' in analysis['metadata']:
                summary.append(f"Duration: {analysis['metadata']['duration']:.2f} seconds")
        
        return '\n'.join(summary) 