import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

class CNNClassifier:
    def __init__(self, model_path=None):
        """
        Initialize the CNN classifier for AI-generated image detection.
        
        Args:
            model_path: Path to a pre-trained model file (optional)
        """
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("debug_output/cnn", exist_ok=True)
        
        # Default model path
        self.default_model_path = "models/ai_detector.h5"
        
        # Input image size for the model
        self.input_size = (224, 224)
        
        # Load or create model
        if model_path and os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Loaded model from {model_path}")
        elif os.path.exists(self.default_model_path):
            self.model = load_model(self.default_model_path)
            print(f"Loaded model from {self.default_model_path}")
        else:
            print("No model found. Creating a new model.")
            self.model = self._create_model()
            # Save the model architecture for future use
            self.model.save(self.default_model_path)
    
    def _create_model(self):
        """
        Create a CNN model with transfer learning using EfficientNet.
        """
        # Load the pre-trained EfficientNetB0 model without the top classification layer
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        
        # Freeze the base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        # Add custom classification layers
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(1, activation='sigmoid')(x)  # Binary classification: real or AI-generated
        
        # Create the full model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess an image for the model.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image array
        """
        # Load and resize image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.input_size)
        
        # Convert to array and preprocess
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        return img_array
    
    def predict(self, image_path):
        """
        Predict whether an image is AI-generated or real.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess the image
            img_array = self.preprocess_image(image_path)
            
            # Make prediction
            prediction = self.model.predict(img_array)[0][0]
            
            # Generate heatmap using Grad-CAM
            heatmap_path = self._generate_gradcam(image_path, img_array)
            
            return {
                'ai_generated_probability': float(prediction),
                'is_ai_generated': prediction > 0.5,
                'confidence': float(abs(prediction - 0.5) * 2),  # Scale to 0-1
                'visualization_path': heatmap_path,
                'analysis_complete': True
            }
        except Exception as e:
            return {
                'error': str(e),
                'analysis_complete': False
            }
    
    def _generate_gradcam(self, image_path, preprocessed_img):
        """
        Generate a Grad-CAM heatmap to visualize which parts of the image influenced the prediction.
        
        Args:
            image_path: Path to the original image
            preprocessed_img: Preprocessed image array for the model
            
        Returns:
            Path to the saved visualization
        """
        # Get the last convolutional layer
        last_conv_layer = None
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        
        if not last_conv_layer:
            # If no convolutional layer found, return None
            return None
        
        # Create a model that maps the input to the last conv layer's output
        grad_model = Model(
            inputs=self.model.inputs,
            outputs=[
                self.model.get_layer(last_conv_layer).output,
                self.model.output
            ]
        )
        
        # Record operations for automatic differentiation
        with tf.GradientTape() as tape:
            # Forward pass
            conv_outputs, predictions = grad_model(preprocessed_img)
            # Target prediction (AI-generated probability)
            target_class = predictions[:, 0]
        
        # Gradient of the target with respect to the output of the last conv layer
        grads = tape.gradient(target_class, conv_outputs)
        
        # Vector of shape (batch_size, features), where features is the sum of each feature map
        pooled_grads = tf.reduce_mean(grads, axis=(1, 2))
        
        # Calculate the weighted feature maps
        heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
        heatmap = heatmap.numpy()
        
        # Resize heatmap to original image size
        img = cv2.imread(image_path)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Apply colormap
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Superimpose heatmap on original image
        superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
        
        # Save the visualization
        output_path = "debug_output/cnn/gradcam.jpg"
        cv2.imwrite(output_path, superimposed)
        
        return output_path
    
    def fine_tune(self, real_images_dir, fake_images_dir, epochs=5, batch_size=32):
        """
        Fine-tune the model on custom data.
        
        Args:
            real_images_dir: Directory containing real images
            fake_images_dir: Directory containing AI-generated images
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Training history
        """
        # This method would be implemented to fine-tune the model on custom datasets
        # For now, we'll return a placeholder message
        return {
            'message': 'Fine-tuning functionality not implemented in this version.',
            'status': 'not_implemented'
        }
    
    def evaluate_statistical_metrics(self, image_path):
        """
        Calculate additional statistical metrics for the image that can help with classification.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary of statistical metrics
        """
        try:
            # Read the image
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to different color spaces
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            
            # Calculate metrics for each channel
            metrics = {}
            
            # RGB channels
            for i, channel in enumerate(['red', 'green', 'blue']):
                channel_data = img_rgb[:,:,i].flatten()
                metrics[channel] = {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'entropy': float(self._calculate_entropy(channel_data))
                }
            
            # HSV channels
            for i, channel in enumerate(['hue', 'saturation', 'value']):
                channel_data = img_hsv[:,:,i].flatten()
                metrics[channel] = {
                    'mean': float(np.mean(channel_data)),
                    'std': float(np.std(channel_data)),
                    'entropy': float(self._calculate_entropy(channel_data))
                }
            
            # Calculate gradients
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            metrics['gradients'] = {
                'mean': float(np.mean(gradient_magnitude)),
                'std': float(np.std(gradient_magnitude)),
                'entropy': float(self._calculate_entropy(gradient_magnitude.flatten()))
            }
            
            # Calculate texture metrics using GLCM (simplified)
            texture_contrast = self._calculate_glcm_contrast(gray)
            texture_homogeneity = self._calculate_glcm_homogeneity(gray)
            
            metrics['texture'] = {
                'contrast': float(texture_contrast),
                'homogeneity': float(texture_homogeneity)
            }
            
            return metrics
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data."""
        hist, _ = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero probability values
        return -np.sum(hist * np.log2(hist))
    
    def _calculate_glcm_contrast(self, gray_img, distance=1):
        """Calculate GLCM contrast feature (simplified)."""
        # Simplified GLCM contrast calculation
        height, width = gray_img.shape
        contrast_sum = 0
        
        # Examine a subset of pixels for efficiency
        step = max(1, min(height, width) // 100)
        count = 0
        
        for i in range(0, height-distance, step):
            for j in range(0, width-distance, step):
                # Calculate contrast between current pixel and pixel at distance
                diff = float(gray_img[i, j] - gray_img[i+distance, j+distance])
                contrast_sum += diff * diff
                count += 1
        
        if count > 0:
            return contrast_sum / count
        return 0
    
    def _calculate_glcm_homogeneity(self, gray_img, distance=1):
        """Calculate GLCM homogeneity feature (simplified)."""
        # Simplified GLCM homogeneity calculation
        height, width = gray_img.shape
        homogeneity_sum = 0
        
        # Examine a subset of pixels for efficiency
        step = max(1, min(height, width) // 100)
        count = 0
        
        for i in range(0, height-distance, step):
            for j in range(0, width-distance, step):
                # Calculate homogeneity between current pixel and pixel at distance
                diff = float(gray_img[i, j] - gray_img[i+distance, j+distance])
                homogeneity_sum += 1.0 / (1.0 + diff * diff)
                count += 1
        
        if count > 0:
            return homogeneity_sum / count
        return 0 