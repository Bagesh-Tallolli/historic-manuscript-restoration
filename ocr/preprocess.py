"""
Image preprocessing utilities for OCR
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance


class OCRPreprocessor:
    """Preprocessing pipeline for Sanskrit manuscript OCR"""
    
    def __init__(self):
        pass
    
    def preprocess(self, image, apply_all=True):
        """
        Apply complete preprocessing pipeline

        Args:
            image: Input image (numpy array or PIL Image)
            apply_all: If True, applies all preprocessing steps

        Returns:
            Preprocessed image ready for OCR
        """
        # Convert to numpy if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Apply preprocessing steps
        if apply_all:
            image = self.grayscale(image)
            image = self.denoise(image)
            image = self.deskew(image)
            image = self.binarize(image)
            image = self.remove_borders(image)
        
        return image
    
    def grayscale(self, image):
        """Convert to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return image
    
    def denoise(self, image):
        """Remove noise from image"""
        # Non-local means denoising works better for documents
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    def binarize(self, image, method='adaptive'):
        """
        Convert to binary (black and white)

        Args:
            image: Grayscale image
            method: 'otsu', 'adaptive', or 'sauvola'
        """
        if method == 'otsu':
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
        
        elif method == 'sauvola':
            binary = self._sauvola_threshold(image)
        
        else:
            # Default to Otsu
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _sauvola_threshold(self, image, window_size=15, k=0.2, r=128):
        """
        Sauvola binarization - works well for degraded documents

        Args:
            image: Grayscale image
            window_size: Local window size
            k: Parameter (typically 0.2-0.5)
            r: Dynamic range of standard deviation
        """
        # Compute local mean and standard deviation
        mean = cv2.boxFilter(image.astype(np.float32), -1, (window_size, window_size))
        sqr_mean = cv2.boxFilter(image.astype(np.float32)**2, -1, (window_size, window_size))
        std = np.sqrt(sqr_mean - mean**2)
        
        # Sauvola threshold
        threshold = mean * (1 + k * ((std / r) - 1))
        
        binary = np.where(image > threshold, 255, 0).astype(np.uint8)
        return binary
    
    def deskew(self, image):
        """
        Correct skew/rotation in the image
        """
        # Find all non-zero points
        coords = np.column_stack(np.where(image > 0))
        
        if len(coords) == 0:
            return image
        
        # Find minimum area rectangle
        angle = cv2.minAreaRect(coords)[-1]
        
        # Correct angle
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Rotate image
        if abs(angle) > 0.5:  # Only rotate if angle is significant
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(
                image, M, (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE
            )
            return rotated
        
        return image
    
    def remove_borders(self, image, threshold=200):
        """Remove border artifacts"""
        # Find contours
        contours, _ = cv2.findContours(
            image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return image
        
        # Get bounding box of largest contour (assumed to be text)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add small padding
        padding = 10
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        # Crop to content
        cropped = image[y:y+h, x:x+w]
        
        return cropped
    
    def enhance_contrast(self, image, clip_limit=2.0):
        """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def line_segmentation(self, image):
        """
        Segment image into individual text lines

        Returns:
            List of line images
        """
        # Horizontal projection
        horizontal_projection = np.sum(image == 0, axis=1)
        
        # Find line boundaries (where projection drops to near zero)
        threshold = np.max(horizontal_projection) * 0.1
        line_start = []
        line_end = []
        
        in_line = False
        for i, val in enumerate(horizontal_projection):
            if val > threshold and not in_line:
                line_start.append(i)
                in_line = True
            elif val <= threshold and in_line:
                line_end.append(i)
                in_line = False
        
        # Extract line images
        lines = []
        for start, end in zip(line_start, line_end):
            if end - start > 5:  # Minimum line height
                line_img = image[start:end, :]
                lines.append(line_img)
        
        return lines
    
    def word_segmentation(self, line_image):
        """
        Segment a line into individual words

        Returns:
            List of word images
        """
        # Vertical projection
        vertical_projection = np.sum(line_image == 0, axis=0)

        # Find word boundaries
        threshold = np.max(vertical_projection) * 0.1
        word_start = []
        word_end = []

        in_word = False
        for i, val in enumerate(vertical_projection):
            if val > threshold and not in_word:
                word_start.append(i)
                in_word = True
            elif val <= threshold and in_word:
                word_end.append(i)
                in_word = False

        # Extract word images
        words = []
        for start, end in zip(word_start, word_end):
            if end - start > 3:  # Minimum word width
                word_img = line_image[:, start:end]
                words.append(word_img)

        return words


def resize_with_aspect_ratio(image, max_width=2000, max_height=2000):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(max_width / w, max_height / h)
    
    if scale < 1:
        new_w = int(w * scale)
        new_h = int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return image


if __name__ == "__main__":
    # Test preprocessing
    print("Testing OCR preprocessing...")
    
    # Create a sample image
    test_img = np.ones((500, 500, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "रामः वनं गच्छति", (50, 250), 
                cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 3)
    
    preprocessor = OCRPreprocessor()
    processed = preprocessor.preprocess(test_img)
    
    print(f"Original shape: {test_img.shape}")
    print(f"Processed shape: {processed.shape}")
    print("Preprocessing successful!")
