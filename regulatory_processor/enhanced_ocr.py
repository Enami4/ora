"""
Enhanced OCR extraction module with advanced preprocessing and accuracy improvements.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from typing import Dict, List, Tuple, Optional, Any
import logging
import re
from dataclasses import dataclass
from .improvements import OCRImprovements

logger = logging.getLogger(__name__)


@dataclass
class OCRConfig:
    """Configuration for enhanced OCR processing."""
    target_dpi: int = 300
    languages: str = 'fra+eng'
    tesseract_config: str = '--oem 3 --psm 6 -c preserve_interword_spaces=1'
    confidence_threshold: float = 0.6
    use_preprocessing: bool = True
    use_postprocessing: bool = True
    enable_table_detection: bool = True
    enable_structure_analysis: bool = True


class EnhancedOCRProcessor:
    """Enhanced OCR processor with advanced image preprocessing and text correction."""
    
    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()
        self.improvements = OCRImprovements()
        self.corrections = self.improvements.get_regulatory_ocr_corrections()
        
    def process_image(self, image: Image.Image) -> Tuple[str, float]:
        """Process image with enhanced OCR pipeline."""
        try:
            # Step 1: Preprocess image
            if self.config.use_preprocessing:
                image = self._preprocess_image(image)
            
            # Step 2: Extract text with confidence
            text, confidence = self._extract_text_with_confidence(image)
            
            # Step 3: Post-process text
            if self.config.use_postprocessing:
                text = self._postprocess_text(text)
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return "", 0.0
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply comprehensive image preprocessing for better OCR accuracy."""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Convert to numpy array for OpenCV processing
            img_array = np.array(image)
            
            # Apply advanced preprocessing
            img_array = self._apply_opencv_preprocessing(img_array)
            
            # Convert back to PIL Image
            processed_image = Image.fromarray(img_array)
            
            # Apply PIL-based enhancements
            processed_image = self._apply_pil_enhancements(processed_image)
            
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}")
            return image
    
    def _apply_opencv_preprocessing(self, img: np.ndarray) -> np.ndarray:
        """Apply OpenCV-based image preprocessing."""
        try:
            # Noise reduction
            img = cv2.bilateralFilter(img, 9, 75, 75)
            
            # Adaptive thresholding for better binarization
            img = cv2.adaptiveThreshold(
                img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up
            kernel = np.ones((1, 1), np.uint8)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            
            # Deskewing
            img = self._deskew_image(img)
            
            return img
            
        except Exception as e:
            logger.warning(f"OpenCV preprocessing failed: {e}")
            return img
    
    def _apply_pil_enhancements(self, image: Image.Image) -> Image.Image:
        """Apply PIL-based image enhancements."""
        try:
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Brightness adjustment
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.1)
            
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.2)
            
            # Noise reduction
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
            return image
            
        except Exception as e:
            logger.warning(f"PIL enhancement failed: {e}")
            return image
    
    def _deskew_image(self, img: np.ndarray) -> np.ndarray:
        """Deskew image to improve OCR accuracy."""
        try:
            # Find skew angle
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi
                    angles.append(angle)
                
                # Calculate median angle
                if angles:
                    median_angle = np.median(angles)
                    skew_angle = median_angle - 90
                    
                    # Only deskew if angle is significant
                    if abs(skew_angle) > 0.5:
                        # Rotate image
                        height, width = img.shape[:2]
                        center = (width // 2, height // 2)
                        rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                        img = cv2.warpAffine(img, rotation_matrix, (width, height), 
                                           flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            
            return img
            
        except Exception as e:
            logger.warning(f"Deskewing failed: {e}")
            return img
    
    def _extract_text_with_confidence(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text with confidence scoring."""
        try:
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(
                image, 
                lang=self.config.languages,
                config=self.config.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter by confidence
            confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
            texts = [text for i, text in enumerate(ocr_data['text']) 
                    if int(ocr_data['conf'][i]) >= self.config.confidence_threshold * 100]
            
            # Combine text
            filtered_text = ' '.join(texts)
            
            # Calculate overall confidence
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
            
            # Fallback to standard extraction if confidence is too low
            if avg_confidence < self.config.confidence_threshold:
                fallback_text = pytesseract.image_to_string(
                    image, 
                    lang=self.config.languages,
                    config=self.config.tesseract_config
                )
                return fallback_text, avg_confidence
            
            return filtered_text, avg_confidence
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return "", 0.0
    
    def _postprocess_text(self, text: str) -> str:
        """Apply comprehensive text post-processing."""
        try:
            # Apply character corrections
            for pattern, replacement in self.corrections['character_corrections'].items():
                text = re.sub(pattern, replacement, text)
            
            # Apply terminology corrections
            for pattern, replacement in self.corrections['terminology_corrections'].items():
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            
            # Apply structural corrections
            for pattern, replacement in self.corrections['structural_corrections'].items():
                text = re.sub(pattern, replacement, text)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            # Apply context-aware corrections
            text = self._apply_context_corrections(text)
            
            return text
            
        except Exception as e:
            logger.warning(f"Text post-processing failed: {e}")
            return text
    
    def _apply_context_corrections(self, text: str) -> str:
        """Apply context-aware corrections based on regulatory document patterns."""
        try:
            # Fix common regulatory text patterns
            patterns = [
                # Article numbering
                (r'Article\s+(\d+)\s*[\.:]?\s*([A-Z])', r'Article \1. \2'),
                # Legal structure
                (r'TITRE\s+([IVX]+)\s*[\.:]?\s*([A-Z])', r'TITRE \1 - \2'),
                (r'CHAPITRE\s+([IVX]+)\s*[\.:]?\s*([A-Z])', r'CHAPITRE \1 - \2'),
                # Regulatory references
                (r'COBAC\s+([RID][-\s]*\d+[-\s]*\d+)', r'COBAC \1'),
                (r'CEMAC\s+([RID][-\s]*\d+[-\s]*\d+)', r'CEMAC \1'),
                # Dates
                (r'(\d{1,2})\s*[/\-\.]\s*(\d{1,2})\s*[/\-\.]\s*(\d{4})', r'\1/\2/\3'),
                # Percentages
                (r'(\d+)\s*%', r'\1%'),
                # Currency
                (r'(\d+)\s*(FCFA|EUR|USD)', r'\1 \2'),
            ]
            
            for pattern, replacement in patterns:
                text = re.sub(pattern, replacement, text, flags=re.MULTILINE)
            
            return text
            
        except Exception as e:
            logger.warning(f"Context corrections failed: {e}")
            return text
    
    def detect_document_structure(self, image: Image.Image) -> Dict[str, Any]:
        """Detect document structure for layout-aware processing."""
        try:
            # Convert to numpy array
            img_array = np.array(image.convert('L'))
            
            # Detect text regions
            text_regions = self._detect_text_regions(img_array)
            
            # Detect table structures
            table_regions = self._detect_table_regions(img_array)
            
            # Analyze layout
            layout_analysis = self._analyze_layout(img_array, text_regions, table_regions)
            
            return {
                'text_regions': text_regions,
                'table_regions': table_regions,
                'layout_analysis': layout_analysis
            }
            
        except Exception as e:
            logger.warning(f"Structure detection failed: {e}")
            return {}
    
    def _detect_text_regions(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions in the image."""
        try:
            # Apply morphological operations to find text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
            dilated = cv2.dilate(img, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 50 and h > 20:  # Filter small regions
                    regions.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'area': w * h,
                        'aspect_ratio': w / h
                    })
            
            return regions
            
        except Exception as e:
            logger.warning(f"Text region detection failed: {e}")
            return []
    
    def _detect_table_regions(self, img: np.ndarray) -> List[Dict[str, Any]]:
        """Detect table regions in the image."""
        try:
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find table contours
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            tables = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 100 and h > 100:  # Filter small regions
                    tables.append({
                        'x': x, 'y': y, 'width': w, 'height': h,
                        'area': w * h
                    })
            
            return tables
            
        except Exception as e:
            logger.warning(f"Table detection failed: {e}")
            return []
    
    def _analyze_layout(self, img: np.ndarray, text_regions: List[Dict], 
                       table_regions: List[Dict]) -> Dict[str, Any]:
        """Analyze document layout."""
        try:
            height, width = img.shape[:2]
            
            # Calculate region statistics
            total_text_area = sum(region['area'] for region in text_regions)
            total_table_area = sum(region['area'] for region in table_regions)
            
            # Determine document type
            document_type = 'text'
            if total_table_area > total_text_area * 0.3:
                document_type = 'mixed'
            elif total_table_area > total_text_area:
                document_type = 'table'
            
            # Analyze reading order
            reading_order = sorted(text_regions, key=lambda r: (r['y'], r['x']))
            
            return {
                'document_type': document_type,
                'total_text_area': total_text_area,
                'total_table_area': total_table_area,
                'text_density': total_text_area / (width * height),
                'reading_order': reading_order,
                'layout_complexity': len(text_regions) + len(table_regions)
            }
            
        except Exception as e:
            logger.warning(f"Layout analysis failed: {e}")
            return {}
    
    def validate_ocr_quality(self, text: str, confidence: float) -> Dict[str, Any]:
        """Validate OCR quality using multiple metrics."""
        try:
            validation_rules = self.improvements.get_ocr_validation_rules()
            
            # Check minimum confidence
            confidence_score = confidence >= validation_rules['minimum_confidence']
            
            # Check for regulatory keywords
            regulatory_keywords = validation_rules['regulatory_keywords']
            found_keywords = sum(1 for keyword in regulatory_keywords 
                               if keyword.lower() in text.lower())
            keyword_score = found_keywords / len(regulatory_keywords)
            
            # Check text quality indicators
            quality_indicators = validation_rules['quality_indicators']
            quality_score = 0
            
            # Has articles
            if re.search(r'article\s+\d+', text.lower()):
                quality_score += quality_indicators['has_articles']
            
            # Has structure
            if re.search(r'(titre|chapitre|section)', text.lower()):
                quality_score += quality_indicators['has_structure']
            
            # Has regulatory terms
            if found_keywords > 0:
                quality_score += quality_indicators['has_regulatory_terms']
            
            # Proper formatting
            if re.search(r'\d+[°\.]', text):
                quality_score += quality_indicators['proper_formatting']
            
            # Calculate overall quality
            overall_quality = (confidence * 0.4 + keyword_score * 0.3 + 
                             (quality_score / 30) * 0.3)
            
            return {
                'confidence_score': confidence,
                'keyword_score': keyword_score,
                'quality_score': quality_score,
                'overall_quality': overall_quality,
                'is_valid': overall_quality >= 0.6,
                'found_keywords': found_keywords,
                'recommendations': self._get_quality_recommendations(overall_quality)
            }
            
        except Exception as e:
            logger.warning(f"Quality validation failed: {e}")
            return {'is_valid': False, 'overall_quality': 0.0}
    
    def _get_quality_recommendations(self, quality_score: float) -> List[str]:
        """Get quality improvement recommendations."""
        recommendations = []
        
        if quality_score < 0.5:
            recommendations.extend([
                "Consider using higher DPI settings",
                "Apply additional image preprocessing",
                "Try alternative OCR engines",
                "Manual review recommended"
            ])
        elif quality_score < 0.7:
            recommendations.extend([
                "Fine-tune OCR parameters",
                "Apply post-processing corrections",
                "Verify key regulatory terms"
            ])
        elif quality_score < 0.8:
            recommendations.extend([
                "Minor corrections may be needed",
                "Validate article numbering"
            ])
        else:
            recommendations.append("Quality is acceptable")
        
        return recommendations