#!/usr/bin/env python3
"""
Advanced OCR Engine for comprehensive document analysis.
Optimized for regulatory documents with advanced preprocessing and language support.
"""

import os
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import cv2
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract
from pdf2image import convert_from_path

logger = logging.getLogger(__name__)

class AdvancedOCREngine:
    """Advanced OCR engine with sophisticated preprocessing and analysis."""
    
    def __init__(self):
        self.tesseract_paths = [
            r"C:\Users\doupa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",  # Your installation
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
            "tesseract"  # System PATH
        ]
        
        self.tesseract_cmd = self._find_tesseract()
        if self.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            logger.info(f"Tesseract found at: {self.tesseract_cmd}")
        else:
            logger.warning("Tesseract not found. OCR functionality disabled.")
            
        # OCR configurations optimized for different document types
        self.ocr_configs = {
            'default': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ.,;:!?()[]{}"\'-+= ',
            'regulatory': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ.,;:!?()[]{}"\'-+=°§€%‰',
            'articles': r'--oem 3 --psm 6',
            'tables': r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,;:!?()[]{}"\'-+= |',
            'numbers': r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.,-%',
            'single_block': r'--oem 3 --psm 8',
            'sparse_text': r'--oem 3 --psm 11'
        }
        
        # Language configurations for French regulatory documents
        self.languages = 'fra+eng'  # French + English for better accuracy
        
    def _find_tesseract(self) -> Optional[str]:
        """Find Tesseract executable in common locations."""
        for path in self.tesseract_paths:
            if os.path.exists(path):
                return path
        return None
    
    def is_available(self) -> bool:
        """Check if OCR is available."""
        return self.tesseract_cmd is not None
    
    def extract_text_from_pdf(self, pdf_path: str, enhancement_level: str = 'high') -> Tuple[str, Dict[int, str], Dict[str, Any]]:
        """
        Extract text from PDF with advanced OCR processing.
        
        Args:
            pdf_path: Path to PDF file
            enhancement_level: 'low', 'medium', 'high', 'extreme'
            
        Returns:
            Tuple of (full_text, page_texts, metadata)
        """
        if not self.is_available():
            raise RuntimeError("OCR engine not available")
            
        logger.info(f"Starting advanced OCR extraction for: {pdf_path}")
        
        # Convert PDF to high-quality images
        images = self._pdf_to_images(pdf_path)
        
        full_text = ""
        page_texts = {}
        metadata = {
            'total_pages': len(images),
            'enhancement_level': enhancement_level,
            'ocr_confidence_scores': [],
            'preprocessing_methods': [],
            'detected_elements': {
                'articles': [],
                'tables': [],
                'headers': [],
                'lists': []
            }
        }
        
        for page_num, image in enumerate(images, 1):
            logger.info(f"Processing page {page_num}/{len(images)}")
            
            # Apply advanced preprocessing
            enhanced_image = self._enhance_image(image, enhancement_level)
            
            # Multi-pass OCR with different configurations
            page_text, page_confidence, detected_elements = self._multi_pass_ocr(enhanced_image)
            
            if page_text.strip():
                page_texts[page_num] = page_text
                full_text += f"\\n--- PAGE {page_num} ---\\n{page_text}\\n\\n"
                
                metadata['ocr_confidence_scores'].append({
                    'page': page_num,
                    'confidence': page_confidence
                })
                
                # Merge detected elements
                for element_type, elements in detected_elements.items():
                    metadata['detected_elements'][element_type].extend(elements)
        
        logger.info(f"OCR extraction completed. {len(page_texts)} pages processed.")
        
        return full_text.strip(), page_texts, metadata
    
    def _pdf_to_images(self, pdf_path: str) -> List[Image.Image]:
        """Convert PDF to high-quality images."""
        try:
            # Try with poppler first
            return convert_from_path(
                pdf_path,
                dpi=300,  # High DPI for better OCR accuracy
                fmt='PNG',
                thread_count=4,
                grayscale=False  # Keep color for better preprocessing
            )
        except Exception as e:
            logger.warning(f"pdf2image failed (Poppler not installed?): {e}")
            
            # Fallback to pypdfium2
            try:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(pdf_path)
                images = []
                for i in range(len(pdf)):
                    page = pdf[i]
                    # Render at 300 DPI (scale = 300/72)
                    pil_image = page.render(scale=300/72).to_pil()
                    images.append(pil_image)
                    page.close()
                pdf.close()
                return images
            except Exception as e2:
                logger.error(f"Failed to convert PDF to images: {e2}")
                raise RuntimeError(f"Cannot convert PDF to images: {e2}")
    
    def _enhance_image(self, image: Image.Image, level: str = 'high') -> Image.Image:
        """Apply advanced image preprocessing for better OCR accuracy."""
        enhanced = image.copy()
        
        if level in ['medium', 'high', 'extreme']:
            # Convert to grayscale for processing
            enhanced = enhanced.convert('L')
            
            # Noise reduction
            enhanced = enhanced.filter(ImageFilter.MedianFilter(size=3))
            
            # Contrast enhancement
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)
            
            # Sharpness enhancement
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(1.2)
        
        if level in ['high', 'extreme']:
            # Advanced preprocessing with OpenCV
            cv_image = np.array(enhanced)
            
            # Gaussian blur to reduce noise
            cv_image = cv2.GaussianBlur(cv_image, (1, 1), 0)
            
            # Morphological operations to clean up text
            kernel = np.ones((1, 1), np.uint8)
            cv_image = cv2.morphologyEx(cv_image, cv2.MORPH_CLOSE, kernel)
            
            # Adaptive thresholding for better text separation
            cv_image = cv2.adaptiveThreshold(
                cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            
            enhanced = Image.fromarray(cv_image)
        
        if level == 'extreme':
            # Dilation and erosion for extreme cases
            cv_image = np.array(enhanced)
            kernel = np.ones((2, 2), np.uint8)
            cv_image = cv2.dilate(cv_image, kernel, iterations=1)
            cv_image = cv2.erode(cv_image, kernel, iterations=1)
            enhanced = Image.fromarray(cv_image)
        
        return enhanced
    
    def _multi_pass_ocr(self, image: Image.Image) -> Tuple[str, float, Dict[str, List]]:
        """Perform multi-pass OCR with different configurations for comprehensive extraction."""
        results = []
        detected_elements = {
            'articles': [],
            'tables': [],
            'headers': [],
            'lists': []
        }
        
        # Pass 1: General regulatory document extraction
        try:
            text = pytesseract.image_to_string(
                image, 
                lang=self.languages, 
                config=self.ocr_configs['regulatory']
            )
            confidence = self._calculate_confidence(image, text)
            results.append((text, confidence, 'regulatory'))
        except Exception as e:
            logger.warning(f"Regulatory OCR pass failed: {e}")
        
        # Pass 2: Article-specific extraction
        try:
            text = pytesseract.image_to_string(
                image, 
                lang=self.languages, 
                config=self.ocr_configs['articles']
            )
            confidence = self._calculate_confidence(image, text)
            results.append((text, confidence, 'articles'))
            
            # Detect articles in this pass
            detected_elements['articles'] = self._detect_articles(text)
        except Exception as e:
            logger.warning(f"Articles OCR pass failed: {e}")
        
        # Pass 3: Table extraction
        try:
            text = pytesseract.image_to_string(
                image, 
                lang=self.languages, 
                config=self.ocr_configs['tables']
            )
            if self._contains_tabular_data(text):
                detected_elements['tables'] = self._extract_table_data(text)
        except Exception as e:
            logger.warning(f"Table OCR pass failed: {e}")
        
        # Select best result based on confidence and content quality
        if results:
            best_text, best_confidence, method = max(results, key=lambda x: x[1])
            
            # Post-process the text
            processed_text = self._post_process_text(best_text)
            
            return processed_text, best_confidence, detected_elements
        
        return "", 0.0, detected_elements
    
    def _calculate_confidence(self, image: Image.Image, text: str) -> float:
        """Calculate OCR confidence score."""
        try:
            # Use Tesseract's built-in confidence scoring
            data = pytesseract.image_to_data(image, lang=self.languages, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                
                # Adjust confidence based on text quality indicators
                if len(text.strip()) < 10:
                    avg_confidence *= 0.5  # Penalize very short text
                if text.count('?') > len(text) * 0.1:  # Too many uncertain characters
                    avg_confidence *= 0.7
                if len([c for c in text if c.isalnum()]) / max(len(text), 1) < 0.5:
                    avg_confidence *= 0.8  # Too many non-alphanumeric characters
                
                return min(avg_confidence, 100.0)
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
        
        # Fallback: basic text quality assessment
        if not text.strip():
            return 0.0
        
        # Simple quality metrics
        alphanumeric_ratio = len([c for c in text if c.isalnum()]) / len(text)
        word_count = len(text.split())
        
        base_score = 50.0
        if alphanumeric_ratio > 0.7:
            base_score += 20
        if word_count > 10:
            base_score += 15
        if 'article' in text.lower() or 'règlement' in text.lower():
            base_score += 10
        
        return min(base_score, 100.0)
    
    def _detect_articles(self, text: str) -> List[Dict[str, Any]]:
        """Detect article structures in text."""
        import re
        
        articles = []
        
        # Enhanced article detection patterns
        patterns = [
            r'(Article\\s+\\d+(?:\\.\\d+)?(?:\\s*[a-z])?(?:\\s*bis|\\s*ter|\\s*quater)?)[^\\n]*\\n([^\\n]*(?:\\n(?!Article\\s+\\d+)[^\\n]*)*)',
            r'(ARTICLE\\s+\\d+(?:\\.\\d+)?(?:\\s*[a-z])?(?:\\s*bis|\\s*ter|\\s*quater)?)[^\\n]*\\n([^\\n]*(?:\\n(?!ARTICLE\\s+\\d+)[^\\n]*)*)',
            r'(Art\\.\\s*\\d+(?:\\.\\d+)?(?:\\s*[a-z])?(?:\\s*bis|\\s*ter|\\s*quater)?)[^\\n]*\\n([^\\n]*(?:\\n(?!Art\\.\\s*\\d+)[^\\n]*)*)'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                articles.append({
                    'number': match.group(1).strip(),
                    'content': match.group(2).strip()[:500],  # Limit content length
                    'position': match.start()
                })
        
        return articles
    
    def _contains_tabular_data(self, text: str) -> bool:
        """Check if text contains tabular data."""
        lines = text.split('\\n')
        
        # Look for patterns indicating tables
        pipe_count = text.count('|')
        tab_count = text.count('\\t')
        
        # Count lines with multiple columns (spaces or tabs)
        multi_column_lines = 0
        for line in lines:
            if len(line.split()) > 3 and ('  ' in line or '\\t' in line):
                multi_column_lines += 1
        
        return (pipe_count > 5 or tab_count > 3 or 
                multi_column_lines > len(lines) * 0.3)
    
    def _extract_table_data(self, text: str) -> List[Dict[str, Any]]:
        """Extract structured table data."""
        tables = []
        lines = text.split('\\n')
        
        current_table = []
        in_table = False
        
        for line in lines:
            if self._is_table_row(line):
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line.split())
            else:
                if in_table and current_table:
                    tables.append({
                        'type': 'table',
                        'rows': current_table,
                        'row_count': len(current_table),
                        'col_count': max(len(row) for row in current_table) if current_table else 0
                    })
                    current_table = []
                in_table = False
        
        return tables
    
    def _is_table_row(self, line: str) -> bool:
        """Check if a line appears to be part of a table."""
        if not line.strip():
            return False
        
        # Look for table indicators
        parts = line.split()
        return (len(parts) > 2 and 
                ('|' in line or '\\t' in line or 
                 len([p for p in parts if p.replace('.', '').replace(',', '').isdigit()]) > 1))
    
    def _post_process_text(self, text: str) -> str:
        """Post-process OCR text for better quality."""
        import re
        
        # Fix common OCR errors in French regulatory documents
        corrections = {
            r'\\bl\\b': 'l',  # Common OCR error
            r'\\bO\\b': '0',  # Letter O mistaken for zero
            r'rn': 'm',       # rn mistaken for m
            r'\\s+': ' ',     # Multiple spaces to single space
            r'([.!?])([A-Z])': r'\\1 \\2',  # Missing space after punctuation
        }
        
        processed = text
        for pattern, replacement in corrections.items():
            processed = re.sub(pattern, replacement, processed)
        
        # Clean up extra whitespace
        processed = re.sub(r'\\n\\s*\\n\\s*\\n', '\\n\\n', processed)  # Max 2 consecutive newlines
        processed = processed.strip()
        
        return processed
    
    def get_tesseract_info(self) -> Dict[str, Any]:
        """Get information about Tesseract installation."""
        if not self.is_available():
            return {'available': False, 'error': 'Tesseract not found'}
        
        try:
            version = pytesseract.get_tesseract_version()
            languages = pytesseract.get_languages()
            
            return {
                'available': True,
                'version': str(version),
                'languages': languages,
                'executable_path': self.tesseract_cmd,
                'supported_formats': ['PNG', 'JPEG', 'TIFF', 'BMP', 'GIF'],
                'ocr_modes': list(self.ocr_configs.keys())
            }
        except Exception as e:
            return {'available': False, 'error': str(e)}

# Test function
def test_ocr_engine():
    """Test the OCR engine functionality."""
    engine = AdvancedOCREngine()
    info = engine.get_tesseract_info()
    
    print("=== OCR Engine Test ===")
    print(f"Available: {info['available']}")
    if info['available']:
        print(f"Version: {info['version']}")
        print(f"Languages: {info['languages']}")
        print(f"Path: {info['executable_path']}")
    else:
        print(f"Error: {info['error']}")

if __name__ == "__main__":
    test_ocr_engine()