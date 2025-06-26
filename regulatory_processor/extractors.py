"""
PDF text extraction module with multiple extraction methods for robustness.
"""

import os
import logging
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import PyPDF2
import pdfplumber
from pathlib import Path
import chardet

# Optional advanced extraction libraries
try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False
    
try:
    import pytesseract
    from pdf2image import convert_from_path
    # Check if Tesseract executable is available
    try:
        pytesseract.get_tesseract_version()
        HAS_OCR = True
    except:
        HAS_OCR = False
except ImportError:
    HAS_OCR = False

try:
    import tabula
    HAS_TABULA = True
except ImportError:
    HAS_TABULA = False

logger = logging.getLogger(__name__)

# Log the status after logger is initialized
if not HAS_OCR:
    logger.warning("OCR functionality disabled (Tesseract not found)")
if not HAS_PDFIUM:
    logger.info("pypdfium2 not available, using fallback extraction methods")
if not HAS_TABULA:
    logger.info("tabula-py not available, table extraction disabled")


class PDFExtractor:
    """Handles PDF text extraction with multiple fallback methods."""
    
    def __init__(self):
        self.extraction_methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2,
        ]
        
        # Add advanced methods if available
        if HAS_PDFIUM:
            self.extraction_methods.insert(0, self._extract_with_pdfium)
            
        if HAS_OCR:
            self.extraction_methods.append(self._extract_with_ocr)
            
        logger.info(f"PDFExtractor initialized with {len(self.extraction_methods)} extraction methods")
    
    def extract_text(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """
        Extract text from PDF using multiple methods.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (full_text, page_texts_dict)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        full_text = ""
        page_texts = {}
        
        for method in self.extraction_methods:
            try:
                full_text, page_texts = method(pdf_path)
                if full_text.strip():
                    logger.info(f"Successfully extracted text using {method.__name__}")
                    break
            except Exception as e:
                logger.warning(f"Extraction method {method.__name__} failed: {e}")
                continue
        
        if not full_text.strip():
            logger.error(f"All extraction methods failed for {pdf_path}")
        
        return full_text, page_texts
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """Extract text using pdfplumber."""
        full_text = ""
        page_texts = {}
        
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    page_texts[i + 1] = page_text
                    full_text += page_text + "\n\n"
        
        return full_text.strip(), page_texts
    
    def _extract_with_pypdf2(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """Extract text using PyPDF2."""
        full_text = ""
        page_texts = {}
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    page_texts[i + 1] = page_text
                    full_text += page_text + "\n\n"
        
        return full_text.strip(), page_texts
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, any]:
        """Extract metadata from PDF file."""
        metadata = {
            'file_path': pdf_path,
            'file_name': os.path.basename(pdf_path),
            'file_size': os.path.getsize(pdf_path),
            'extraction_date': datetime.now().isoformat(),
            'page_count': 0,
            'title': '',
            'author': '',
            'subject': '',
            'creator': '',
            'producer': '',
            'creation_date': None,
            'modification_date': None,
            'pdf_version': ''
        }
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                metadata['page_count'] = len(pdf_reader.pages)
                
                if hasattr(pdf_reader, 'pdf_header'):
                    metadata['pdf_version'] = pdf_reader.pdf_header
                
                if pdf_reader.metadata:
                    metadata['title'] = self._safe_get_metadata(pdf_reader.metadata, '/Title')
                    metadata['author'] = self._safe_get_metadata(pdf_reader.metadata, '/Author')
                    metadata['subject'] = self._safe_get_metadata(pdf_reader.metadata, '/Subject')
                    metadata['creator'] = self._safe_get_metadata(pdf_reader.metadata, '/Creator')
                    metadata['producer'] = self._safe_get_metadata(pdf_reader.metadata, '/Producer')
                    metadata['creation_date'] = self._safe_get_metadata(pdf_reader.metadata, '/CreationDate')
                    metadata['modification_date'] = self._safe_get_metadata(pdf_reader.metadata, '/ModDate')
        
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
        
        return metadata
    
    def _safe_get_metadata(self, metadata_obj, key: str) -> str:
        """Safely extract metadata value."""
        try:
            value = metadata_obj.get(key, '')
            if value:
                return str(value)
            return ''
        except:
            return ''
    
    def validate_extraction(self, text: str, min_length: int = 100) -> bool:
        """
        Validate that extracted text meets minimum requirements.
        
        Args:
            text: Extracted text to validate
            min_length: Minimum required text length
            
        Returns:
            Boolean indicating if text is valid
        """
        if not text or len(text.strip()) < min_length:
            return False
        
        word_count = len(text.split())
        if word_count < 10:
            return False
        
        return True
    
    def _extract_with_pdfium(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """Extract text using pypdfium2 for better character encoding."""
        full_text = ""
        page_texts = {}
        
        pdf = pdfium.PdfDocument(pdf_path)
        for i in range(len(pdf)):
            page = pdf[i]
            textpage = page.get_textpage()
            page_text = textpage.get_text_range()
            
            if page_text:
                page_texts[i + 1] = page_text
                full_text += page_text + "\n\n"
            
            textpage.close()
            page.close()
        
        pdf.close()
        return full_text.strip(), page_texts
    
    def _extract_with_ocr(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """Extract text using OCR for scanned PDFs."""
        full_text = ""
        page_texts = {}
        
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)
        
        for i, image in enumerate(images):
            # Perform OCR on each page
            page_text = pytesseract.image_to_string(image, lang='fra')  # French language
            
            if page_text.strip():
                page_texts[i + 1] = page_text
                full_text += page_text + "\n\n"
        
        return full_text.strip(), page_texts
    
    def detect_encoding(self, pdf_path: str) -> str:
        """Detect the character encoding of the PDF."""
        try:
            with open(pdf_path, 'rb') as file:
                raw_data = file.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result['encoding'] or 'utf-8'
        except Exception as e:
            logger.warning(f"Could not detect encoding: {e}")
            return 'utf-8'
    
    def extract_tables(self, pdf_path: str) -> List[Dict[str, any]]:
        """Extract tables from PDF if available."""
        tables = []
        
        if HAS_TABULA:
            try:
                # Extract tables from all pages
                dfs = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True, silent=True)
                for i, df in enumerate(dfs):
                    tables.append({
                        'table_index': i,
                        'data': df.to_dict('records'),
                        'shape': df.shape
                    })
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")
        
        return tables