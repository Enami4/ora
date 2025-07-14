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
from .caching import cache_result, get_cache
from .error_recovery import with_retry, with_fallback, get_error_manager
from .page_selector import PageSelection, PageRangeParser

# Import enhanced OCR components
try:
    from .enhanced_ocr import EnhancedOCRProcessor, OCRConfig
    HAS_ENHANCED_OCR = True
except ImportError:
    HAS_ENHANCED_OCR = False
    logger.warning("Enhanced OCR not available, using basic OCR")

# Optional advanced extraction libraries
try:
    import pypdfium2 as pdfium
    HAS_PDFIUM = True
except ImportError:
    HAS_PDFIUM = False
    
try:
    import pytesseract
    from pdf2image import convert_from_path
    # Configure Tesseract path
    tesseract_path = r"C:\Users\doupa\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
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
    
    def __init__(self, use_enhanced_ocr: bool = True):
        self.use_enhanced_ocr = use_enhanced_ocr and HAS_ENHANCED_OCR
        
        # Initialize enhanced OCR processor if available
        if self.use_enhanced_ocr:
            self.enhanced_ocr = EnhancedOCRProcessor(OCRConfig(
                target_dpi=300,
                languages='fra+eng',
                confidence_threshold=0.6,
                use_preprocessing=True,
                use_postprocessing=True
            ))
            logger.info("Enhanced OCR processor initialized")
        
        self.extraction_methods = [
            self._extract_with_pdfplumber,
            self._extract_with_pypdf2,
        ]
        
        # Add advanced methods if available
        if HAS_PDFIUM:
            self.extraction_methods.insert(0, self._extract_with_pdfium)
            
        if HAS_OCR or self.use_enhanced_ocr:
            self.extraction_methods.append(self._extract_with_ocr)
            
        logger.info(f"PDFExtractor initialized with {len(self.extraction_methods)} extraction methods")
    
    @cache_result("text_extraction", ttl_hours=48)
    @with_retry(max_retries=2, base_delay=1.0, retry_exceptions=(OSError, PermissionError))
    def extract_text(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """
        Extract text from PDF using multiple methods with intelligent fallback.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (full_text, page_texts_dict)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        error_manager = get_error_manager()
        
        # Try to get from cache first (handled by decorator)
        # If not cached, proceed with extraction
        
        full_text = ""
        page_texts = {}
        
        # Try standard extraction methods first
        for method in self.extraction_methods:
            if method.__name__ == '_extract_with_ocr':
                continue  # Skip OCR in first pass
                
            try:
                full_text, page_texts = method(pdf_path)
                if full_text.strip() and self.validate_extraction(full_text):
                    logger.info(f"Successfully extracted text using {method.__name__}")
                    
                    # Check if we should enhance with OCR for missing pages
                    if HAS_OCR and self._should_enhance_with_ocr(full_text, page_texts):
                        logger.info("Enhancing extraction with OCR for potentially missing content")
                        try:
                            ocr_full_text, ocr_page_texts = self._extract_with_ocr_safe(pdf_path)
                            if ocr_full_text:
                                full_text, page_texts = self._merge_extractions(
                                    full_text, page_texts, ocr_full_text, ocr_page_texts
                                )
                        except Exception as ocr_error:
                            logger.warning(f"OCR enhancement failed: {ocr_error}")
                            # Continue with standard extraction
                    
                    return full_text, page_texts
            except Exception as e:
                error_manager._record_error(e, method.__name__, pdf_path)
                logger.warning(f"Extraction method {method.__name__} failed: {e}")
                continue
        
        # If all standard methods failed or returned poor results, try OCR
        if HAS_OCR and '_extract_with_ocr' in [m.__name__ for m in self.extraction_methods]:
            logger.info("Standard extraction methods failed or returned poor results. Attempting OCR extraction...")
            try:
                full_text, page_texts = self._extract_with_ocr_safe(pdf_path)
                if full_text.strip():
                    logger.info("Successfully extracted text using OCR")
                    return full_text, page_texts
            except Exception as e:
                error_manager._record_error(e, "extract_with_ocr", pdf_path)
                logger.error(f"OCR extraction also failed: {e}")
        
        if not full_text.strip():
            logger.error(f"All extraction methods failed for {pdf_path}")
            # Return empty results instead of failing completely
            return "", {}
        
        return full_text, page_texts
    
    @cache_result("selective_text_extraction", ttl_hours=48)
    @with_retry(max_retries=2, base_delay=1.0, retry_exceptions=(OSError, PermissionError))
    def extract_text_selective(self, pdf_path: str, page_selection: PageSelection) -> Tuple[str, Dict[int, str]]:
        """
        Extract text from specific pages of a PDF.
        
        Args:
            pdf_path: Path to the PDF file
            page_selection: PageSelection object with validated page ranges
            
        Returns:
            Tuple of (full_text, page_texts_dict) with original page numbering
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting {page_selection.selected_count} pages from {pdf_path}")
        
        full_text = ""
        page_texts = {}
        
        # Try standard extraction methods first
        for method in self.extraction_methods:
            if method.__name__ == '_extract_with_ocr':
                continue  # Skip OCR in first pass
            
            try:
                # Extract only selected pages
                method_name = method.__name__.replace('_extract_with_', '') + '_selective'
                if hasattr(self, f'_{method_name}'):
                    selective_method = getattr(self, f'_{method_name}')
                    full_text, page_texts = selective_method(pdf_path, page_selection)
                else:
                    # Fallback to extracting all and filtering
                    all_text, all_pages = method(pdf_path)
                    full_text, page_texts = self._filter_pages(all_pages, page_selection)
                
                if full_text.strip() and self.validate_extraction(full_text):
                    logger.info(f"Successfully extracted selected pages using {method.__name__}")
                    
                    # Check if we should enhance with OCR for missing pages in selection
                    if HAS_OCR and self._should_enhance_with_ocr_selective(full_text, page_texts, page_selection):
                        logger.info("Enhancing selective extraction with OCR for potentially missing content")
                        try:
                            ocr_full_text, ocr_page_texts = self._extract_with_ocr_selective(pdf_path, page_selection)
                            if ocr_full_text:
                                full_text, page_texts = self._merge_extractions(
                                    full_text, page_texts, ocr_full_text, ocr_page_texts
                                )
                        except Exception as ocr_error:
                            logger.warning(f"Selective OCR enhancement failed: {ocr_error}")
                            # Continue with standard selective extraction
                    
                    return full_text, page_texts
                    
            except Exception as e:
                logger.warning(f"Selective extraction with {method.__name__} failed: {e}")
                continue
        
        # If all methods failed, try OCR on selected pages
        if HAS_OCR and '_extract_with_ocr' in [m.__name__ for m in self.extraction_methods]:
            logger.info("Attempting OCR extraction for selected pages...")
            try:
                full_text, page_texts = self._extract_with_ocr_selective(pdf_path, page_selection)
                if full_text.strip():
                    logger.info("Successfully extracted selected pages using OCR")
                    return full_text, page_texts
            except Exception as e:
                logger.error(f"OCR extraction for selected pages failed: {e}")
        
        if not full_text.strip():
            logger.error(f"All extraction methods failed for selected pages in {pdf_path}")
            return "", {}
        
        return full_text, page_texts
    
    def _pdfplumber_selective(self, pdf_path: str, page_selection: PageSelection) -> Tuple[str, Dict[int, str]]:
        """Extract specific pages using pdfplumber."""
        full_text = ""
        page_texts = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num in page_selection.selected_pages:
                    if page_num <= len(pdf.pages):
                        try:
                            page = pdf.pages[page_num - 1]  # pdfplumber uses 0-based indexing
                            page_text = page.extract_text()
                            if page_text:
                                page_texts[page_num] = page_text  # Use original page number
                                full_text += f"\n--- PAGE {page_num} ---\n{page_text}\n\n"
                        except Exception as e:
                            logger.warning(f"Failed to extract page {page_num}: {e}")
                            continue
        except Exception as e:
            logger.error(f"pdfplumber selective extraction failed: {e}")
            raise
        
        return full_text.strip(), page_texts
    
    def _pypdf2_selective(self, pdf_path: str, page_selection: PageSelection) -> Tuple[str, Dict[int, str]]:
        """Extract specific pages using PyPDF2."""
        full_text = ""
        page_texts = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in page_selection.selected_pages:
                    if page_num <= len(pdf_reader.pages):
                        try:
                            page = pdf_reader.pages[page_num - 1]  # PyPDF2 uses 0-based indexing
                            page_text = page.extract_text()
                            if page_text:
                                page_texts[page_num] = page_text  # Use original page number
                                full_text += f"\n--- PAGE {page_num} ---\n{page_text}\n\n"
                        except Exception as e:
                            logger.warning(f"Failed to extract page {page_num} with PyPDF2: {e}")
                            continue
        except Exception as e:
            logger.error(f"PyPDF2 selective extraction failed: {e}")
            raise
        
        return full_text.strip(), page_texts
    
    def _pdfium_selective(self, pdf_path: str, page_selection: PageSelection) -> Tuple[str, Dict[int, str]]:
        """Extract specific pages using pypdfium2."""
        full_text = ""
        page_texts = {}
        
        pdf = pdfium.PdfDocument(pdf_path)
        for page_num in page_selection.selected_pages:
            if page_num <= len(pdf):
                try:
                    page = pdf[page_num - 1]  # pypdfium2 uses 0-based indexing
                    textpage = page.get_textpage()
                    page_text = textpage.get_text_range()
                    
                    if page_text:
                        page_texts[page_num] = page_text  # Use original page number
                        full_text += f"\n--- PAGE {page_num} ---\n{page_text}\n\n"
                    
                    textpage.close()
                    page.close()
                except Exception as e:
                    logger.warning(f"Failed to extract page {page_num} with pdfium: {e}")
                    continue
        
        pdf.close()
        return full_text.strip(), page_texts
    
    def _extract_with_ocr_selective(self, pdf_path: str, page_selection: PageSelection) -> Tuple[str, Dict[int, str]]:
        """Extract specific pages using OCR."""
        full_text = ""
        page_texts = {}
        
        logger.info(f"Converting {page_selection.selected_count} selected pages to images for OCR: {pdf_path}")
        
        # Convert only selected pages to images to avoid processing all pages
        try:
            # Convert selected pages only using first_page and last_page parameters
            images = {}
            for page_num in page_selection.selected_pages:
                logger.info(f"Converting page {page_num} to image for OCR")
                try:
                    # Convert single page to image
                    page_images = convert_from_path(
                        pdf_path, 
                        dpi=300, 
                        fmt='PNG',
                        first_page=page_num,
                        last_page=page_num
                    )
                    if page_images:
                        images[page_num] = page_images[0]
                except Exception as e:
                    logger.warning(f"Failed to convert page {page_num} to image: {e}")
                    continue
        except Exception as e:
            logger.warning(f"pdf2image failed: {e}")
            # Try pypdfium2 fallback for selected pages only
            if HAS_PDFIUM:
                images = self._convert_selected_pages_with_pdfium(pdf_path, page_selection)
            else:
                logger.error("No PDF to image conversion available")
                return "", {}
        
        # Process converted images
        for page_num in page_selection.selected_pages:
            if page_num in images:
                logger.info(f"Processing page {page_num} with OCR")
                
                try:
                    image = images[page_num]
                    
                    # Use enhanced OCR if available
                    if self.use_enhanced_ocr:
                        page_text, confidence = self.enhanced_ocr.process_image(image)
                        if confidence < 0.6:
                            logger.warning(f"Low OCR confidence on page {page_num}: {confidence:.2f}")
                    else:
                        # Basic OCR processing
                        page_text = pytesseract.image_to_string(
                            image, 
                            lang='fra+eng',
                            config='--oem 3 --psm 6'
                        )
                    
                    if page_text.strip():
                        page_texts[page_num] = page_text  # Use original page number
                        full_text += f"\n--- PAGE {page_num} ---\n{page_text}\n\n"
                    else:
                        logger.warning(f"No text extracted from page {page_num}")
                        
                except Exception as e:
                    logger.error(f"OCR failed for page {page_num}: {e}")
        
        logger.info(f"OCR extraction completed. Extracted {len(page_texts)} selected pages.")
        return full_text.strip(), page_texts
    
    def _convert_pages_with_pdfium(self, pdf_path: str, total_pages: int) -> List:
        """Convert PDF pages to images using pypdfium2."""
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        images = []
        
        for i in range(min(total_pages, len(pdf))):
            page = pdf[i]
            pil_image = page.render(scale=300/72).to_pil()  # 300 DPI
            images.append(pil_image)
            page.close()
        
        pdf.close()
        return images
    
    def _convert_selected_pages_with_pdfium(self, pdf_path: str, page_selection: PageSelection) -> Dict[int, any]:
        """Convert only selected PDF pages to images using pypdfium2."""
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(pdf_path)
        images = {}
        
        for page_num in page_selection.selected_pages:
            if page_num <= len(pdf):
                logger.info(f"Converting page {page_num} to image with pypdfium2")
                page = pdf[page_num - 1]  # 0-based indexing
                pil_image = page.render(scale=300/72).to_pil()  # 300 DPI
                images[page_num] = pil_image
                page.close()
        
        pdf.close()
        return images
    
    def _filter_pages(self, all_pages: Dict[int, str], page_selection: PageSelection) -> Tuple[str, Dict[int, str]]:
        """Filter extracted pages to include only selected ones."""
        filtered_pages = {}
        full_text = ""
        
        for page_num in page_selection.selected_pages:
            if page_num in all_pages:
                filtered_pages[page_num] = all_pages[page_num]
                full_text += f"\n--- PAGE {page_num} ---\n{all_pages[page_num]}\n\n"
        
        return full_text.strip(), filtered_pages
    
    def create_page_subset_pdf(self, pdf_path: str, page_selection: PageSelection, 
                             output_path: Optional[str] = None) -> str:
        """
        Create a new PDF containing only selected pages.
        
        Args:
            pdf_path: Path to source PDF
            page_selection: PageSelection object
            output_path: Optional output path (auto-generated if None)
            
        Returns:
            Path to the created subset PDF
        """
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_path = f"{base_name}_pages_{page_selection.ranges[0][0]}-{page_selection.ranges[-1][1]}.pdf"
        
        try:
            with open(pdf_path, 'rb') as input_file:
                pdf_reader = PyPDF2.PdfReader(input_file)
                pdf_writer = PyPDF2.PdfWriter()
                
                # Add selected pages to writer
                for page_num in page_selection.selected_pages:
                    if page_num <= len(pdf_reader.pages):
                        pdf_writer.add_page(pdf_reader.pages[page_num - 1])
                
                # Write to output file
                with open(output_path, 'wb') as output_file:
                    pdf_writer.write(output_file)
            
            logger.info(f"Created page subset PDF: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create page subset PDF: {e}")
            raise
    
    def _should_enhance_with_ocr(self, text: str, page_texts: Dict[int, str]) -> bool:
        """Determine if OCR enhancement would be beneficial."""
        # Get text density (chars per page)
        num_pages = len(page_texts) if page_texts else 1
        text_density = len(text) / num_pages
        
        # If text density is very low, likely scanned document
        if text_density < 100:
            logger.info(f"Low text density ({text_density:.1f} chars/page), document likely scanned")
            return True
        
        # Very short text might indicate poor extraction
        if len(text) < 500:
            logger.info(f"Very short text ({len(text)} chars), considering OCR enhancement")
            return True
        
        # Check for missing pages (gaps in page numbers)
        if page_texts:
            page_numbers = sorted(page_texts.keys())
            expected_pages = set(range(1, max(page_numbers) + 1))
            missing_pages = expected_pages - set(page_numbers)
            if missing_pages:
                # Be more conservative for large documents
                total_pages = max(page_numbers)
                missing_count = len(missing_pages)
                missing_percentage = (missing_count / total_pages) * 100
                
                # Only trigger OCR enhancement if:
                # 1. Small document (< 50 pages) and any pages missing
                # 2. Medium document (50-500 pages) and > 10% missing
                # 3. Large document (> 500 pages) and > 20% missing
                if total_pages < 50:
                    logger.info(f"Missing pages detected in small document: {missing_pages}, considering OCR enhancement")
                    return True
                elif total_pages < 500 and missing_percentage > 10:
                    logger.info(f"Missing {missing_count} pages ({missing_percentage:.1f}%) in medium document, considering OCR enhancement")
                    return True
                elif total_pages >= 500 and missing_percentage > 20:
                    logger.info(f"Missing {missing_count} pages ({missing_percentage:.1f}%) in large document, considering OCR enhancement")
                    return True
                else:
                    logger.info(f"Missing {missing_count} pages ({missing_percentage:.1f}%) in {total_pages}-page document, but extraction quality seems adequate - skipping OCR enhancement")
                    return False
        
        # Check for signs of poor extraction (too many special characters)
        if text:
            special_char_ratio = len([c for c in text if not c.isalnum() and c not in ' .,;:!?\n']) / len(text)
            if special_char_ratio > 0.3:
                logger.info(f"High special char ratio ({special_char_ratio:.2f}), considering OCR enhancement")
                return True
        
        # Check for extraction quality indicators
        quality_score = self._calculate_extraction_quality(text)
        if quality_score < 0.6:
            logger.info(f"Low extraction quality score ({quality_score:.2f}), considering OCR enhancement")
            return True
        
        return False
    
    def _should_enhance_with_ocr_selective(self, text: str, page_texts: Dict[int, str], page_selection: PageSelection) -> bool:
        """Determine if OCR enhancement would be beneficial for selected pages only."""
        # For selective processing, we only check the quality of selected pages
        if not text or not page_texts:
            return True
        
        # Calculate text density for selected pages only
        selected_page_count = len(page_texts)
        if selected_page_count == 0:
            return True
            
        text_density = len(text) / selected_page_count
        
        # If text density is very low for selected pages, likely scanned
        if text_density < 100:
            logger.info(f"Low text density ({text_density:.1f} chars/page) in selected pages, document likely scanned")
            return True
        
        # Very short text from selected pages might indicate poor extraction
        if len(text) < 500:
            logger.info(f"Very short text ({len(text)} chars) from selected pages, considering OCR enhancement")
            return True
        
        # Check for missing pages within our selection (not outside it)
        if page_texts:
            selected_page_numbers = set(page_selection.selected_pages)
            extracted_page_numbers = set(page_texts.keys())
            missing_in_selection = selected_page_numbers - extracted_page_numbers
            
            if missing_in_selection:
                logger.info(f"Missing pages within selection: {missing_in_selection}, considering OCR enhancement")
                return True
        
        # Check for signs of poor extraction quality
        quality_score = self._calculate_extraction_quality(text)
        if quality_score < 0.6:
            logger.info(f"Low extraction quality score ({quality_score:.2f}) for selected pages, considering OCR enhancement")
            return True
        
        return False
    
    def _calculate_extraction_quality(self, text: str) -> float:
        """Calculate extraction quality score (0.0 to 1.0)."""
        if not text:
            return 0.0
        
        score = 0.0
        
        # Word count factor
        words = text.split()
        if len(words) > 50:
            score += 0.3
        elif len(words) > 20:
            score += 0.2
        elif len(words) > 10:
            score += 0.1
        
        # Alphanumeric ratio
        alnum_chars = sum(1 for c in text if c.isalnum())
        if text:
            alnum_ratio = alnum_chars / len(text)
            score += min(alnum_ratio * 0.4, 0.4)
        
        # Sentence structure (periods, punctuation)
        sentences = text.count('.') + text.count('!') + text.count('?')
        if sentences > 5:
            score += 0.2
        elif sentences > 2:
            score += 0.1
        
        # Legal/regulatory content indicators
        legal_indicators = [
            'article', 'règlement', 'instruction', 'cobac', 'cemac',
            'chapitre', 'section', 'titre', 'commission', 'bancaire'
        ]
        found_indicators = sum(1 for indicator in legal_indicators if indicator in text.lower())
        if found_indicators >= 3:
            score += 0.1
        
        return min(score, 1.0)
    
    def _merge_extractions(self, text1: str, pages1: Dict[int, str], 
                          text2: str, pages2: Dict[int, str]) -> Tuple[str, Dict[int, str]]:
        """Merge two extraction results, preferring the better quality text."""
        merged_pages = {}
        
        # Merge page by page
        all_page_nums = set(pages1.keys()) | set(pages2.keys())
        
        for page_num in sorted(all_page_nums):
            page1_text = pages1.get(page_num, "")
            page2_text = pages2.get(page_num, "")
            
            # Choose the better quality text
            if len(page1_text) > len(page2_text) * 1.5:
                merged_pages[page_num] = page1_text
            elif len(page2_text) > len(page1_text) * 1.5:
                merged_pages[page_num] = page2_text
            else:
                # Similar length, choose based on quality indicators
                if self._text_quality_score(page1_text) >= self._text_quality_score(page2_text):
                    merged_pages[page_num] = page1_text
                else:
                    merged_pages[page_num] = page2_text
        
        # Reconstruct full text
        merged_text = "\n\n".join(merged_pages[p] for p in sorted(merged_pages.keys()))
        
        return merged_text, merged_pages
    
    def _text_quality_score(self, text: str) -> float:
        """Calculate a simple quality score for extracted text."""
        if not text:
            return 0.0
        
        # Factors that indicate good quality
        score = 0.0
        
        # Word count
        words = text.split()
        if len(words) > 10:
            score += 0.3
        
        # Alphanumeric ratio
        alnum_ratio = len([c for c in text if c.isalnum()]) / len(text)
        score += alnum_ratio * 0.3
        
        # Presence of article indicators
        if any(marker in text.lower() for marker in ['article', 'règlement', 'chapitre']):
            score += 0.2
        
        # Reasonable line length
        lines = text.split('\n')
        avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
        if 20 < avg_line_length < 100:
            score += 0.2
        
        return min(score, 1.0)
    
    @with_retry(max_retries=2, retry_exceptions=(OSError,))
    def _extract_with_pdfplumber(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """Extract text using pdfplumber."""
        full_text = ""
        page_texts = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            page_texts[i + 1] = page_text
                            full_text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {i+1}: {e}")
                        continue
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            raise
        
        return full_text.strip(), page_texts
    
    @with_retry(max_retries=2, retry_exceptions=(OSError,))
    def _extract_with_pypdf2(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """Extract text using PyPDF2."""
        full_text = ""
        page_texts = {}
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            page_texts[i + 1] = page_text
                            full_text += page_text + "\n\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {i+1} with PyPDF2: {e}")
                        continue
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {e}")
            raise
        
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
        Validate extracted text quality with comprehensive checks.
        
        Args:
            text: Extracted text to validate
            min_length: Minimum required text length
            
        Returns:
            Boolean indicating if text is valid
        """
        if not text or len(text.strip()) < min_length:
            return False
        
        # Basic word count check
        words = text.split()
        if len(words) < 10:
            return False
        
        # Calculate quality score
        quality_score = self._calculate_extraction_quality(text)
        
        # For COBAC documents, be more lenient if we find regulatory indicators
        regulatory_indicators = [
            'commission bancaire', 'cobac', 'cemac', 'règlement', 'instruction',
            'article', 'chapitre', 'decide', 'adopte', 'vu'
        ]
        
        found_indicators = sum(1 for indicator in regulatory_indicators if indicator in text.lower())
        
        # If we found regulatory content, lower the quality threshold
        if found_indicators >= 2:
            logger.info(f"Found {found_indicators} regulatory indicators, using relaxed validation")
            return quality_score >= 0.4  # Lower threshold for regulatory docs
        
        # For other documents, use standard threshold
        return quality_score >= 0.6
    
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
        """Extract text using OCR for scanned PDFs with advanced processing."""
        # Safety check: Get page count before processing
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                # Warn for very large documents
                if total_pages > 1000:
                    logger.warning(f"Large document detected ({total_pages} pages). OCR processing may take a very long time.")
                    logger.warning("Consider using selective page processing for large documents.")
        except Exception:
            total_pages = 0
        
        # Try to use the advanced OCR engine if available
        try:
            from ocr_engine import AdvancedOCREngine
            engine = AdvancedOCREngine()
            if engine.is_available():
                logger.info("Using advanced OCR engine for extraction")
                full_text, page_texts, metadata = engine.extract_text_from_pdf(pdf_path, enhancement_level='high')
                return full_text, page_texts
        except ImportError:
            logger.info("Advanced OCR engine not available, using basic OCR")
        
        # Fallback to basic OCR
        full_text = ""
        page_texts = {}
        
        # Convert PDF to images with high quality
        logger.info(f"Converting PDF to images for OCR: {pdf_path}")
        
        # Try to use poppler if available, otherwise use pypdfium2
        try:
            images = convert_from_path(pdf_path, dpi=300, fmt='PNG')
        except Exception as e:
            logger.warning(f"pdf2image failed (Poppler not installed?): {e}")
            logger.info("Falling back to pypdfium2 for PDF to image conversion")
            
            # Use pypdfium2 as fallback
            if HAS_PDFIUM:
                import pypdfium2 as pdfium
                pdf = pdfium.PdfDocument(pdf_path)
                images = []
                for i in range(len(pdf)):
                    page = pdf[i]
                    pil_image = page.render(scale=300/72).to_pil()  # 300 DPI
                    images.append(pil_image)
                    page.close()
                pdf.close()
            else:
                logger.error("Neither Poppler nor pypdfium2 available for PDF to image conversion")
                return "", {}
        
        # Configure OCR for French regulatory documents
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        
        for i, image in enumerate(images):
            logger.info(f"Processing page {i+1}/{len(images)} with OCR")
            
            try:
                # Use enhanced OCR if available
                if self.use_enhanced_ocr:
                    page_text, confidence = self.enhanced_ocr.process_image(image)
                    
                    # Log OCR quality
                    if confidence < 0.6:
                        logger.warning(f"Low OCR confidence on page {i+1}: {confidence:.2f}")
                    else:
                        logger.debug(f"OCR confidence on page {i+1}: {confidence:.2f}")
                
                else:
                    # Fallback to basic OCR processing
                    from PIL import ImageEnhance, ImageFilter
                    
                    # Convert to grayscale
                    image = image.convert('L')
                    
                    # Enhance contrast
                    enhancer = ImageEnhance.Contrast(image)
                    image = enhancer.enhance(1.5)
                    
                    # Reduce noise
                    image = image.filter(ImageFilter.MedianFilter(size=3))
                    
                    # Perform OCR with both French and English
                    page_text = pytesseract.image_to_string(
                        image, 
                        lang='fra+eng',  # French + English
                        config=custom_config
                    )
                    confidence = 0.7  # Default confidence for basic OCR
                
                if page_text.strip():
                    # Post-process the text (enhanced version includes better corrections)
                    if not self.use_enhanced_ocr:
                        page_text = self._post_process_ocr_text(page_text)
                    
                    page_texts[i + 1] = page_text
                    full_text += f"\n--- PAGE {i+1} ---\n{page_text}\n\n"
                else:
                    logger.warning(f"No text extracted from page {i+1}")
                    
            except Exception as e:
                logger.error(f"OCR failed for page {i+1}: {e}")
        
        logger.info(f"OCR extraction completed. Extracted {len(page_texts)} pages with text.")
        return full_text.strip(), page_texts
    
    def _extract_with_ocr_safe(self, pdf_path: str) -> Tuple[str, Dict[int, str]]:
        """Safe wrapper for OCR extraction with error recovery."""
        error_manager = get_error_manager()
        
        try:
            return self._extract_with_ocr(pdf_path)
        except Exception as e:
            error_manager._record_error(e, "_extract_with_ocr_safe", pdf_path)
            logger.warning(f"OCR extraction failed safely: {e}")
            return "", {}
    
    def _post_process_ocr_text(self, text: str) -> str:
        """Post-process OCR text to fix common errors."""
        import re
        
        # Fix common OCR errors in French regulatory documents
        replacements = {
            r'\bl\b': '1',  # Lowercase l mistaken for 1
            r'\bO\b': '0',  # Letter O mistaken for 0
            r'(\d),(\d)': r'\1.\2',  # French decimal separator
            r'\s+': ' ',  # Multiple spaces to single
            r'([.!?])([A-Z])': r'\1 \2',  # Missing space after punctuation
            r'AIT\.': 'ART.',  # Common misrecognition
            r'Aiticle': 'Article',
            r'aiticle': 'article',
        }
        
        processed = text
        for pattern, replacement in replacements.items():
            processed = re.sub(pattern, replacement, processed)
        
        # Clean up excessive newlines
        processed = re.sub(r'\n\s*\n\s*\n', '\n\n', processed)
        
        return processed.strip()
    
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