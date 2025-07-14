"""
Page selection and range parsing module for selective document processing.
This module handles parsing, validation, and normalization of page ranges.
"""

import re
import logging
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
import PyPDF2

logger = logging.getLogger(__name__)


@dataclass
class PageSelection:
    """Represents a validated page selection."""
    ranges: List[Tuple[int, int]]  # List of (start, end) tuples
    total_pages: int               # Total pages in document
    selected_pages: List[int]      # Normalized list of page numbers
    selected_count: int            # Number of pages selected
    original_numbering: Dict[int, int]  # Maps processed index to original page number
    
    def contains_page(self, page_num: int) -> bool:
        """Check if a page number is in the selection."""
        return page_num in self.selected_pages
    
    def get_selected_percentage(self) -> float:
        """Get percentage of document selected."""
        if self.total_pages == 0:
            return 0.0
        return (self.selected_count / self.total_pages) * 100


class PageRangeParser:
    """Parse and validate page range specifications."""
    
    def __init__(self):
        # Regex patterns for different range formats
        self.single_page_pattern = re.compile(r'^\d+$')
        self.range_pattern = re.compile(r'^(\d+)\s*-\s*(\d+)$')
        self.valid_chars_pattern = re.compile(r'^[\d\s,\-]+$')
    
    def parse_range_string(self, range_str: str) -> List[Tuple[int, int]]:
        """
        Parse a page range string into a list of (start, end) tuples.
        
        Args:
            range_str: String like "1-10, 15, 20-25, 30"
            
        Returns:
            List of (start, end) tuples
            
        Raises:
            ValueError: If the range string is invalid
        """
        if not range_str or not range_str.strip():
            raise ValueError("Page range string cannot be empty")
        
        # Remove extra whitespace
        range_str = range_str.strip()
        
        # Validate characters
        if not self.valid_chars_pattern.match(range_str):
            raise ValueError(f"Invalid characters in page range: {range_str}")
        
        ranges = []
        
        # Split by comma
        parts = [part.strip() for part in range_str.split(',')]
        
        for part in parts:
            if not part:
                continue
                
            # Check if it's a single page
            if self.single_page_pattern.match(part):
                page_num = int(part)
                if page_num < 1:
                    raise ValueError(f"Page number must be positive: {page_num}")
                ranges.append((page_num, page_num))
                
            # Check if it's a range
            elif self.range_pattern.match(part):
                match = self.range_pattern.match(part)
                start, end = int(match.group(1)), int(match.group(2))
                
                if start < 1 or end < 1:
                    raise ValueError(f"Page numbers must be positive: {part}")
                
                if start > end:
                    raise ValueError(f"Invalid range (start > end): {part}")
                
                ranges.append((start, end))
                
            else:
                raise ValueError(f"Invalid page range format: {part}")
        
        if not ranges:
            raise ValueError("No valid page ranges found")
        
        return ranges
    
    def validate_ranges(self, ranges: List[Tuple[int, int]], max_pages: int) -> Tuple[bool, Optional[str]]:
        """
        Validate page ranges against document page count.
        
        Args:
            ranges: List of (start, end) tuples
            max_pages: Total number of pages in document
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if max_pages <= 0:
            return False, "Document has no pages"
        
        for start, end in ranges:
            if start > max_pages:
                return False, f"Page {start} exceeds document length ({max_pages} pages)"
            if end > max_pages:
                return False, f"Page {end} exceeds document length ({max_pages} pages)"
        
        return True, None
    
    def normalize_ranges(self, ranges: List[Tuple[int, int]]) -> List[int]:
        """
        Convert ranges to a sorted list of unique page numbers.
        
        Args:
            ranges: List of (start, end) tuples
            
        Returns:
            Sorted list of unique page numbers
        """
        page_set: Set[int] = set()
        
        for start, end in ranges:
            page_set.update(range(start, end + 1))
        
        return sorted(list(page_set))
    
    def merge_overlapping_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Merge overlapping or adjacent ranges for efficiency.
        
        Args:
            ranges: List of (start, end) tuples
            
        Returns:
            Optimized list of non-overlapping ranges
        """
        if not ranges:
            return []
        
        # Sort ranges by start position
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        
        merged = [sorted_ranges[0]]
        
        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            
            # Check if ranges overlap or are adjacent
            if current_start <= last_end + 1:
                # Merge ranges
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                # Add as new range
                merged.append((current_start, current_end))
        
        return merged
    
    def get_page_count(self, pdf_path: str) -> int:
        """
        Get the total number of pages in a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Number of pages
            
        Raises:
            Exception: If PDF cannot be read
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return len(pdf_reader.pages)
        except Exception as e:
            logger.error(f"Failed to get page count from {pdf_path}: {e}")
            raise
    
    def create_page_selection(self, range_str: str, pdf_path: str) -> PageSelection:
        """
        Create a validated PageSelection object from a range string and PDF.
        
        Args:
            range_str: Page range specification
            pdf_path: Path to PDF file
            
        Returns:
            PageSelection object
            
        Raises:
            ValueError: If ranges are invalid
        """
        # Parse ranges
        ranges = self.parse_range_string(range_str)
        
        # Get document page count
        total_pages = self.get_page_count(pdf_path)
        
        # Validate ranges
        is_valid, error_msg = self.validate_ranges(ranges, total_pages)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Merge overlapping ranges for efficiency
        optimized_ranges = self.merge_overlapping_ranges(ranges)
        
        # Normalize to page list
        selected_pages = self.normalize_ranges(optimized_ranges)
        
        # Create page numbering map (for maintaining original page numbers)
        original_numbering = {i: page for i, page in enumerate(selected_pages)}
        
        return PageSelection(
            ranges=optimized_ranges,
            total_pages=total_pages,
            selected_pages=selected_pages,
            selected_count=len(selected_pages),
            original_numbering=original_numbering
        )
    
    def format_ranges_display(self, selection: PageSelection) -> str:
        """
        Format page ranges for display.
        
        Args:
            selection: PageSelection object
            
        Returns:
            Formatted string for display
        """
        range_strs = []
        for start, end in selection.ranges:
            if start == end:
                range_strs.append(str(start))
            else:
                range_strs.append(f"{start}-{end}")
        
        ranges_text = ", ".join(range_strs)
        percentage = selection.get_selected_percentage()
        
        return f"{ranges_text} ({selection.selected_count}/{selection.total_pages} pages, {percentage:.1f}%)"
    
    def suggest_corrections(self, range_str: str, max_pages: int) -> List[str]:
        """
        Suggest corrections for invalid page ranges.
        
        Args:
            range_str: Invalid range string
            max_pages: Total pages in document
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Check for common mistakes
        if "-" not in range_str and "," not in range_str:
            # Single number that might be too high
            try:
                num = int(range_str.strip())
                if num > max_pages:
                    suggestions.append(f"Did you mean: 1-{max_pages}?")
                    suggestions.append(f"Or perhaps: {max_pages}")
            except ValueError:
                pass
        
        # Check for reversed ranges
        if "-" in range_str:
            parts = range_str.split("-")
            if len(parts) == 2:
                try:
                    start, end = int(parts[0].strip()), int(parts[1].strip())
                    if start > end:
                        suggestions.append(f"Did you mean: {end}-{start}?")
                except ValueError:
                    pass
        
        # Suggest valid format examples
        if not suggestions:
            suggestions.append("Valid formats: '1-10', '1,3,5', '1-10,15-20'")
            suggestions.append(f"Document has {max_pages} pages (1-{max_pages})")
        
        return suggestions


class SmartPageDetector:
    """Basic smart page detection for common patterns."""
    
    @staticmethod
    def detect_table_of_contents(pdf_path: str, max_pages: int = 10) -> Optional[Tuple[int, int]]:
        """
        Try to detect table of contents pages.
        
        Args:
            pdf_path: Path to PDF
            max_pages: Maximum pages to check
            
        Returns:
            Tuple of (start, end) pages for TOC, or None
        """
        # This is a placeholder for more sophisticated detection
        # In a real implementation, we'd analyze page content
        logger.debug("Table of contents detection not yet implemented")
        return None
    
    @staticmethod
    def suggest_content_pages(pdf_path: str) -> Optional[str]:
        """
        Suggest page ranges based on document structure.
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            Suggested range string, or None
        """
        # This is a placeholder for AI-based detection
        # Could analyze page density, headers, etc.
        logger.debug("Content page suggestion not yet implemented")
        return None


# Convenience functions
def parse_page_range(range_str: str, pdf_path: str) -> PageSelection:
    """
    Convenience function to parse and validate page ranges.
    
    Args:
        range_str: Page range specification
        pdf_path: Path to PDF file
        
    Returns:
        PageSelection object
    """
    parser = PageRangeParser()
    return parser.create_page_selection(range_str, pdf_path)


def validate_page_range(range_str: str, max_pages: int) -> Tuple[bool, Optional[str]]:
    """
    Convenience function to validate page ranges.
    
    Args:
        range_str: Page range specification
        max_pages: Total pages in document
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    parser = PageRangeParser()
    try:
        ranges = parser.parse_range_string(range_str)
        return parser.validate_ranges(ranges, max_pages)
    except ValueError as e:
        return False, str(e)