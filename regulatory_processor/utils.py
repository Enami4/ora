"""
Utility functions for the regulatory document processor.
"""

import os
import re
import hashlib
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use ('sha256' or 'md5')
        
    Returns:
        Hex digest of the file hash
    """
    hash_func = hashlib.sha256() if algorithm == 'sha256' else hashlib.md5()
    
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    text = re.sub(r'\s+', ' ', text)
    
    text = re.sub(r'\.{3,}', '...', text)
    
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    text = re.sub(r'[^\S\n]+', ' ', text)
    
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if line:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def identify_document_type(file_path: str) -> str:
    """
    Identify document type based on file path and name.
    
    Args:
        file_path: Path to the document
        
    Returns:
        Document type string
    """
    file_name = os.path.basename(file_path).lower()
    path_lower = file_path.lower()
    
    type_patterns = {
        'INSTRUCTION': ['instruction', 'i-20', 'cbi-'],
        'REGLEMENT': ['reglement', 'r-20', 'cbr-', 'rglt'],
        'CODE_PENAL': ['code', 'penal', 'pénal'],
        'DECISION': ['decision', 'dec-', 'dcobac'],
        'LETTRE_CIRCULAIRE': ['lc-', 'lettre', 'circulaire'],
    }
    
    for doc_type, patterns in type_patterns.items():
        if any(pattern in file_name or pattern in path_lower for pattern in patterns):
            return doc_type
    
    return 'OTHER'


def extract_document_number(file_name: str) -> Optional[str]:
    """
    Extract document number from filename.
    
    Args:
        file_name: Name of the file
        
    Returns:
        Document number if found
    """
    patterns = [
        r'[IR]-(\d{4}[-_]\d{2})',
        r'cb[IR]-(\d{2,4}[-_]\d{2})',
        r'R-(\d{4}[-_]\d{2})',
        r'LC-(\d{3})',
        r'Dec(\d{2}-\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, file_name, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def validate_pdf_file(file_path: str, max_size_mb: int = 100) -> tuple[bool, str]:
    """
    Validate PDF file before processing.
    
    Args:
        file_path: Path to PDF file
        max_size_mb: Maximum allowed file size in MB
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not os.path.exists(file_path):
        return False, f"File does not exist: {file_path}"
    
    if not file_path.lower().endswith('.pdf'):
        return False, f"Not a PDF file: {file_path}"
    
    file_size = os.path.getsize(file_path)
    if file_size > max_size_mb * 1024 * 1024:
        return False, f"File too large: {format_file_size(file_size)} (max: {max_size_mb}MB)"
    
    if file_size == 0:
        return False, "File is empty"
    
    try:
        with open(file_path, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                return False, "Invalid PDF header"
    except Exception as e:
        return False, f"Cannot read file: {e}"
    
    return True, ""


def create_output_directory(base_path: str, subfolder: str = "output") -> str:
    """
    Create output directory if it doesn't exist.
    
    Args:
        base_path: Base directory path
        subfolder: Name of output subfolder
        
    Returns:
        Path to output directory
    """
    output_dir = os.path.join(base_path, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_pdf_files(directory: str, recursive: bool = True) -> List[str]:
    """
    Get all PDF files in a directory.
    
    Args:
        directory: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        List of PDF file paths
    """
    pdf_files = []
    
    if recursive:
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
    else:
        pdf_files = [
            os.path.join(directory, f) 
            for f in os.listdir(directory) 
            if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(directory, f))
        ]
    
    return sorted(pdf_files)


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    Truncate text to maximum length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add when truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix