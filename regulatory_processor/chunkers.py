"""
Text chunking module for document segmentation.
"""

import re
import logging
import hashlib
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

logger = logging.getLogger(__name__)


class TextChunker:
    """Handles text chunking with various strategies."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the text chunker.
        
        Args:
            chunk_size: Maximum size of chunks in characters
            chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._ensure_nltk_data()
    
    def _ensure_nltk_data(self):
        """Download required NLTK data if not present."""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', quiet=True)
            except Exception as e:
                logger.warning(f"Failed to download NLTK punkt tokenizer: {e}")
    
    def chunk_by_sentences(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text by sentences while respecting chunk size limits.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunk dictionaries
        """
        if not text:
            return []
        
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = self._fallback_sentence_split(text)
        
        chunks = []
        current_chunk = ""
        current_sentences = []
        chunk_start_index = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if len(current_chunk) + len(sentence) + 1 <= self.chunk_size:
                current_chunk += sentence + " "
                current_sentences.append(sentence)
            else:
                if current_chunk:
                    chunk_data = self._create_chunk(
                        text=current_chunk.strip(),
                        sentences=current_sentences,
                        index=len(chunks),
                        start_char=chunk_start_index,
                        metadata=metadata
                    )
                    chunks.append(chunk_data)
                
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = current_chunk[-self.chunk_overlap:].strip()
                    current_chunk = overlap_text + " " + sentence + " "
                    chunk_start_index += len(current_chunk) - self.chunk_overlap - len(sentence) - 1
                else:
                    current_chunk = sentence + " "
                    chunk_start_index += len(chunks[-1]['text']) if chunks else 0
                
                current_sentences = [sentence]
        
        if current_chunk:
            chunk_data = self._create_chunk(
                text=current_chunk.strip(),
                sentences=current_sentences,
                index=len(chunks),
                start_char=chunk_start_index,
                metadata=metadata
            )
            chunks.append(chunk_data)
        
        return chunks
    
    def chunk_by_paragraphs(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text by paragraphs.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunk dictionaries
        """
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_start_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            if len(paragraph) > self.chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        text=current_chunk.strip(),
                        sentences=None,
                        index=len(chunks),
                        start_char=chunk_start_index,
                        metadata=metadata
                    ))
                    chunk_start_index += len(current_chunk)
                    current_chunk = ""
                
                sub_chunks = self.chunk_by_sentences(paragraph, metadata)
                for sub_chunk in sub_chunks:
                    sub_chunk['index'] = len(chunks)
                    chunks.append(sub_chunk)
            
            elif len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk:
                    chunks.append(self._create_chunk(
                        text=current_chunk.strip(),
                        sentences=None,
                        index=len(chunks),
                        start_char=chunk_start_index,
                        metadata=metadata
                    ))
                    chunk_start_index += len(current_chunk)
                
                current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(self._create_chunk(
                text=current_chunk.strip(),
                sentences=None,
                index=len(chunks),
                start_char=chunk_start_index,
                metadata=metadata
            ))
        
        return chunks
    
    def chunk_by_tokens(self, text: str, metadata: Optional[Dict[str, Any]] = None, 
                       tokens_per_chunk: int = 200) -> List[Dict[str, Any]]:
        """
        Chunk text by token count.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            tokens_per_chunk: Number of tokens per chunk
            
        Returns:
            List of chunk dictionaries
        """
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()
        
        chunks = []
        for i in range(0, len(tokens), tokens_per_chunk - self.chunk_overlap // 4):
            chunk_tokens = tokens[i:i + tokens_per_chunk]
            chunk_text = ' '.join(chunk_tokens)
            
            chunks.append(self._create_chunk(
                text=chunk_text,
                sentences=None,
                index=len(chunks),
                start_char=i,
                metadata=metadata
            ))
        
        return chunks
    
    def _create_chunk(self, text: str, sentences: Optional[List[str]], 
                     index: int, start_char: int, 
                     metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a structured chunk dictionary."""
        chunk_data = {
            'chunk_index': index,
            'text': text,
            'text_length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(sentences) if sentences else text.count('.') + text.count('!') + text.count('?'),
            'start_char': start_char,
            'chunk_hash': hashlib.md5(text.encode()).hexdigest()
        }
        
        if metadata:
            chunk_data.update({
                'file_name': metadata.get('file_name', ''),
                'file_path': metadata.get('file_path', ''),
                'document_type': metadata.get('document_type', ''),
                'chunk_id': f"{metadata.get('file_hash', 'unknown')}_{index}"
            })
        
        return chunk_data
    
    def _fallback_sentence_split(self, text: str) -> List[str]:
        """Fallback sentence splitting when NLTK is not available."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(TextChunker):
    """Advanced chunker that considers semantic boundaries."""
    
    def chunk_by_sections(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk text by detecting section headers and boundaries.
        
        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks
            
        Returns:
            List of chunk dictionaries
        """
        section_patterns = [
            r'^[IVX]+\.\s+',
            r'^\d+\.\s+',
            r'^[A-Z][A-Z\s]+:',
            r'^Article\s+\d+',
            r'^Section\s+\d+',
            r'^Chapitre\s+[IVX]+',
            r'^TITRE\s+[IVX]+',
        ]
        
        combined_pattern = '|'.join(f'({pattern})' for pattern in section_patterns)
        
        sections = re.split(f'({combined_pattern})', text, flags=re.MULTILINE)
        
        chunks = []
        current_section = ""
        current_header = ""
        
        for i, part in enumerate(sections):
            if part and re.match(combined_pattern, part):
                if current_section:
                    section_chunks = self.chunk_by_sentences(
                        current_header + current_section, 
                        metadata
                    )
                    chunks.extend(section_chunks)
                current_header = part
                current_section = ""
            elif part:
                current_section += part
        
        if current_section:
            section_chunks = self.chunk_by_sentences(
                current_header + current_section, 
                metadata
            )
            chunks.extend(section_chunks)
        
        for i, chunk in enumerate(chunks):
            chunk['chunk_index'] = i
        
        return chunks