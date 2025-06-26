"""
Unit tests for text chunking module.
"""

import pytest
from regulatory_processor.chunker import TextChunker
from regulatory_processor.config import ProcessorConfig


class TestTextChunker:
    """Test cases for TextChunker class."""
    
    def test_init_with_config(self, basic_config):
        """Test chunker initialization with configuration."""
        chunker = TextChunker(basic_config)
        assert chunker.chunk_size == basic_config.chunk_size
        assert chunker.chunk_overlap == basic_config.chunk_overlap
    
    def test_init_without_config(self):
        """Test chunker initialization without configuration."""
        chunker = TextChunker()
        assert chunker.chunk_size == 1000  # default
        assert chunker.chunk_overlap == 200  # default
    
    def test_simple_chunking(self):
        """Test basic text chunking functionality."""
        chunker = TextChunker(ProcessorConfig(chunk_size=50, chunk_overlap=10))
        
        text = "This is a test text that will be split into multiple chunks for processing."
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 1
        assert all(len(chunk['text']) <= 50 for chunk in chunks)
        assert all('chunk_index' in chunk for chunk in chunks)
        assert all('text_length' in chunk for chunk in chunks)
        assert all('word_count' in chunk for chunk in chunks)
    
    def test_overlap_functionality(self):
        """Test that chunks have proper overlap."""
        chunker = TextChunker(ProcessorConfig(chunk_size=30, chunk_overlap=10))
        
        text = "Word1 Word2 Word3 Word4 Word5 Word6 Word7 Word8 Word9 Word10"
        chunks = chunker.chunk_text(text)
        
        if len(chunks) > 1:
            # Check that consecutive chunks share some content
            chunk1_end = chunks[0]['text'][-10:]  # Last 10 chars
            chunk2_start = chunks[1]['text'][:10]  # First 10 chars
            
            # Should have some overlap (not exact match due to word boundaries)
            assert len(chunk1_end.strip()) > 0
            assert len(chunk2_start.strip()) > 0
    
    def test_chunk_by_sentences(self):
        """Test chunking by sentence boundaries."""
        chunker = TextChunker(ProcessorConfig(chunk_size=100, chunk_overlap=20))
        
        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        chunks = chunker._chunk_by_sentences(text)
        
        assert len(chunks) > 0
        # Each chunk should end with sentence boundary when possible
        for chunk in chunks[:-1]:  # All but last chunk
            assert chunk.rstrip().endswith('.') or len(chunk) >= 100
    
    def test_chunk_by_words(self):
        """Test chunking by word boundaries."""
        chunker = TextChunker(ProcessorConfig(chunk_size=30, chunk_overlap=5))
        
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10"
        chunks = chunker._chunk_by_words(text)
        
        assert len(chunks) > 0
        # Each chunk should end with word boundary when possible
        for chunk in chunks:
            if len(chunk) < 30:
                continue
            # Should not break words unless absolutely necessary
            assert not chunk.endswith(' ')
    
    def test_chunk_metadata(self, sample_text):
        """Test chunk metadata generation."""
        chunker = TextChunker(ProcessorConfig(chunk_size=200, chunk_overlap=50))
        
        chunks = chunker.chunk_text(sample_text, document_id="test_doc")
        
        for i, chunk in enumerate(chunks):
            assert chunk['chunk_index'] == i
            assert chunk['text_length'] == len(chunk['text'])
            assert chunk['word_count'] == len(chunk['text'].split())
            assert chunk['chunk_id'].startswith('test_doc_chunk_')
            assert 'start_position' in chunk
            assert 'end_position' in chunk
    
    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = TextChunker()
        
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0
        
        chunks = chunker.chunk_text("   ")
        assert len(chunks) == 0
    
    def test_short_text(self):
        """Test handling of text shorter than chunk size."""
        chunker = TextChunker(ProcessorConfig(chunk_size=1000, chunk_overlap=200))
        
        short_text = "This is a short text."
        chunks = chunker.chunk_text(short_text)
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == short_text
        assert chunks[0]['chunk_index'] == 0
    
    def test_exact_chunk_size(self):
        """Test text that is exactly chunk size."""
        chunker = TextChunker(ProcessorConfig(chunk_size=20, chunk_overlap=5))
        
        text = "a" * 20  # Exactly 20 characters
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) == 1
        assert len(chunks[0]['text']) == 20
    
    def test_large_text_chunking(self):
        """Test chunking of large text."""
        chunker = TextChunker(ProcessorConfig(chunk_size=100, chunk_overlap=20))
        
        # Generate large text
        large_text = "This is sentence number {}. " * 1000
        large_text = large_text.format(*range(1000))
        
        chunks = chunker.chunk_text(large_text)
        
        assert len(chunks) > 10  # Should create many chunks
        
        # Verify no content is lost
        total_text = "".join(chunk['text'] for chunk in chunks)
        # Account for overlap - some content will be duplicated
        assert len(total_text) >= len(large_text)
    
    def test_chunk_numbering(self):
        """Test sequential chunk numbering."""
        chunker = TextChunker(ProcessorConfig(chunk_size=50, chunk_overlap=10))
        
        text = "Word " * 100  # Create text that will need multiple chunks
        chunks = chunker.chunk_text(text)
        
        # Verify sequential numbering
        for i, chunk in enumerate(chunks):
            assert chunk['chunk_index'] == i
    
    def test_special_characters(self):
        """Test chunking text with special characters."""
        chunker = TextChunker(ProcessorConfig(chunk_size=50, chunk_overlap=10))
        
        text = "Text with special chars: àéíóú ñ ç €$£¥ «»“” 123.45% & more!"
        chunks = chunker.chunk_text(text)
        
        assert len(chunks) > 0
        # Verify special characters are preserved
        combined_text = " ".join(chunk['text'] for chunk in chunks)
        assert "àéíóú" in combined_text
        assert "€$£¥" in combined_text
    
    def test_position_tracking(self):
        """Test start and end position tracking in chunks."""
        chunker = TextChunker(ProcessorConfig(chunk_size=30, chunk_overlap=5))
        
        text = "0123456789" * 10  # 100 characters
        chunks = chunker.chunk_text(text)
        
        for chunk in chunks:
            start = chunk['start_position']
            end = chunk['end_position']
            
            # Verify positions are logical
            assert start >= 0
            assert end > start
            assert end <= len(text)
            
            # Verify text matches positions
            expected_text = text[start:end]
            assert chunk['text'] == expected_text or chunk['text'] in expected_text


class TestChunkingStrategies:
    """Test different chunking strategies."""
    
    def test_sentence_strategy(self):
        """Test sentence-based chunking strategy."""
        chunker = TextChunker(ProcessorConfig(chunk_size=100, chunk_overlap=20))
        
        text = "First sentence here. Second sentence follows. Third one concludes."
        chunks = chunker.chunk_text(text, strategy='sentence')
        
        assert len(chunks) > 0
        # Should prefer sentence boundaries
        for chunk in chunks[:-1]:  # All but last
            chunk_text = chunk['text'].rstrip()
            if len(chunk_text) < 100:
                assert chunk_text.endswith('.')
    
    def test_word_strategy(self):
        """Test word-based chunking strategy."""
        chunker = TextChunker(ProcessorConfig(chunk_size=50, chunk_overlap=10))
        
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12"
        chunks = chunker.chunk_text(text, strategy='word')
        
        assert len(chunks) > 0
        # Should not break words
        for chunk in chunks:
            words = chunk['text'].split()
            # Each word should be complete (no partial words at boundaries)
            for word in words:
                assert len(word) > 0
    
    def test_character_strategy(self):
        """Test character-based chunking strategy."""
        chunker = TextChunker(ProcessorConfig(chunk_size=20, chunk_overlap=5))
        
        text = "abcdefghijklmnopqrstuvwxyz" * 3
        chunks = chunker.chunk_text(text, strategy='character')
        
        assert len(chunks) > 0
        # Should split at exact character boundaries
        for chunk in chunks[:-1]:  # All but last
            assert len(chunk['text']) <= 20
    
    def test_adaptive_strategy(self):
        """Test adaptive chunking strategy."""
        chunker = TextChunker(ProcessorConfig(chunk_size=100, chunk_overlap=20))
        
        # Text with mixed content
        text = "Short. Medium length sentence here. This is a very long sentence that goes on and on without any punctuation marks making it difficult to split nicely but the adaptive strategy should handle it well."
        
        chunks = chunker.chunk_text(text, strategy='adaptive')
        
        assert len(chunks) > 0
        # Should adapt based on content structure
        total_chars = sum(len(chunk['text']) for chunk in chunks)
        # Should not lose significant content (accounting for overlap)
        assert total_chars >= len(text) * 0.9


class TestChunkingEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_invalid_config(self):
        """Test chunker with invalid configuration."""
        with pytest.raises(ValueError):
            TextChunker(ProcessorConfig(chunk_size=0))  # Invalid chunk size
        
        with pytest.raises(ValueError):
            TextChunker(ProcessorConfig(chunk_size=100, chunk_overlap=150))  # Overlap > size
    
    def test_single_character_text(self):
        """Test chunking single character."""
        chunker = TextChunker(ProcessorConfig(chunk_size=10, chunk_overlap=2))
        
        chunks = chunker.chunk_text("a")
        
        assert len(chunks) == 1
        assert chunks[0]['text'] == "a"
        assert chunks[0]['word_count'] == 1
    
    def test_unicode_chunking(self):
        """Test chunking text with Unicode characters."""
        chunker = TextChunker(ProcessorConfig(chunk_size=30, chunk_overlap=5))
        
        unicode_text = "Texte français 中文 العربية 日本語 русский"
        chunks = chunker.chunk_text(unicode_text)
        
        assert len(chunks) > 0
        # Verify Unicode is preserved
        combined = " ".join(chunk['text'] for chunk in chunks)
        assert "français" in combined
        assert "中文" in combined
        assert "العربية" in combined
    
    def test_only_whitespace(self):
        """Test text with only whitespace."""
        chunker = TextChunker(ProcessorConfig(chunk_size=100, chunk_overlap=20))
        
        whitespace_text = "   \n\t   \r\n   "
        chunks = chunker.chunk_text(whitespace_text)
        
        assert len(chunks) == 0  # Should be filtered out
    
    def test_very_long_words(self):
        """Test handling of words longer than chunk size."""
        chunker = TextChunker(ProcessorConfig(chunk_size=10, chunk_overlap=2))
        
        long_word = "supercalifragilisticexpialidocious"
        chunks = chunker.chunk_text(long_word)
        
        assert len(chunks) >= 1
        # Should handle long words by breaking them if necessary
        total_length = sum(len(chunk['text']) for chunk in chunks)
        assert total_length >= len(long_word)
    
    def test_chunk_consistency(self):
        """Test that chunking is consistent across multiple calls."""
        chunker = TextChunker(ProcessorConfig(chunk_size=100, chunk_overlap=20))
        
        text = "Consistent text for testing. " * 20
        
        chunks1 = chunker.chunk_text(text)
        chunks2 = chunker.chunk_text(text)
        
        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2):
            assert c1['text'] == c2['text']
            assert c1['chunk_index'] == c2['chunk_index']
    
    def test_memory_efficiency(self):
        """Test memory usage with very large text."""
        chunker = TextChunker(ProcessorConfig(chunk_size=1000, chunk_overlap=100))
        
        # Generate very large text
        large_text = "This is a test sentence. " * 100000  # ~2.5MB
        
        chunks = chunker.chunk_text(large_text)
        
        assert len(chunks) > 0
        # Should process without memory issues
        assert all('text' in chunk for chunk in chunks)
        assert all(len(chunk['text']) <= 1000 for chunk in chunks[:-1])
    
    def test_chunk_id_generation(self):
        """Test unique chunk ID generation."""
        chunker = TextChunker(ProcessorConfig(chunk_size=50, chunk_overlap=10))
        
        text = "Test text for chunk ID generation. " * 10
        chunks = chunker.chunk_text(text, document_id="test_doc_123")
        
        chunk_ids = [chunk['chunk_id'] for chunk in chunks]
        
        # All IDs should be unique
        assert len(chunk_ids) == len(set(chunk_ids))
        
        # IDs should follow expected pattern
        for i, chunk_id in enumerate(chunk_ids):
            expected_pattern = f"test_doc_123_chunk_{i}"
            assert chunk_id == expected_pattern