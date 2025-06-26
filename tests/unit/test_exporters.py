"""
Unit tests for Excel export module.
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from regulatory_processor.exporters import ExcelExporter, ExcelFormatter
from regulatory_processor.config import ProcessorConfig
from regulatory_processor.validators import Article, MaterialityLevel, ValidationScore


class TestExcelFormatter:
    """Test ExcelFormatter class."""
    
    def test_format_chunk_data(self, sample_chunks):
        """Test formatting chunk data for Excel."""
        formatter = ExcelFormatter()
        
        formatted = formatter.format_chunks(sample_chunks)
        
        assert isinstance(formatted, pd.DataFrame)
        assert len(formatted) == len(sample_chunks)
        assert 'chunk_index' in formatted.columns
        assert 'text' in formatted.columns
        assert 'text_length' in formatted.columns
        assert 'word_count' in formatted.columns
    
    def test_format_articles_data(self, sample_articles):
        """Test formatting articles data for Excel."""
        formatter = ExcelFormatter()
        
        formatted = formatter.format_articles(sample_articles)
        
        assert isinstance(formatted, pd.DataFrame)
        assert len(formatted) == len(sample_articles)
        assert 'number' in formatted.columns
        assert 'title' in formatted.columns
        assert 'content' in formatted.columns
        assert 'materiality' in formatted.columns
        assert 'materiality_reasoning' in formatted.columns
    
    def test_format_validation_scores(self):
        """Test formatting validation scores for Excel."""
        formatter = ExcelFormatter()
        
        scores = [
            ValidationScore(
                completeness_score=85,
                reliability_score=90,
                legal_structure_score=75,
                overall_score=83,
                completeness_issues=["Issue 1"],
                reliability_issues=[],
                legal_structure_issues=["Issue 2"],
                recommendations=["Recommendation 1"]
            ),
            ValidationScore(
                completeness_score=80,
                reliability_score=85,
                legal_structure_score=70,
                overall_score=78
            )
        ]
        
        formatted = formatter.format_validation_scores(scores)
        
        assert isinstance(formatted, pd.DataFrame)
        assert len(formatted) == len(scores)
        assert 'completeness_score' in formatted.columns
        assert 'reliability_score' in formatted.columns
        assert 'legal_structure_score' in formatted.columns
        assert 'overall_score' in formatted.columns
    
    def test_format_with_user_info(self, sample_chunks, user_info):
        """Test formatting with user information."""
        formatter = ExcelFormatter()
        
        formatted = formatter.format_chunks(sample_chunks, user_info=user_info)
        
        assert 'user_name' in formatted.columns
        assert 'user_surname' in formatted.columns
        assert all(formatted['user_name'] == user_info['name'])
        assert all(formatted['user_surname'] == user_info['surname'])
    
    def test_empty_data_formatting(self):
        """Test formatting empty data."""
        formatter = ExcelFormatter()
        
        # Empty chunks
        empty_chunks = formatter.format_chunks([])
        assert isinstance(empty_chunks, pd.DataFrame)
        assert len(empty_chunks) == 0
        
        # Empty articles
        empty_articles = formatter.format_articles([])
        assert isinstance(empty_articles, pd.DataFrame)
        assert len(empty_articles) == 0
        
        # Empty scores
        empty_scores = formatter.format_validation_scores([])
        assert isinstance(empty_scores, pd.DataFrame)
        assert len(empty_scores) == 0
    
    def test_materiality_level_conversion(self, sample_articles):
        """Test proper conversion of MaterialityLevel enum."""
        formatter = ExcelFormatter()
        
        formatted = formatter.format_articles(sample_articles)
        
        # Should convert enum to string values
        materiality_values = formatted['materiality'].unique()
        assert all(isinstance(val, str) for val in materiality_values)
        assert 'HIGH' in materiality_values or 'MEDIUM' in materiality_values
    
    def test_issue_list_formatting(self):
        """Test formatting of issue lists."""
        formatter = ExcelFormatter()
        
        scores = [
            ValidationScore(
                completeness_issues=["Issue 1", "Issue 2"],
                reliability_issues=["Reliability issue"],
                legal_structure_issues=[]
            )
        ]
        
        formatted = formatter.format_validation_scores(scores)
        
        # Issues should be formatted as strings
        assert isinstance(formatted.iloc[0]['completeness_issues'], str)
        assert "Issue 1" in formatted.iloc[0]['completeness_issues']
        assert "Issue 2" in formatted.iloc[0]['completeness_issues']


class TestExcelExporter:
    """Test ExcelExporter class."""
    
    def test_init_with_config(self, basic_config):
        """Test exporter initialization with configuration."""
        exporter = ExcelExporter(basic_config)
        
        assert exporter.config == basic_config
        assert isinstance(exporter.formatter, ExcelFormatter)
    
    def test_init_without_config(self):
        """Test exporter initialization without configuration."""
        exporter = ExcelExporter()
        
        assert isinstance(exporter.config, ProcessorConfig)
        assert isinstance(exporter.formatter, ExcelFormatter)
    
    @patch('pandas.ExcelWriter')
    def test_export_chunks_only(self, mock_excel_writer, tmp_path, sample_chunks, user_info):
        """Test exporting chunks only."""
        exporter = ExcelExporter()
        output_file = tmp_path / "test_chunks.xlsx"
        
        # Mock the Excel writer
        mock_writer = Mock()
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        
        exporter.export_chunks(
            chunks=sample_chunks,
            output_file=output_file,
            user_info=user_info
        )
        
        # Verify Excel writer was called
        mock_excel_writer.assert_called_once_with(output_file, engine='openpyxl')
        
        # Verify data was written to sheets
        assert mock_writer.write.called
    
    @patch('pandas.ExcelWriter')
    def test_export_complete_analysis(self, mock_excel_writer, tmp_path, sample_chunks, 
                                    sample_articles, user_info):
        """Test exporting complete analysis with all data."""
        exporter = ExcelExporter()
        output_file = tmp_path / "complete_analysis.xlsx"
        
        # Create sample validation scores
        validation_scores = [
            ValidationScore(overall_score=85) for _ in sample_chunks
        ]
        
        mock_writer = Mock()
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        
        exporter.export_complete_analysis(
            chunks=sample_chunks,
            articles=sample_articles,
            validation_scores=validation_scores,
            output_file=output_file,
            user_info=user_info
        )
        
        mock_excel_writer.assert_called_once()
        
        # Should write multiple sheets
        call_count = mock_writer.write.call_count
        assert call_count >= 3  # At least chunks, articles, and validation sheets
    
    def test_generate_filename(self, user_info):
        """Test filename generation."""
        exporter = ExcelExporter()
        
        filename = exporter.generate_filename(user_info=user_info)
        
        assert isinstance(filename, str)
        assert filename.endswith('.xlsx')
        assert user_info['name'].lower() in filename.lower()
        assert user_info['surname'].lower() in filename.lower()
    
    def test_generate_filename_without_user(self):
        """Test filename generation without user info."""
        exporter = ExcelExporter()
        
        filename = exporter.generate_filename()
        
        assert isinstance(filename, str)
        assert filename.endswith('.xlsx')
        assert 'regulatory_analysis' in filename.lower()
    
    def test_generate_filename_with_prefix(self, user_info):
        """Test filename generation with custom prefix."""
        exporter = ExcelExporter()
        
        filename = exporter.generate_filename(
            user_info=user_info,
            prefix="custom_analysis"
        )
        
        assert "custom_analysis" in filename
        assert user_info['name'].lower() in filename.lower()
    
    @patch('pandas.ExcelWriter')
    def test_export_with_styling(self, mock_excel_writer, tmp_path, sample_chunks):
        """Test export with Excel styling."""
        config = ProcessorConfig()
        exporter = ExcelExporter(config)
        output_file = tmp_path / "styled_export.xlsx"
        
        mock_writer = Mock()
        mock_workbook = Mock()
        mock_worksheet = Mock()
        
        mock_writer.book = mock_workbook
        mock_writer.sheets = {'Chunks': mock_worksheet}
        mock_excel_writer.return_value.__enter__.return_value = mock_writer
        
        exporter.export_chunks(
            chunks=sample_chunks,
            output_file=output_file,
            apply_styling=True
        )
        
        # Verify styling was applied
        assert mock_writer.write.called
    
    def test_export_error_handling(self, tmp_path, sample_chunks):
        """Test error handling during export."""
        exporter = ExcelExporter()
        
        # Try to write to a read-only location (simulate permission error)
        with patch('pandas.ExcelWriter', side_effect=PermissionError("Access denied")):
            with pytest.raises(PermissionError):
                exporter.export_chunks(
                    chunks=sample_chunks,
                    output_file=tmp_path / "readonly.xlsx"
                )
    
    def test_large_dataset_export(self, tmp_path):
        """Test export of large datasets."""
        exporter = ExcelExporter()
        
        # Generate large dataset
        large_chunks = []
        for i in range(1000):
            large_chunks.append({
                'chunk_index': i,
                'text': f'Large chunk text content {i}' * 10,
                'text_length': len(f'Large chunk text content {i}' * 10),
                'word_count': len(f'Large chunk text content {i}'.split()) * 10,
                'chunk_id': f'large_chunk_{i}'
            })
        
        output_file = tmp_path / "large_export.xlsx"
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = Mock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            # Should handle large dataset without errors
            exporter.export_chunks(
                chunks=large_chunks,
                output_file=output_file
            )
            
            assert mock_excel_writer.called
    
    def test_unicode_content_export(self, tmp_path):
        """Test export of content with Unicode characters."""
        exporter = ExcelExporter()
        
        unicode_chunks = [
            {
                'chunk_index': 0,
                'text': 'Texte français avec accents: éàüñ 中文 العربية',
                'text_length': 45,
                'word_count': 8,
                'chunk_id': 'unicode_chunk_0'
            }
        ]
        
        output_file = tmp_path / "unicode_export.xlsx"
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = Mock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            # Should handle Unicode content without errors
            exporter.export_chunks(
                chunks=unicode_chunks,
                output_file=output_file
            )
            
            assert mock_excel_writer.called
    
    def test_metadata_sheet_creation(self, tmp_path, sample_chunks, user_info):
        """Test creation of metadata sheet."""
        exporter = ExcelExporter()
        output_file = tmp_path / "metadata_test.xlsx"
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = Mock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            exporter.export_chunks(
                chunks=sample_chunks,
                output_file=output_file,
                user_info=user_info,
                include_metadata=True
            )
            
            # Verify multiple sheets were created
            call_args_list = mock_writer.write.call_args_list
            sheet_names = [call[1]['sheet_name'] for call in call_args_list if 'sheet_name' in call[1]]
            
            assert len(sheet_names) >= 2  # At least data sheet and metadata sheet


class TestExportIntegration:
    """Test integration of export functionality."""
    
    def test_full_export_pipeline(self, tmp_path, sample_chunks, sample_articles, user_info):
        """Test complete export pipeline."""
        exporter = ExcelExporter()
        
        # Create validation scores
        validation_scores = [
            ValidationScore(
                completeness_score=85,
                reliability_score=90,
                legal_structure_score=75,
                overall_score=83
            ) for _ in sample_chunks
        ]
        
        output_file = tmp_path / "full_pipeline_test.xlsx"
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = Mock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            # Export complete analysis
            result_file = exporter.export_complete_analysis(
                chunks=sample_chunks,
                articles=sample_articles,
                validation_scores=validation_scores,
                output_file=output_file,
                user_info=user_info
            )
            
            assert result_file == output_file
            assert mock_excel_writer.called
    
    def test_export_with_custom_formatting(self, tmp_path, sample_chunks):
        """Test export with custom formatting options."""
        exporter = ExcelExporter()
        
        custom_format = {
            'header_style': {'bold': True, 'bg_color': '#4CAF50'},
            'data_style': {'font_size': 10},
            'column_widths': {'text': 50, 'chunk_index': 15}
        }
        
        output_file = tmp_path / "custom_format_test.xlsx"
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = Mock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            exporter.export_chunks(
                chunks=sample_chunks,
                output_file=output_file,
                format_options=custom_format
            )
            
            assert mock_excel_writer.called
    
    def test_concurrent_exports(self, tmp_path, sample_chunks):
        """Test handling of concurrent export operations."""
        exporter = ExcelExporter()
        
        # Simulate multiple concurrent exports
        files = [tmp_path / f"concurrent_test_{i}.xlsx" for i in range(3)]
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = Mock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            # Export to multiple files
            for output_file in files:
                exporter.export_chunks(
                    chunks=sample_chunks,
                    output_file=output_file
                )
            
            # Should handle all exports successfully
            assert mock_excel_writer.call_count == len(files)
    
    def test_export_file_overwrite(self, tmp_path, sample_chunks):
        """Test overwriting existing export files."""
        exporter = ExcelExporter()
        output_file = tmp_path / "overwrite_test.xlsx"
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = Mock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            # First export
            exporter.export_chunks(
                chunks=sample_chunks,
                output_file=output_file
            )
            
            # Second export (should overwrite)
            exporter.export_chunks(
                chunks=sample_chunks[:1],  # Different data
                output_file=output_file
            )
            
            # Should have called Excel writer twice
            assert mock_excel_writer.call_count == 2


class TestExportUtilities:
    """Test export utility functions."""
    
    def test_sanitize_sheet_name(self):
        """Test sheet name sanitization."""
        exporter = ExcelExporter()
        
        # Test various problematic characters
        test_cases = [
            ("Valid Name", "Valid Name"),
            ("Name/With\\Slash", "Name_With_Slash"),
            ("Name:With*Special?Chars", "Name_With_Special_Chars"),
            ("Very Long Sheet Name That Exceeds Excel Limits" * 2, "Very Long Sheet Name That Exceeds Excel Limits"[:31]),
            ("", "Sheet1"),
            ("   ", "Sheet1")
        ]
        
        for input_name, expected in test_cases:
            result = exporter._sanitize_sheet_name(input_name)
            assert len(result) <= 31  # Excel limit
            assert not any(char in result for char in ['/', '\\', ':', '*', '?', '[', ']'])
    
    def test_format_timestamp(self):
        """Test timestamp formatting for filenames."""
        exporter = ExcelExporter()
        
        timestamp = exporter._format_timestamp()
        
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0
        # Should be in a format suitable for filenames
        assert not any(char in timestamp for char in [':', '*', '?', '<', '>', '|'])
    
    def test_calculate_column_widths(self, sample_chunks):
        """Test automatic column width calculation."""
        formatter = ExcelFormatter()
        
        df = formatter.format_chunks(sample_chunks)
        widths = formatter._calculate_column_widths(df)
        
        assert isinstance(widths, dict)
        assert all(isinstance(width, (int, float)) for width in widths.values())
        assert all(width > 0 for width in widths.values())
    
    def test_memory_efficient_export(self, tmp_path):
        """Test memory-efficient export for large datasets."""
        exporter = ExcelExporter()
        
        # Simulate very large dataset
        def chunk_generator():
            for i in range(10000):
                yield {
                    'chunk_index': i,
                    'text': f'Chunk {i} content',
                    'text_length': 15,
                    'word_count': 3,
                    'chunk_id': f'chunk_{i}'
                }
        
        output_file = tmp_path / "memory_test.xlsx"
        
        with patch('pandas.ExcelWriter') as mock_excel_writer:
            mock_writer = Mock()
            mock_excel_writer.return_value.__enter__.return_value = mock_writer
            
            # Should handle large dataset efficiently
            exporter.export_chunks_generator(
                chunk_generator(),
                output_file=output_file,
                batch_size=1000
            )
            
            assert mock_excel_writer.called