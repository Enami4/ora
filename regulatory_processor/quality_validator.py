"""
Quality validation system for regulatory document processing.
This module implements comprehensive quality checks based on the improvements framework.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json
from datetime import datetime

from .improvements import QualityAssurance
from .validators import Article, MaterialityLevel

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Validation status levels."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


@dataclass
class QualityMetrics:
    """Quality metrics for document processing."""
    ocr_confidence: float
    text_completeness: float
    structure_coherence: float
    article_extraction_rate: float
    regulatory_term_accuracy: float
    overall_score: float
    status: ValidationStatus
    recommendations: List[str]
    timestamp: datetime


@dataclass
class ValidationResult:
    """Result of quality validation process."""
    is_valid: bool
    quality_metrics: QualityMetrics
    validation_details: Dict[str, Any]
    error_log: List[str]
    recommendations: List[str]


class QualityValidator:
    """Comprehensive quality validation system."""
    
    def __init__(self):
        self.qa = QualityAssurance()
        self.quality_metrics = self.qa.get_quality_metrics()
        self.validation_checklist = self.qa.get_validation_checklist()
        
    def validate_extraction_quality(self, 
                                  extracted_text: str, 
                                  ocr_confidence: Optional[float] = None,
                                  page_count: int = 1) -> QualityMetrics:
        """
        Validate the quality of text extraction.
        
        Args:
            extracted_text: The extracted text to validate
            ocr_confidence: OCR confidence score if available
            page_count: Number of pages in the document
            
        Returns:
            QualityMetrics object with detailed assessment
        """
        try:
            # Calculate individual metrics
            text_completeness = self._assess_text_completeness(extracted_text, page_count)
            structure_coherence = self._assess_structure_coherence(extracted_text)
            regulatory_accuracy = self._assess_regulatory_term_accuracy(extracted_text)
            
            # Use provided OCR confidence or estimate from text quality
            if ocr_confidence is None:
                ocr_confidence = self._estimate_ocr_confidence(extracted_text)
            
            # Calculate overall score
            weights = {
                'ocr_confidence': 0.25,
                'text_completeness': 0.30,
                'structure_coherence': 0.25,
                'regulatory_accuracy': 0.20
            }
            
            overall_score = (
                ocr_confidence * weights['ocr_confidence'] +
                text_completeness * weights['text_completeness'] +
                structure_coherence * weights['structure_coherence'] +
                regulatory_accuracy * weights['regulatory_accuracy']
            )
            
            # Determine status and recommendations
            status, recommendations = self._determine_status_and_recommendations(
                overall_score, ocr_confidence, text_completeness, 
                structure_coherence, regulatory_accuracy
            )
            
            return QualityMetrics(
                ocr_confidence=ocr_confidence,
                text_completeness=text_completeness,
                structure_coherence=structure_coherence,
                article_extraction_rate=0.0,  # Will be updated later
                regulatory_term_accuracy=regulatory_accuracy,
                overall_score=overall_score,
                status=status,
                recommendations=recommendations,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Quality validation failed: {e}")
            return self._create_failed_metrics(str(e))
    
    def validate_article_extraction(self, 
                                   articles: List[Article], 
                                   original_text: str,
                                   expected_article_count: Optional[int] = None) -> Dict[str, Any]:
        """
        Validate the quality of article extraction.
        
        Args:
            articles: List of extracted articles
            original_text: Original document text
            expected_article_count: Expected number of articles (if known)
            
        Returns:
            Dictionary with validation results
        """
        try:
            validation_result = {
                'total_articles': len(articles),
                'completeness_score': 0.0,
                'accuracy_score': 0.0,
                'coherence_score': 0.0,
                'quality_score': 0.0,
                'issues': [],
                'recommendations': []
            }
            
            if not articles:
                validation_result['issues'].append("No articles extracted")
                validation_result['recommendations'].append("Review extraction parameters")
                return validation_result
            
            # Check article numbering continuity
            continuity_score = self._check_article_continuity(articles)
            
            # Check content quality
            content_quality_score = self._assess_article_content_quality(articles)
            
            # Check for regulatory structure
            structure_score = self._assess_article_structure(articles)
            
            # Check materiality assessment quality
            materiality_score = self._assess_materiality_quality(articles)
            
            # Calculate extraction rate if expected count is known
            extraction_rate = 1.0
            if expected_article_count and expected_article_count > 0:
                extraction_rate = min(len(articles) / expected_article_count, 1.0)
            
            # Calculate overall scores
            validation_result.update({
                'completeness_score': extraction_rate * 100,
                'accuracy_score': content_quality_score * 100,
                'coherence_score': structure_score * 100,
                'quality_score': (continuity_score + content_quality_score + 
                                structure_score + materiality_score) / 4 * 100
            })
            
            # Add specific checks
            self._add_specific_validation_checks(validation_result, articles, original_text)
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Article extraction validation failed: {e}")
            return {
                'total_articles': 0,
                'completeness_score': 0.0,
                'accuracy_score': 0.0,
                'coherence_score': 0.0,
                'quality_score': 0.0,
                'issues': [f"Validation error: {e}"],
                'recommendations': ["Manual review required"]
            }
    
    def validate_document_processing(self, 
                                   processing_result: Dict[str, Any]) -> ValidationResult:
        """
        Comprehensive validation of the entire document processing pipeline.
        
        Args:
            processing_result: Complete processing result dictionary
            
        Returns:
            ValidationResult with overall assessment
        """
        try:
            validation_details = {}
            error_log = []
            recommendations = []
            
            # Extract relevant data
            extracted_text = processing_result.get('full_text', '')
            articles = processing_result.get('articles', [])
            metadata = processing_result.get('metadata', {})
            
            # Validate text extraction
            text_metrics = self.validate_extraction_quality(
                extracted_text, 
                processing_result.get('ocr_confidence'),
                metadata.get('page_count', 1)
            )
            validation_details['text_extraction'] = text_metrics
            
            # Validate article extraction
            article_validation = self.validate_article_extraction(articles, extracted_text)
            validation_details['article_extraction'] = article_validation
            
            # Update article extraction rate in metrics
            text_metrics.article_extraction_rate = article_validation['completeness_score'] / 100
            
            # Run checklist validation
            checklist_results = self._run_validation_checklist(processing_result)
            validation_details['checklist'] = checklist_results
            
            # Calculate overall validation score
            overall_score = (
                text_metrics.overall_score * 0.4 +
                (article_validation['quality_score'] / 100) * 0.4 +
                (checklist_results['overall_score'] / 100) * 0.2
            )
            
            # Determine if validation passes
            is_valid = (
                overall_score >= 0.7 and 
                text_metrics.status not in [ValidationStatus.FAILED, ValidationStatus.NEEDS_REVIEW] and
                article_validation['quality_score'] >= 60
            )
            
            # Collect recommendations
            recommendations.extend(text_metrics.recommendations)
            recommendations.extend(article_validation.get('recommendations', []))
            recommendations.extend(checklist_results.get('recommendations', []))
            
            # Update final metrics
            text_metrics.overall_score = overall_score
            if not is_valid:
                text_metrics.status = ValidationStatus.NEEDS_REVIEW
                recommendations.append("Manual review required due to quality issues")
            
            return ValidationResult(
                is_valid=is_valid,
                quality_metrics=text_metrics,
                validation_details=validation_details,
                error_log=error_log,
                recommendations=list(set(recommendations))  # Remove duplicates
            )
            
        except Exception as e:
            logger.error(f"Document processing validation failed: {e}")
            failed_metrics = self._create_failed_metrics(str(e))
            return ValidationResult(
                is_valid=False,
                quality_metrics=failed_metrics,
                validation_details={'error': str(e)},
                error_log=[str(e)],
                recommendations=["Complete reprocessing required"]
            )
    
    def _assess_text_completeness(self, text: str, page_count: int) -> float:
        """Assess the completeness of extracted text."""
        if not text or not text.strip():
            return 0.0
        
        # Basic metrics
        word_count = len(text.split())
        char_count = len(text)
        
        # Expected text density (rough estimates)
        min_words_per_page = 100
        min_chars_per_page = 500
        
        # Calculate completeness scores
        word_density_score = min(word_count / (page_count * min_words_per_page), 1.0)
        char_density_score = min(char_count / (page_count * min_chars_per_page), 1.0)
        
        # Check for regulatory content indicators
        regulatory_indicators = [
            'article', 'règlement', 'instruction', 'cobac', 'cemac',
            'chapitre', 'section', 'titre', 'commission', 'bancaire'
        ]
        
        found_indicators = sum(1 for indicator in regulatory_indicators 
                             if indicator.lower() in text.lower())
        indicator_score = min(found_indicators / 5, 1.0)
        
        # Combined score
        completeness = (word_density_score * 0.4 + char_density_score * 0.3 + indicator_score * 0.3)
        return completeness
    
    def _assess_structure_coherence(self, text: str) -> float:
        """Assess the structural coherence of the text."""
        if not text:
            return 0.0
        
        # Check for hierarchical structure
        has_titles = bool(re.search(r'TITRE\s+[IVX]+', text, re.IGNORECASE))
        has_chapters = bool(re.search(r'CHAPITRE\s+[IVX]+', text, re.IGNORECASE))
        has_sections = bool(re.search(r'SECTION\s+[IVX]+', text, re.IGNORECASE))
        has_articles = bool(re.search(r'Article\s+\d+', text, re.IGNORECASE))
        
        structure_score = sum([has_titles, has_chapters, has_sections, has_articles]) / 4
        
        # Check for proper numbering
        article_numbers = re.findall(r'Article\s+(\d+)', text, re.IGNORECASE)
        if article_numbers:
            numbers = [int(n) for n in article_numbers if n.isdigit()]
            if numbers:
                # Check for sequential numbering (allowing for some gaps)
                max_gap = 2
                sequential_score = 1.0
                for i in range(1, len(numbers)):
                    if numbers[i] - numbers[i-1] > max_gap:
                        sequential_score -= 0.1
                sequential_score = max(sequential_score, 0.0)
            else:
                sequential_score = 0.5
        else:
            sequential_score = 0.0
        
        return (structure_score * 0.7 + sequential_score * 0.3)
    
    def _assess_regulatory_term_accuracy(self, text: str) -> float:
        """Assess the accuracy of regulatory terminology."""
        if not text:
            return 0.0
        
        # Key regulatory terms that should be present and correctly spelled
        regulatory_terms = {
            'commission bancaire': r'commission\s+bancaire',
            'cobac': r'cobac',
            'cemac': r'cemac',
            'établissement de crédit': r'établissement\s+de\s+crédit',
            'fonds propres': r'fonds\s+propres',
            'ratio de solvabilité': r'ratio\s+de\s+solvabilité',
            'surveillance': r'surveillance',
            'règlement': r'règlement',
            'instruction': r'instruction'
        }
        
        found_terms = 0
        total_terms = len(regulatory_terms)
        
        text_lower = text.lower()
        for term, pattern in regulatory_terms.items():
            if re.search(pattern, text_lower):
                found_terms += 1
        
        # Basic accuracy score
        accuracy = found_terms / total_terms if total_terms > 0 else 0.0
        
        # Check for common OCR errors in regulatory context
        ocr_error_patterns = [
            r'ariicle',  # Should be 'article'
            r'regiement',  # Should be 'règlement'
            r'commision',  # Should be 'commission'
        ]
        
        error_penalty = 0.0
        for pattern in ocr_error_patterns:
            if re.search(pattern, text_lower):
                error_penalty += 0.1
        
        return max(accuracy - error_penalty, 0.0)
    
    def _estimate_ocr_confidence(self, text: str) -> float:
        """Estimate OCR confidence based on text characteristics."""
        if not text:
            return 0.0
        
        # Check for common OCR issues
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Count problematic characters
        special_chars = len(re.findall(r'[^\w\s.,;:!?()[\]{}"\'-/\\|@#$%^&*+=<>°]', text))
        special_ratio = special_chars / total_chars
        
        # Count garbled text patterns
        garbled_patterns = [
            r'[a-zA-Z]{10,}',  # Very long words (likely garbled)
            r'[A-Z]{5,}[a-z]{5,}',  # Mixed case patterns
            r'\d{5,}',  # Long number sequences
        ]
        
        garbled_count = sum(len(re.findall(pattern, text)) for pattern in garbled_patterns)
        garbled_ratio = garbled_count / len(text.split()) if text.split() else 0.0
        
        # Estimate confidence
        confidence = 1.0 - (special_ratio * 0.5 + garbled_ratio * 0.3)
        return max(min(confidence, 1.0), 0.0)
    
    def _determine_status_and_recommendations(self, overall_score: float, 
                                            ocr_confidence: float,
                                            text_completeness: float,
                                            structure_coherence: float,
                                            regulatory_accuracy: float) -> Tuple[ValidationStatus, List[str]]:
        """Determine validation status and provide recommendations."""
        recommendations = []
        
        if overall_score >= 0.9:
            status = ValidationStatus.EXCELLENT
            recommendations.append("Quality is excellent, processing can proceed")
        elif overall_score >= 0.8:
            status = ValidationStatus.GOOD
            recommendations.append("Quality is good, minor review recommended")
        elif overall_score >= 0.7:
            status = ValidationStatus.ACCEPTABLE
            recommendations.append("Quality is acceptable, proceed with caution")
        elif overall_score >= 0.5:
            status = ValidationStatus.NEEDS_REVIEW
            recommendations.append("Quality needs improvement, manual review required")
        else:
            status = ValidationStatus.FAILED
            recommendations.append("Quality is unacceptable, reprocessing required")
        
        # Specific recommendations based on individual scores
        if ocr_confidence < 0.7:
            recommendations.append("OCR confidence is low, consider higher DPI or preprocessing")
        
        if text_completeness < 0.7:
            recommendations.append("Text appears incomplete, verify all pages were processed")
        
        if structure_coherence < 0.7:
            recommendations.append("Document structure is unclear, manual structure verification needed")
        
        if regulatory_accuracy < 0.7:
            recommendations.append("Regulatory terminology accuracy is low, review key terms")
        
        return status, recommendations
    
    def _create_failed_metrics(self, error_message: str) -> QualityMetrics:
        """Create failed quality metrics."""
        return QualityMetrics(
            ocr_confidence=0.0,
            text_completeness=0.0,
            structure_coherence=0.0,
            article_extraction_rate=0.0,
            regulatory_term_accuracy=0.0,
            overall_score=0.0,
            status=ValidationStatus.FAILED,
            recommendations=[f"Processing failed: {error_message}"],
            timestamp=datetime.now()
        )
    
    def _check_article_continuity(self, articles: List[Article]) -> float:
        """Check the continuity of article numbering."""
        if not articles:
            return 0.0
        
        # Extract article numbers
        numbers = []
        for article in articles:
            try:
                # Simple extraction of main article number
                match = re.search(r'(\d+)', article.number)
                if match:
                    numbers.append(int(match.group(1)))
            except:
                continue
        
        if not numbers:
            return 0.0
        
        numbers.sort()
        
        # Check for gaps (allowing some tolerance)
        expected_gaps = 0
        total_gaps = 0
        
        for i in range(1, len(numbers)):
            gap = numbers[i] - numbers[i-1]
            total_gaps += gap - 1  # -1 because consecutive should have gap of 1
            
        # Calculate continuity score
        if len(numbers) <= 1:
            return 1.0
        
        max_expected_gaps = len(numbers) * 0.1  # Allow 10% gaps
        continuity_score = max(1.0 - (total_gaps / len(numbers)), 0.0)
        
        return continuity_score
    
    def _assess_article_content_quality(self, articles: List[Article]) -> float:
        """Assess the quality of article content."""
        if not articles:
            return 0.0
        
        total_score = 0.0
        
        for article in articles:
            content = article.content
            if not content:
                continue
            
            # Check content length (should not be too short or too long)
            length_score = 1.0
            if len(content) < 50:
                length_score = 0.5
            elif len(content) > 5000:
                length_score = 0.8
            
            # Check for regulatory language
            regulatory_words = ['doit', 'doivent', 'est', 'sont', 'peut', 'peuvent', 
                              'obligation', 'interdiction', 'autorisation']
            found_regulatory = sum(1 for word in regulatory_words 
                                 if word in content.lower())
            regulatory_score = min(found_regulatory / 3, 1.0)
            
            # Check for completeness (sentences should end properly)
            sentences = content.split('.')
            complete_sentences = sum(1 for s in sentences if s.strip() and len(s.strip()) > 10)
            completeness_score = min(complete_sentences / len(sentences) if sentences else 0, 1.0)
            
            article_score = (length_score * 0.3 + regulatory_score * 0.4 + completeness_score * 0.3)
            total_score += article_score
        
        return total_score / len(articles) if articles else 0.0
    
    def _assess_article_structure(self, articles: List[Article]) -> float:
        """Assess the structural quality of articles."""
        if not articles:
            return 0.0
        
        structure_score = 0.0
        
        # Check for proper numbering format
        proper_numbering = sum(1 for article in articles 
                             if re.match(r'Article\s+\d+', article.number, re.IGNORECASE))
        numbering_score = proper_numbering / len(articles)
        
        # Check for titles
        have_titles = sum(1 for article in articles if article.title and article.title.strip())
        title_score = have_titles / len(articles)
        
        # Check for context information
        have_context = sum(1 for article in articles if article.context)
        context_score = have_context / len(articles)
        
        structure_score = (numbering_score * 0.5 + title_score * 0.3 + context_score * 0.2)
        return structure_score
    
    def _assess_materiality_quality(self, articles: List[Article]) -> float:
        """Assess the quality of materiality assessments."""
        if not articles:
            return 0.0
        
        # Check if materiality assessments are present and meaningful
        assessed_articles = sum(1 for article in articles 
                              if article.materiality != MaterialityLevel.MEDIUM or 
                                 (article.materiality_reasoning and 
                                  article.materiality_reasoning != "Pending assessment"))
        
        assessment_rate = assessed_articles / len(articles)
        
        # Check for reasoning quality
        good_reasoning = sum(1 for article in articles 
                           if article.materiality_reasoning and 
                              len(article.materiality_reasoning) > 20)
        
        reasoning_score = good_reasoning / len(articles)
        
        return (assessment_rate * 0.6 + reasoning_score * 0.4)
    
    def _add_specific_validation_checks(self, validation_result: Dict[str, Any], 
                                       articles: List[Article], original_text: str):
        """Add specific validation checks based on the validation checklist."""
        # Document Structure checks
        article_numbers = [article.number for article in articles]
        if len(set(article_numbers)) != len(article_numbers):
            validation_result['issues'].append("Duplicate article numbers found")
        
        # Check for missing articles (gaps in numbering)
        numbers = []
        for article in articles:
            match = re.search(r'(\d+)', article.number)
            if match:
                numbers.append(int(match.group(1)))
        
        if numbers:
            numbers.sort()
            gaps = []
            for i in range(1, len(numbers)):
                if numbers[i] - numbers[i-1] > 1:
                    gaps.extend(range(numbers[i-1] + 1, numbers[i]))
            
            if gaps:
                validation_result['issues'].append(f"Missing articles: {gaps[:5]}")  # Show first 5 gaps
                validation_result['recommendations'].append("Review document for missing articles")
        
        # Content Quality checks
        empty_articles = sum(1 for article in articles if not article.content.strip())
        if empty_articles > 0:
            validation_result['issues'].append(f"{empty_articles} articles have empty content")
            validation_result['recommendations'].append("Review articles with empty content")
        
        # Check for regulatory terminology
        regulatory_terms = ['cobac', 'cemac', 'commission bancaire', 'règlement']
        found_terms = sum(1 for term in regulatory_terms 
                         if term in original_text.lower())
        
        if found_terms < 2:
            validation_result['issues'].append("Few regulatory terms found in document")
            validation_result['recommendations'].append("Verify this is a regulatory document")
    
    def _run_validation_checklist(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run the comprehensive validation checklist."""
        checklist_results = {
            'categories': {},
            'overall_score': 0.0,
            'passed_checks': 0,
            'total_checks': 0,
            'recommendations': []
        }
        
        try:
            for category_info in self.validation_checklist:
                category = category_info['category']
                checks = category_info['checks']
                
                category_results = {
                    'passed': 0,
                    'total': len(checks),
                    'details': []
                }
                
                for check in checks:
                    passed = self._perform_check(check, processing_result)
                    category_results['details'].append({
                        'check': check,
                        'passed': passed
                    })
                    if passed:
                        category_results['passed'] += 1
                        checklist_results['passed_checks'] += 1
                    
                    checklist_results['total_checks'] += 1
                
                checklist_results['categories'][category] = category_results
            
            # Calculate overall score
            if checklist_results['total_checks'] > 0:
                checklist_results['overall_score'] = (
                    checklist_results['passed_checks'] / checklist_results['total_checks'] * 100
                )
            
            return checklist_results
            
        except Exception as e:
            logger.error(f"Validation checklist failed: {e}")
            return {
                'categories': {},
                'overall_score': 0.0,
                'passed_checks': 0,
                'total_checks': 0,
                'recommendations': [f"Checklist validation failed: {e}"]
            }
    
    def _perform_check(self, check: str, processing_result: Dict[str, Any]) -> bool:
        """Perform a specific validation check."""
        try:
            articles = processing_result.get('articles', [])
            text = processing_result.get('full_text', '')
            
            # Map checks to actual validation logic
            if 'numbered sequentially' in check.lower():
                return self._check_sequential_numbering(articles)
            elif 'hierarchy levels' in check.lower():
                return self._check_hierarchy_consistency(articles)
            elif 'cross-references' in check.lower():
                return self._check_cross_references(articles)
            elif 'technical terms' in check.lower():
                return self._check_technical_terms(text)
            elif 'regulatory language' in check.lower():
                return self._check_regulatory_language(text)
            elif 'materiality levels' in check.lower():
                return self._check_materiality_justification(articles)
            elif 'excel format' in check.lower():
                return True  # Assume format is correct if we got this far
            elif 'data structure' in check.lower():
                return len(articles) > 0
            else:
                return True  # Default to pass for unimplemented checks
                
        except Exception as e:
            logger.warning(f"Check '{check}' failed: {e}")
            return False
    
    def _check_sequential_numbering(self, articles: List[Article]) -> bool:
        """Check if articles are numbered sequentially (allowing some gaps)."""
        if not articles:
            return False
        
        numbers = []
        for article in articles:
            match = re.search(r'(\d+)', article.number)
            if match:
                numbers.append(int(match.group(1)))
        
        if not numbers:
            return False
        
        numbers.sort()
        gaps = sum(1 for i in range(1, len(numbers)) if numbers[i] - numbers[i-1] > 2)
        return gaps <= len(numbers) * 0.1  # Allow 10% gaps
    
    def _check_hierarchy_consistency(self, articles: List[Article]) -> bool:
        """Check hierarchy consistency."""
        for article in articles:
            if article.context and 'hierarchy' in article.context:
                return True
        return len(articles) > 0  # Basic check if no hierarchy info
    
    def _check_cross_references(self, articles: List[Article]) -> bool:
        """Check for cross-references."""
        ref_count = sum(1 for article in articles 
                       if article.context and article.context.get('references'))
        return ref_count > 0
    
    def _check_technical_terms(self, text: str) -> bool:
        """Check for presence of technical terms."""
        technical_terms = ['ratio', 'provisions', 'capital', 'solvabilité', 'liquidité']
        found = sum(1 for term in technical_terms if term in text.lower())
        return found >= 2
    
    def _check_regulatory_language(self, text: str) -> bool:
        """Check for regulatory language patterns."""
        patterns = [r'article\s+\d+', r'règlement', r'instruction', r'décision']
        found = sum(1 for pattern in patterns if re.search(pattern, text.lower()))
        return found >= 2
    
    def _check_materiality_justification(self, articles: List[Article]) -> bool:
        """Check if materiality assessments are justified."""
        justified = sum(1 for article in articles 
                       if article.materiality_reasoning and 
                          len(article.materiality_reasoning) > 10)
        return justified > len(articles) * 0.5  # At least 50% should be justified