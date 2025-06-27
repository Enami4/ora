"""
AI-powered validation module with multi-dimensional scoring and materiality assessment.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import anthropic
from datetime import datetime
from .caching import cache_result, get_cache
from .error_recovery import with_retry, with_fallback, get_error_manager, safe_execute

logger = logging.getLogger(__name__)


class MaterialityLevel(Enum):
    """Materiality levels for regulatory articles."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class ValidationScore:
    """Multi-dimensional validation scores."""
    completeness: float  # 0-100
    reliability: float   # 0-100
    legal_structure: float  # 0-100
    overall: float  # 0-100
    details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'completeness_score': self.completeness,
            'reliability_score': self.reliability,
            'legal_structure_score': self.legal_structure,
            'overall_score': self.overall,
            'validation_details': self.details
        }


@dataclass
class Article:
    """Represents a regulatory article."""
    number: str
    title: Optional[str]
    content: str
    materiality: MaterialityLevel
    materiality_reasoning: str
    context: Dict[str, Any]
    validation_score: Optional[ValidationScore] = None


class AIValidator:
    """AI-powered document validation with Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-haiku-20240307"):
        """
        Initialize the AI validator.
        
        Args:
            api_key: Anthropic API key (reads from env if not provided)
            model: Claude model to use
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Claude API: {e}")
        
        # Prompt templates
        self.validation_prompt = self._load_validation_prompt()
        self.materiality_prompt = self._load_materiality_prompt()
        self.article_extraction_prompt = self._load_article_extraction_prompt()
    
    def _load_validation_prompt(self) -> str:
        """Load validation prompt template."""
        return """You are an expert regulatory document validator. Analyze the following text chunk and provide multi-dimensional scoring.

Text Chunk:
{chunk_text}

Document Context:
- Document Type: {document_type}
- File Name: {file_name}
- Chunk Index: {chunk_index}
- Total Chunks: {total_chunks}

Evaluate the chunk on these dimensions:

1. COMPLETENESS SCORE (0-100):
   - Are sentences complete and not cut off mid-way?
   - Are regulatory references intact?
   - Is the chunk boundary appropriate?
   - Does it maintain semantic coherence?

2. RELIABILITY SCORE (0-100):
   - Text clarity and readability
   - Absence of OCR errors or encoding issues
   - Proper formatting preservation
   - Consistent language and terminology

3. LEGAL STRUCTURE SCORE (0-100):
   - Compliance with regulatory document format
   - Proper article/section numbering
   - Hierarchical structure preservation
   - Legal language appropriateness

Provide your analysis in the following JSON format:
{
    "completeness_score": <0-100>,
    "completeness_issues": ["list of specific issues if any"],
    "reliability_score": <0-100>,
    "reliability_issues": ["list of specific issues if any"],
    "legal_structure_score": <0-100>,
    "legal_structure_issues": ["list of specific issues if any"],
    "overall_score": <weighted average>,
    "recommendations": ["list of recommendations for improvement"],
    "chunk_quality": "EXCELLENT|GOOD|FAIR|POOR"
}"""
    
    def _load_materiality_prompt(self) -> str:
        """Load materiality assessment prompt template."""
        return """You are a regulatory compliance expert. Assess the materiality level of the following regulatory article.

Article Content:
{article_content}

Context:
- Regulation: {regulation_name}
- Document Type: {document_type}
- Article Number: {article_number}

Assess the materiality based on:
1. Financial impact potential
2. Operational impact on banking institutions
3. Compliance requirements and penalties
4. Risk management implications
5. Stakeholder impact (customers, shareholders, regulators)

Classify the materiality as:
- LOW: Minimal impact, procedural or administrative
- MEDIUM: Moderate impact, standard compliance requirements
- HIGH: Significant impact, critical compliance or risk management
- CRITICAL: Severe impact, fundamental to banking operations or systemic risk

Provide your assessment in JSON format:
{
    "materiality_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "reasoning": "Detailed explanation of the materiality assessment",
    "key_impacts": ["list of primary impact areas"],
    "compliance_urgency": "IMMEDIATE|SHORT_TERM|MEDIUM_TERM|LONG_TERM",
    "stakeholders_affected": ["list of affected stakeholder groups"]
}"""
    
    def _load_article_extraction_prompt(self) -> str:
        """Load improved article extraction prompt template."""
        return """You are an expert in regulatory document analysis specializing in COBAC and CEMAC banking regulations. Extract ALL articles from the following regulatory text.

IMPORTANT INSTRUCTIONS:
1. Look for ANY mention of "Article" followed by a number (e.g., Article 1, Article 25, L'article 10)
2. Articles may be referenced in various ways: "Article XX", "ARTICLE XX", "Art. XX", "L'article XX"
3. Extract the COMPLETE content of each article until you reach the next article
4. Include sub-articles (e.g., 1.1, 1.2) as part of the main article
5. DO NOT miss any articles - be thorough!

Text:
{text}

Document Context:
- Regulation Name: {regulation_name}
- Document Type: {document_type}

Extract each article with:
1. Article number (exactly as it appears, e.g., "1", "25", "10.1")
2. Article title (if explicitly stated after the number)
3. COMPLETE article content (everything until the next article starts)
4. Any referenced articles or regulations within the content

Format the output as JSON:
{
    "articles": [
        {
            "number": "1",
            "title": "Article title if present, null otherwise",
            "content": "Complete article text including all paragraphs, sub-points, and details",
            "references": ["list of other articles referenced"],
            "has_sub_articles": true/false
        }
    ],
    "extraction_confidence": "HIGH|MEDIUM|LOW",
    "total_articles": <count>,
    "regulation_type": "Règlement COBAC|Règlement CEMAC|Other"
}"""
    
    @with_retry(max_retries=2, base_delay=2.0, retry_exceptions=(ConnectionError, TimeoutError))
    def validate_chunk(self, chunk: Dict[str, Any], document_metadata: Dict[str, Any]) -> ValidationScore:
        """
        Validate a text chunk using AI.
        
        Args:
            chunk: Chunk dictionary with text and metadata
            document_metadata: Document-level metadata
            
        Returns:
            ValidationScore object
        """
        if not self.client:
            # Return default scores if no API client
            return self._get_default_validation_score()
        
        error_manager = get_error_manager()
        
        try:
            prompt = self.validation_prompt.format(
                chunk_text=chunk.get('text', ''),
                document_type=document_metadata.get('document_type', 'UNKNOWN'),
                file_name=document_metadata.get('file_name', ''),
                chunk_index=chunk.get('chunk_index', 0),
                total_chunks=document_metadata.get('total_chunks', 0)
            )
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.1,
                system="You are an expert regulatory document validator. Provide precise JSON-formatted assessments.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response with error recovery
            result = safe_execute(
                self._parse_json_response, 
                response.content[0].text,
                default_return={
                    'completeness_score': 75.0,
                    'reliability_score': 75.0, 
                    'legal_structure_score': 75.0,
                    'overall_score': 75.0
                }
            )
            
            return ValidationScore(
                completeness=result.get('completeness_score', 75.0),
                reliability=result.get('reliability_score', 75.0),
                legal_structure=result.get('legal_structure_score', 75.0),
                overall=result.get('overall_score', 75.0),
                details=result
            )
            
        except Exception as e:
            error_manager._record_error(e, "validate_chunk", context={
                'chunk_index': chunk.get('chunk_index', 0),
                'document_type': document_metadata.get('document_type', 'UNKNOWN'),
                'has_client': bool(self.client)
            })
            logger.error(f"AI validation failed: {e}")
            return self._get_default_validation_score()
    
    @cache_result("article_extraction", ttl_hours=24)
    def extract_articles(self, text: str, document_metadata: Dict[str, Any]) -> List[Article]:
        """
        Extract articles from regulatory text with improved detection.
        
        Args:
            text: Full document text
            document_metadata: Document metadata
            
        Returns:
            List of Article objects
        """
        error_manager = get_error_manager()
        
        # First try regex-based extraction with improved patterns
        articles = safe_execute(
            self._extract_articles_regex,
            text, document_metadata,
            default_return=[]
        )
        
        # If few articles found, try sectioned approach
        if len(articles) < 5 and len(text) > 1000:
            logger.info("Few articles found, trying sectioned extraction approach")
            section_articles = safe_execute(
                self._extract_articles_by_sections,
                text, document_metadata,
                default_return=[]
            )
            articles.extend(section_articles)
        
        # Remove duplicates based on article number
        unique_articles = {}
        for article in articles:
            try:
                key = self._parse_article_number(article.number)
                if key not in unique_articles or len(article.content) > len(unique_articles[key].content):
                    unique_articles[key] = article
            except Exception as e:
                logger.warning(f"Failed to parse article number for {article.number}: {e}")
                # Use article number as key if parsing fails
                unique_articles[article.number] = article
        
        articles = list(unique_articles.values())
        articles.sort(key=lambda x: safe_execute(self._parse_article_number, x.number, default_return=(9999, 0, 0)))
        
        # Then enhance with AI if available
        if self.client and len(text) < 50000:  # Limit for API calls
            try:
                enhanced_articles = safe_execute(
                    self._enhance_articles_with_ai,
                    text, articles, document_metadata,
                    default_return=articles
                )
                if enhanced_articles:
                    articles = enhanced_articles
            except Exception as e:
                error_manager._record_error(e, "_enhance_articles_with_ai")
                logger.warning(f"AI article enhancement failed: {e}")
        
        logger.info(f"Extracted {len(articles)} articles from document")
        return articles
    
    def _extract_articles_regex(self, text: str, metadata: Dict[str, Any]) -> List[Article]:
        """Extract articles using improved regex patterns."""
        articles = []
        seen_articles = set()  # Track already found articles
        
        # Enhanced patterns for COBAC document article detection
        patterns = [
            # Pattern 1: Article premier (First article)
            r'Article\s+premier\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Article\s+(?:\d+|premier))[\s\S]){20,3000})',
            # Pattern 2: Article 1er (Article 1st)
            r'Article\s+(\d+)er\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Article\s+\d+)[\s\S]){20,3000})',
            # Pattern 3: Standard Article XX with various separators
            r'Article\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?(?:\s*bis|\s*ter|\s*quater)?)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Article\s+\d+)[\s\S]){20,3000})',
            # Pattern 4: ARTICLE XX (uppercase)
            r'ARTICLE\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?(?:\s*bis|\s*ter|\s*quater)?)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!ARTICLE\s+\d+)[\s\S]){20,3000})',
            # Pattern 5: Art. XX format
            r'Art\.\s*(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?(?:\s*bis|\s*ter|\s*quater)?)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Art\.\s*\d+)[\s\S]){20,3000})',
            # Pattern 6: Article with space before colon (COBAC format)
            r'Article\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?)\s+:\s*([^\n]{0,200})?\n?((?:(?!Article\s+\d+)[\s\S]){20,3000})',
            # Pattern 7: Article with period-dash (COBAC format)
            r'Article\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?)\s*\.-\s*([^\n]{0,200})?\n?((?:(?!Article\s+\d+)[\s\S]){20,3000})',
            # Pattern 8: Article with underscore format
            r'Article_(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?)\s*\.-?\s*([^\n]{0,200})?\n?((?:(?!Article_\d+)[\s\S]){20,3000})',
            # Pattern 9: Roman numerals for articles
            r'Article\s+([IVX]+)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Article\s+[IVX]+)[\s\S]){20,3000})',
            # Pattern 10: Article in context (e.g., "L'article XX stipule")
            r"[Ll]['']article\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?)\s+(?:stipule|dispose|prévoit|énonce|précise)?[:\s]+([^\n]{0,200})?\n?((?:(?![Ll]['']article\s+\d+)[\s\S]){20,3000})",
            # Pattern 11: Standalone numbered sections with dash
            r'(?:^|\n)\s*(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?)\s*[-–—]\s*([^\n]{0,200})?\n?((?:(?!^\s*\d+\s*[-–—])[\s\S]){20,3000})',
            # Pattern 12: Account modification pattern (COBAC specific)
            r'Le\s+compte\s+divisionnaire\s+«\s*(\d+[-\s]*[^»]*)\s*»\s*([^\n]{0,200})?\n?((?:(?!Le\s+compte\s+divisionnaire)[\s\S]){20,3000})',
            # Pattern 13: Paragraph format with parentheses
            r'(\d+)\)\s*([^\n]{0,200})?\n?((?:(?!\d+\))[\s\S]){20,3000})'
        ]
        
        # Determine regulation type from filename or content
        filename = metadata.get('file_name', '')
        regulation_type = 'UNKNOWN'
        if 'COBAC' in filename.upper() or 'COBAC' in text[:1000].upper():
            regulation_type = 'Règlement COBAC'
        elif 'CEMAC' in filename.upper() or 'CEMAC' in text[:1000].upper():
            regulation_type = 'Règlement CEMAC'
        
        context = {
            'regulation_name': metadata.get('file_name', '').replace('.pdf', ''),
            'regulation_type': regulation_type,
            'document_type': metadata.get('document_type', 'UNKNOWN'),
            'extraction_method': 'regex'
        }
        
        for pattern_idx, pattern in enumerate(patterns):
            matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                # Handle different pattern formats
                if pattern_idx == 0:  # Article premier
                    article_number = "premier"
                    article_title = match.group(1).strip() if match.group(1) else None
                    article_content = match.group(2).strip()
                elif pattern_idx == 11:  # Account modification pattern
                    article_number = f"Compte {match.group(1).strip()}"
                    article_title = match.group(2).strip() if match.group(2) else None
                    article_content = match.group(3).strip()
                else:  # Standard patterns
                    article_number = match.group(1).strip()
                    article_title = match.group(2).strip() if match.group(2) else None
                    article_content = match.group(3).strip()
                
                # Normalize article number (replace comma with dot)
                article_number = article_number.replace(',', '.')
                
                # Create unique identifier including pattern type
                article_key = f"{article_number}_{pattern_idx}"
                
                # Skip if already processed (but allow same number from different patterns)
                if article_key in seen_articles:
                    continue
                seen_articles.add(article_key)
                
                # Clean up content
                article_content = re.sub(r'\n{3,}', '\n\n', article_content)
                article_content = re.sub(r'\s+', ' ', article_content)
                article_content = article_content.strip()
                
                # Extract until next article or significant break
                next_article_pos = float('inf')
                next_patterns = [
                    r'Article\s+(?:\d+|premier)', r'ARTICLE\s+\d+', r'Art\.\s+\d+', 
                    r'^\d+\s*[-–—]', r'Le\s+compte\s+divisionnaire'
                ]
                for next_pattern in next_patterns:
                    next_match = re.search(next_pattern, article_content, re.MULTILINE | re.IGNORECASE)
                    if next_match:
                        next_article_pos = min(next_article_pos, next_match.start())
                
                if next_article_pos < float('inf'):
                    article_content = article_content[:next_article_pos].strip()
                
                # Minimum content length check (more lenient for account modifications)
                min_length = 30 if pattern_idx == 11 else 50
                
                if article_content and len(article_content) > min_length:
                    # Determine article format for display
                    if pattern_idx == 0:  # Article premier
                        display_number = "Article premier"
                    elif pattern_idx == 11:  # Account modification
                        display_number = article_number
                    else:
                        display_number = f"Article {article_number}"
                    
                    article = Article(
                        number=display_number,
                        title=article_title,
                        content=article_content,
                        materiality=MaterialityLevel.MEDIUM,  # Default
                        materiality_reasoning="Pending AI assessment",
                        context=context
                    )
                    articles.append(article)
        
        # Sort articles by number
        articles.sort(key=lambda x: self._parse_article_number(x.number))
        
        return articles
    
    def _parse_article_number(self, article_number: str) -> tuple:
        """Parse article number for sorting (e.g., "12-1" -> (12, 1))."""
        import re
        
        # Remove 'Article' prefix if present
        number_str = article_number.replace('Article', '').strip()
        
        # Handle various formats: 12, 12-1, 12 bis, 12.1, etc.
        match = re.match(r'(\d+)(?:[-.](\d+))?(?:\s*(bis|ter|quater))?', number_str, re.IGNORECASE)
        if match:
            main_num = int(match.group(1))
            sub_num = int(match.group(2)) if match.group(2) else 0
            suffix = match.group(3)
            suffix_val = 0
            if suffix:
                suffix_map = {'bis': 1, 'ter': 2, 'quater': 3}
                suffix_val = suffix_map.get(suffix.lower(), 0)
            return (main_num, sub_num, suffix_val)
        
        # If parsing fails, return a high value to put it at the end
        return (9999, 0, 0)
    
    def _enhance_articles_with_ai(self, text: str, 
                                 regex_articles: List[Article], 
                                 metadata: Dict[str, Any]) -> List[Article]:
        """Enhance article extraction with AI."""
        if not self.client:
            return regex_articles
        
        try:
            prompt = self.article_extraction_prompt.format(
                text=text[:10000],  # Limit text length
                regulation_name=metadata.get('file_name', ''),
                document_type=metadata.get('document_type', '')
            )
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.1,
                system="You are an expert in regulatory document analysis. Extract articles precisely.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = self._parse_json_response(response.content[0].text)
            ai_articles = result.get('articles', [])
            
            # Merge AI results with regex results
            enhanced_articles = []
            for ai_art in ai_articles:
                article = Article(
                    number=ai_art.get('number', ''),
                    title=ai_art.get('title'),
                    content=ai_art.get('content', ''),
                    materiality=MaterialityLevel.MEDIUM,
                    materiality_reasoning="Pending assessment",
                    context={
                        'regulation_name': metadata.get('file_name', ''),
                        'document_type': metadata.get('document_type', ''),
                        'extraction_method': 'ai_enhanced'
                    }
                )
                enhanced_articles.append(article)
            
            return enhanced_articles if enhanced_articles else regex_articles
            
        except Exception as e:
            logger.error(f"AI article extraction failed: {e}")
            return regex_articles
    
    def assess_materiality(self, article: Article) -> Article:
        """
        Assess the materiality of an article.
        
        Args:
            article: Article object to assess
            
        Returns:
            Article with updated materiality assessment
        """
        if not self.client:
            return self._assess_materiality_rules_based(article)
        
        try:
            prompt = self.materiality_prompt.format(
                article_content=article.content[:2000],  # Limit content
                regulation_name=article.context.get('regulation_name', ''),
                document_type=article.context.get('document_type', ''),
                article_number=article.number
            )
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                temperature=0.2,
                system="You are a regulatory compliance expert. Provide precise materiality assessments.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = self._parse_json_response(response.content[0].text)
            
            article.materiality = MaterialityLevel[result.get('materiality_level', 'MEDIUM')]
            article.materiality_reasoning = result.get('reasoning', '')
            
            return article
            
        except Exception as e:
            logger.error(f"AI materiality assessment failed: {e}")
            return self._assess_materiality_rules_based(article)
    
    def _assess_materiality_rules_based(self, article: Article) -> Article:
        """Rules-based materiality assessment as fallback."""
        content_lower = article.content.lower()
        
        # High materiality keywords
        high_keywords = [
            'capital', 'risque', 'sanction', 'pénalité', 'obligation', 
            'interdiction', 'limite', 'ratio', 'prudentiel', 'systémique',
            'blanchiment', 'terrorisme', 'compliance', 'gouvernance'
        ]
        
        # Critical materiality keywords
        critical_keywords = [
            'dissolution', 'retrait d\'agrément', 'suspension', 'fermeture',
            'amende', 'emprisonnement', 'pénal', 'criminel'
        ]
        
        # Check for keywords
        if any(keyword in content_lower for keyword in critical_keywords):
            article.materiality = MaterialityLevel.CRITICAL
            article.materiality_reasoning = "Contains critical regulatory terms"
        elif any(keyword in content_lower for keyword in high_keywords):
            article.materiality = MaterialityLevel.HIGH
            article.materiality_reasoning = "Contains high-impact regulatory terms"
        elif len(article.content) > 500:
            article.materiality = MaterialityLevel.MEDIUM
            article.materiality_reasoning = "Substantial regulatory content"
        else:
            article.materiality = MaterialityLevel.LOW
            article.materiality_reasoning = "Administrative or procedural content"
        
        return article
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from AI model with enhanced error handling."""
        # Log raw response for debugging
        logger.debug(f"Raw AI response: {response_text[:200]}...")
        
        # Try multiple parsing strategies
        parsing_strategies = [
            self._extract_markdown_json,
            self._extract_raw_json,
            self._extract_labeled_json,
            self._extract_key_values_fallback
        ]
        
        for strategy in parsing_strategies:
            try:
                result = strategy(response_text)
                if result and self._validate_json_structure(result):
                    return result
            except Exception as e:
                logger.debug(f"Parsing strategy {strategy.__name__} failed: {e}")
                continue
        
        logger.error(f"All JSON parsing strategies failed for response: {response_text[:500]}...")
        return {}
    
    def _extract_markdown_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from markdown code blocks."""
        pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            json_text = match.group(1).strip()
            return json.loads(self._clean_json_text(json_text))
        raise ValueError("No markdown JSON found")
    
    def _extract_raw_json(self, text: str) -> Dict[str, Any]:
        """Extract raw JSON from text."""
        # Find JSON object boundaries
        pattern = r'\{[\s\S]*\}'
        match = re.search(pattern, text)
        if match:
            json_text = match.group(0)
            return json.loads(self._clean_json_text(json_text))
        raise ValueError("No raw JSON found")
    
    def _extract_labeled_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON that follows labels like 'json:' or 'response:'."""
        patterns = [
            r'(?:json|response|result)\s*:\s*(\{[\s\S]*\})',
            r'(?:json|response|result)\s*=\s*(\{[\s\S]*\})',
            r'(?:json|response|result)\s*-\s*(\{[\s\S]*\})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                json_text = match.group(1)
                return json.loads(self._clean_json_text(json_text))
        
        raise ValueError("No labeled JSON found")
    
    def _clean_json_text(self, json_text: str) -> str:
        """Clean common JSON formatting issues."""
        # Remove trailing commas
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # Fix common quote issues
        json_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_text)  # Unquoted keys
        
        # Remove comments
        json_text = re.sub(r'//.*?$', '', json_text, flags=re.MULTILINE)
        json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
        
        return json_text.strip()
    
    def _extract_key_values_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback: extract key-value pairs manually."""
        result = {}
        
        # Common patterns for validation scores
        patterns = {
            'completeness_score': r'(?:completeness[_\s]*score|completeness)[:\s]*(\d+(?:\.\d+)?)',
            'reliability_score': r'(?:reliability[_\s]*score|reliability)[:\s]*(\d+(?:\.\d+)?)',
            'legal_structure_score': r'(?:legal[_\s]*structure[_\s]*score|legal[_\s]*structure)[:\s]*(\d+(?:\.\d+)?)',
            'overall_score': r'(?:overall[_\s]*score|overall)[:\s]*(\d+(?:\.\d+)?)',
            'overall': r'(?:^|\s)overall[:\s]*(\d+(?:\.\d+)?)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                try:
                    result[key] = float(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        # If we found any scores, create a minimal valid structure
        if result:
            return {
                'completeness': result.get('completeness_score', result.get('overall', 75.0)),
                'reliability': result.get('reliability_score', result.get('overall', 75.0)),
                'legal_structure': result.get('legal_structure_score', result.get('overall', 75.0)),
                'overall': result.get('overall_score', result.get('overall', 75.0)),
                'details': {
                    'method': 'fallback_extraction',
                    'timestamp': datetime.now().isoformat(),
                    'extracted_values': len(result)
                }
            }
        
        raise ValueError("No extractable values found")
    
    def _validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """Validate that the JSON has required fields."""
        required_fields = ['completeness', 'reliability', 'legal_structure', 'overall']
        
        # Check if it's a validation score structure
        if all(field in data for field in required_fields):
            return True
        
        # Check if it has at least some score fields
        score_fields = [field for field in required_fields if field in data]
        if len(score_fields) >= 2:
            return True
        
        # Check if it's article extraction response
        if 'articles' in data or 'materiality' in data:
            return True
        
        return False
    
    def _get_default_validation_score(self) -> ValidationScore:
        """Get default validation score when AI is not available."""
        return ValidationScore(
            completeness=75.0,
            reliability=75.0,
            legal_structure=75.0,
            overall=75.0,
            details={
                'method': 'rules_based',
                'timestamp': datetime.now().isoformat()
            }
        )
    
    def extract_document_structure(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract COBAC document structure including sections and hierarchies.
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            Dictionary containing document structure information
        """
        structure = {
            'document_type': self._identify_document_type(text, metadata),
            'sections': self._extract_document_sections(text),
            'hierarchy': self._extract_document_hierarchy(text),
            'metadata_extracted': self._extract_document_metadata(text),
            'references': self._extract_regulatory_references(text)
        }
        
        logger.info(f"Extracted document structure: {structure['document_type']} with {len(structure['sections'])} sections")
        return structure
    
    def _identify_document_type(self, text: str, metadata: Dict[str, Any]) -> str:
        """Identify the type of COBAC/CEMAC document."""
        filename = metadata.get('file_name', '').upper()
        text_upper = text[:2000].upper()
        
        # Document type patterns
        if any(pattern in filename for pattern in ['R-', 'REGLEMENT', 'RÈGLEMENT']):
            if 'COBAC' in filename or 'COBAC' in text_upper:
                return 'Règlement COBAC'
            elif 'CEMAC' in filename or 'CEMAC' in text_upper:
                return 'Règlement CEMAC'
            else:
                return 'Règlement'
        elif any(pattern in filename for pattern in ['I-', 'INSTRUCTION']):
            if 'COBAC' in filename or 'COBAC' in text_upper:
                return 'Instruction COBAC'
            elif 'CEMAC' in filename or 'CEMAC' in text_upper:
                return 'Instruction CEMAC'
            else:
                return 'Instruction'
        elif any(pattern in filename for pattern in ['C-', 'CIRCULAIRE']):
            return 'Circulaire'
        elif any(pattern in filename for pattern in ['D-', 'DECISION', 'DÉCISION']):
            return 'Décision'
        else:
            # Analyze content for type identification
            if 'règlement' in text_upper:
                return 'Règlement'
            elif 'instruction' in text_upper:
                return 'Instruction'
            elif 'circulaire' in text_upper:
                return 'Circulaire'
            elif 'décision' in text_upper:
                return 'Décision'
            else:
                return 'Document réglementaire'
    
    def _extract_document_sections(self, text: str) -> List[Dict[str, Any]]:
        """Extract main document sections (Header, Preamble, DECIDE, Articles, Signature)."""
        sections = []
        
        # Section patterns for COBAC documents
        section_patterns = [
            # Header section
            {
                'name': 'Header',
                'pattern': r'^(.*?(?:COMMISSION\s+BANCAIRE|COBAC|CEMAC).*?)(?=\n\s*\n|\n(?:VU|CONSIDÉRANT|DECIDE|Article))',
                'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE
            },
            # Preamble section (VU, CONSIDÉRANT)
            {
                'name': 'Preamble',
                'pattern': r'((?:VU|CONSIDÉRANT|AYANT\s+ÉGARD).*?)(?=\n\s*\n\s*DECIDE|Article\s+premier|Article\s+1)',
                'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE
            },
            # DECIDE section
            {
                'name': 'Decide',
                'pattern': r'(DECIDE\s*:?\s*\n.*?)(?=Article\s+(?:premier|\d+)|TITRE|CHAPITRE)',
                'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE
            },
            # Articles section (main body)
            {
                'name': 'Articles',
                'pattern': r'((?:Article\s+(?:premier|\d+).*?)(?=\n\s*\n\s*(?:Fait|Le\s+Secrétaire|ANNEXE|$)))',
                'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE
            },
            # Signature section
            {
                'name': 'Signature',
                'pattern': r'((?:Fait|Le\s+Secrétaire|Le\s+Président).*?)$',
                'flags': re.MULTILINE | re.DOTALL | re.IGNORECASE
            }
        ]
        
        for section_info in section_patterns:
            matches = re.finditer(section_info['pattern'], text, section_info['flags'])
            for match in matches:
                content = match.group(1).strip()
                if content and len(content) > 10:  # Minimum content length
                    sections.append({
                        'name': section_info['name'],
                        'content': content,
                        'start_position': match.start(),
                        'end_position': match.end(),
                        'word_count': len(content.split())
                    })
                    break  # Take first match for each section
        
        return sections
    
    def _extract_document_hierarchy(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract hierarchical structure (TITRE → CHAPITRE → Section → Article)."""
        hierarchy = {
            'titres': [],
            'chapitres': [],
            'sections': [],
            'articles': []
        }
        
        # Hierarchy patterns
        hierarchy_patterns = {
            'titres': [
                r'TITRE\s+([IVX]+|PREMIER|\d+)\s*[:.-]?\s*([^\n]+)',
                r'TITRE\s+([IVX]+|PREMIER|\d+)\s*\n\s*([^\n]+)'
            ],
            'chapitres': [
                r'CHAPITRE\s+([IVX]+|PREMIER|\d+)\s*[:.-]?\s*([^\n]+)',
                r'Chapitre\s+([IVX]+|premier|\d+)\s*[:.-]?\s*([^\n]+)'
            ],
            'sections': [
                r'Section\s+([IVX]+|\d+)\s*[:.-]?\s*([^\n]+)',
                r'SECTION\s+([IVX]+|\d+)\s*[:.-]?\s*([^\n]+)'
            ]
        }
        
        for category, patterns in hierarchy_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    number = match.group(1).strip()
                    title = match.group(2).strip() if len(match.groups()) > 1 else ''
                    
                    hierarchy[category].append({
                        'number': number,
                        'title': title,
                        'position': match.start(),
                        'full_match': match.group(0)
                    })
        
        # Sort by position in document
        for category in hierarchy:
            hierarchy[category].sort(key=lambda x: x['position'])
        
        return hierarchy
    
    def _extract_document_metadata(self, text: str) -> Dict[str, Any]:
        """Extract document metadata from content."""
        metadata = {}
        
        # Reference number patterns
        ref_patterns = [
            r'(?:Règlement|Instruction|Circulaire|Décision)\s+(?:n°\s*)?([RIC]-\d{4}[_/-]\d{2,3})',
            r'(?:N°|n°)\s*([RIC]-\d{4}[_/-]\d{2,3})',
            r'([RIC]-\d{4}[_/-]\d{2,3})'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                metadata['reference_number'] = match.group(1)
                break
        
        # Date patterns
        date_patterns = [
            r'du\s+(\d{1,2})\s+(\w+)\s+(\d{4})',
            r'(\d{1,2})[/-](\d{1,2})[/-](\d{4})',
            r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text[:1000], re.IGNORECASE)
            if match:
                metadata['date_raw'] = match.group(0)
                break
        
        # Subject/title extraction
        title_patterns = [
            r'(?:Règlement|Instruction|Circulaire|Décision).*?(?:relatif|relative)\s+(?:à|aux?)\s+([^\n\.]+)',
            r'Objet\s*:\s*([^\n]+)',
            r'portant\s+([^\n\.]+)'
        ]
        
        for pattern in title_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                metadata['subject'] = match.group(1).strip()
                break
        
        # Extract signatory information
        signature_patterns = [
            r'Le\s+Secrétaire\s+Général\s*,?\s*([^\n]+)',
            r'Le\s+Président\s*,?\s*([^\n]+)',
            r'Pour\s+le\s+Secrétaire\s+Général\s*,?\s*([^\n]+)'
        ]
        
        for pattern in signature_patterns:
            match = re.search(pattern, text[-1000:], re.IGNORECASE)
            if match:
                metadata['signatory'] = match.group(1).strip()
                break
        
        return metadata
    
    def _extract_regulatory_references(self, text: str) -> List[Dict[str, str]]:
        """Extract references to other regulations, laws, and international standards."""
        references = []
        
        # Reference patterns
        ref_patterns = [
            # Other COBAC/CEMAC regulations
            {
                'type': 'Regulation',
                'pattern': r'(?:Règlement|règlement)\s+(?:COBAC|CEMAC)?\s*(?:n°\s*)?([RIC]-\d{4}[_/-]\d{2,3})',
            },
            # Instructions
            {
                'type': 'Instruction',
                'pattern': r'(?:Instruction|instruction)\s+(?:COBAC|CEMAC)?\s*(?:n°\s*)?([RIC]-\d{4}[_/-]\d{2,3})',
            },
            # International standards
            {
                'type': 'Basel',
                'pattern': r'(Bâle\s+[IVX]+|Basel\s+[IVX]+|Accords?\s+de\s+Bâle)',
            },
            # CEMAC treaties
            {
                'type': 'Treaty',
                'pattern': r'(Traité\s+CEMAC|Convention\s+CEMAC|Acte\s+additionnel)',
            },
            # Laws and codes
            {
                'type': 'Law',
                'pattern': r'(Loi\s+n°\s*[^,\n]+|Code\s+[^,\n]+)',
            }
        ]
        
        for ref_info in ref_patterns:
            matches = re.finditer(ref_info['pattern'], text, re.IGNORECASE)
            for match in matches:
                references.append({
                    'type': ref_info['type'],
                    'reference': match.group(1) if len(match.groups()) > 0 else match.group(0),
                    'context': text[max(0, match.start()-50):match.end()+50].strip(),
                    'position': match.start()
                })
        
        # Remove duplicates and sort by position
        unique_refs = []
        seen = set()
        for ref in references:
            key = (ref['type'], ref['reference'])
            if key not in seen:
                seen.add(key)
                unique_refs.append(ref)
        
        unique_refs.sort(key=lambda x: x['position'])
        return unique_refs
    
    def _extract_articles_by_sections(self, text: str, metadata: Dict[str, Any]) -> List[Article]:
        """Extract articles by analyzing document sections."""
        articles = []
        
        # Split text into sections based on common patterns
        section_patterns = [
            r'\n{2,}(?=Article\s+\d+)',
            r'\n{2,}(?=ARTICLE\s+\d+)',
            r'\n{2,}(?=Art\.\s+\d+)',
            r'\n{2,}(?=\d+\s*[-–—])'
        ]
        
        sections = [text]
        for pattern in section_patterns:
            new_sections = []
            for section in sections:
                parts = re.split(pattern, section)
                new_sections.extend(parts)
            sections = new_sections
        
        # Process each section to find articles
        for section in sections:
            if len(section.strip()) < 50:
                continue
                
            # Look for article indicators at the beginning of sections
            article_match = re.match(
                r'^(Article|ARTICLE|Art\.)\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?)\s*[:.-]?\s*([^\n]{0,200})?',
                section.strip(), 
                re.IGNORECASE
            )
            
            if article_match:
                article_number = article_match.group(2).replace(',', '.')
                article_title = article_match.group(3).strip() if article_match.group(3) else None
                
                # Get content after the article header
                content_start = article_match.end()
                article_content = section[content_start:].strip()
                
                if article_content and len(article_content) > 50:
                    context = {
                        'regulation_name': metadata.get('file_name', '').replace('.pdf', ''),
                        'regulation_type': self._determine_regulation_type(text, metadata),
                        'document_type': metadata.get('document_type', 'UNKNOWN'),
                        'extraction_method': 'section_based'
                    }
                    
                    article = Article(
                        number=f"Article {article_number}",
                        title=article_title,
                        content=article_content,
                        materiality=MaterialityLevel.MEDIUM,
                        materiality_reasoning="Pending AI assessment",
                        context=context
                    )
                    articles.append(article)
        
        return articles
    
    def _determine_regulation_type(self, text: str, metadata: Dict[str, Any]) -> str:
        """Determine regulation type from text and metadata."""
        filename = metadata.get('file_name', '')
        if 'COBAC' in filename.upper() or 'COBAC' in text[:1000].upper():
            return 'Règlement COBAC'
        elif 'CEMAC' in filename.upper() or 'CEMAC' in text[:1000].upper():
            return 'Règlement CEMAC'
        return 'UNKNOWN'


class ValidationChain:
    """Prompt reasoning chain for document validation."""
    
    def __init__(self, validator: AIValidator):
        self.validator = validator
        self.chain_steps = [
            self._validate_extraction_quality,
            self._validate_chunk_boundaries,
            self._validate_legal_structure,
            self._assess_document_completeness
        ]
    
    def validate_document(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the full validation chain on a document.
        
        Args:
            document_data: Processed document data
            
        Returns:
            Enhanced document data with validation results
        """
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'chunk_validations': [],
            'article_validations': [],
            'document_validation': {}
        }
        
        # Validate each chunk
        for chunk in document_data.get('chunks', []):
            score = self.validator.validate_chunk(chunk, document_data['metadata'])
            chunk['validation_score'] = score.to_dict()
            validation_results['chunk_validations'].append(score)
        
        # Extract and validate articles
        articles = self.validator.extract_articles(
            document_data.get('cleaned_text', ''),
            document_data['metadata']
        )
        
        # Extract document structure for COBAC documents
        document_structure = self.validator.extract_document_structure(
            document_data.get('cleaned_text', ''),
            document_data['metadata']
        )
        document_data['document_structure'] = document_structure
        
        # Assess materiality for each article
        for article in articles:
            article = self.validator.assess_materiality(article)
            validation_results['article_validations'].append(article)
        
        # Run chain steps
        for step in self.chain_steps:
            step(document_data, validation_results)
        
        # Calculate overall document score
        validation_results['document_validation']['overall_score'] = self._calculate_overall_score(
            validation_results
        )
        
        document_data['validation_results'] = validation_results
        document_data['articles'] = articles
        
        return document_data
    
    def _validate_extraction_quality(self, document_data: Dict[str, Any], 
                                   validation_results: Dict[str, Any]) -> None:
        """Validate overall extraction quality."""
        total_chars = document_data['statistics'].get('total_characters', 0)
        total_words = document_data['statistics'].get('total_words', 0)
        
        quality_score = 100.0
        issues = []
        
        if total_chars == 0:
            quality_score = 0.0
            issues.append("No text extracted")
        elif total_words < 10:
            quality_score = 25.0
            issues.append("Very limited text extracted")
        elif total_words / (total_chars / 5) < 0.5:  # Word/char ratio check
            quality_score -= 25.0
            issues.append("Potential OCR or encoding issues")
        
        validation_results['document_validation']['extraction_quality'] = {
            'score': quality_score,
            'issues': issues
        }
    
    def _validate_chunk_boundaries(self, document_data: Dict[str, Any], 
                                  validation_results: Dict[str, Any]) -> None:
        """Validate chunk boundary integrity."""
        chunk_scores = [v.completeness for v in validation_results['chunk_validations']]
        
        if chunk_scores:
            avg_completeness = sum(chunk_scores) / len(chunk_scores)
            validation_results['document_validation']['chunk_boundary_integrity'] = {
                'average_score': avg_completeness,
                'chunks_below_threshold': sum(1 for s in chunk_scores if s < 70)
            }
    
    def _validate_legal_structure(self, document_data: Dict[str, Any], 
                                 validation_results: Dict[str, Any]) -> None:
        """Validate legal document structure."""
        articles = validation_results.get('article_validations', [])
        
        structure_score = 100.0
        issues = []
        
        if not articles:
            structure_score = 50.0
            issues.append("No articles detected")
        else:
            # Check article numbering sequence
            article_numbers = []
            for article in articles:
                try:
                    num = float(article.number.replace('Article', '').strip())
                    article_numbers.append(num)
                except:
                    pass
            
            if article_numbers:
                article_numbers.sort()
                # Check for gaps
                for i in range(1, len(article_numbers)):
                    if article_numbers[i] - article_numbers[i-1] > 5:
                        structure_score -= 10
                        issues.append(f"Large gap between articles {article_numbers[i-1]} and {article_numbers[i]}")
        
        validation_results['document_validation']['legal_structure'] = {
            'score': max(0, structure_score),
            'issues': issues,
            'total_articles': len(articles)
        }
    
    def _assess_document_completeness(self, document_data: Dict[str, Any], 
                                    validation_results: Dict[str, Any]) -> None:
        """Assess overall document completeness."""
        pages = document_data['metadata'].get('page_count', 0)
        articles = len(validation_results.get('article_validations', []))
        
        completeness_score = 100.0
        
        if pages == 0:
            completeness_score = 0.0
        elif articles == 0 and pages > 5:
            completeness_score = 50.0
        elif articles / pages < 0.5 and document_data['metadata'].get('document_type') == 'REGLEMENT':
            completeness_score = 75.0
        
        validation_results['document_validation']['completeness'] = {
            'score': completeness_score,
            'pages': pages,
            'articles': articles,
            'articles_per_page': articles / pages if pages > 0 else 0
        }
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate weighted overall document score."""
        doc_val = validation_results['document_validation']
        
        weights = {
            'extraction_quality': 0.25,
            'chunk_boundary_integrity': 0.25,
            'legal_structure': 0.30,
            'completeness': 0.20
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for key, weight in weights.items():
            if key in doc_val and 'score' in doc_val[key]:
                total_score += doc_val[key]['score'] * weight
                total_weight += weight
            elif key in doc_val and 'average_score' in doc_val[key]:
                total_score += doc_val[key]['average_score'] * weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _parse_article_number(self, article_str: str) -> float:
        """Parse article number for sorting."""
        try:
            # Extract just the number part
            number_match = re.search(r'(\d+(?:\.\d+)?)', article_str)
            if number_match:
                return float(number_match.group(1))
            return 0.0
        except:
            return 0.0