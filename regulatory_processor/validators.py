"""
AI-powered validation module with multi-dimensional scoring and materiality assessment.
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import anthropic
from datetime import datetime
try:
    import networkx as nx
except ImportError:
    nx = None
    logging.warning("NetworkX not available. Cross-reference network analysis will be limited.")
try:
    import pandas as pd
    import camelot
    import tabula
except ImportError:
    pd = None
    camelot = None
    tabula = None
    logging.warning("Table extraction libraries not available. PDF table extraction will be limited.")
try:
    import pytesseract
    from PIL import Image
    import cv2
    import numpy as np
except ImportError:
    pytesseract = None
    Image = None
    cv2 = None
    np = None
    logging.warning("OCR libraries not available. Advanced OCR processing will be limited.")
from .caching import cache_result, get_cache
from .error_recovery import with_retry, with_fallback, get_error_manager, safe_execute

# Import enhanced AI prompts
try:
    from .enhanced_ai_prompts import EnhancedAIPrompts, PromptType
    HAS_ENHANCED_PROMPTS = True
except ImportError:
    HAS_ENHANCED_PROMPTS = False
    logger.warning("Enhanced AI prompts not available, using basic prompts")

logger = logging.getLogger(__name__)


class MaterialityLevel(Enum):
    """Materiality levels for regulatory articles."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class StructureLevel(Enum):
    """Hierarchical structure levels in regulatory documents."""
    DOCUMENT = 0
    TITLE = 1      # TITRE
    CHAPTER = 2    # CHAPITRE
    SECTION = 3    # SECTION
    SUBSECTION = 4 # SOUS-SECTION
    ARTICLE = 5    # ARTICLE
    PARAGRAPH = 6  # Numbered paragraphs within articles
    ITEM = 7       # Lettered or bulleted items


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


@dataclass
class DocumentNode:
    """Represents a node in the document hierarchy."""
    level: StructureLevel
    number: str
    title: str
    content: str
    children: List['DocumentNode'] = field(default_factory=list)
    parent: Optional['DocumentNode'] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, child: 'DocumentNode'):
        """Add a child node and set parent relationship."""
        child.parent = self
        self.children.append(child)
    
    def get_full_path(self) -> str:
        """Get the full hierarchical path to this node."""
        path_parts = []
        node = self
        while node and node.level != StructureLevel.DOCUMENT:
            path_parts.append(f"{node.level.name} {node.number}")
            node = node.parent
        return " > ".join(reversed(path_parts))
    
    def find_articles(self) -> List['DocumentNode']:
        """Recursively find all article nodes."""
        articles = []
        if self.level == StructureLevel.ARTICLE:
            articles.append(self)
        for child in self.children:
            articles.extend(child.find_articles())
        return articles


class HierarchicalParser:
    """Parse regulatory documents into hierarchical structure."""
    
    def __init__(self):
        self.structure_patterns = {
            StructureLevel.TITLE: [
                r'TITRE\s+([IVXLCDM]+)\s*[:.–-]?\s*([^\n]+)',
                r'TITLE\s+([IVXLCDM]+)\s*[:.–-]?\s*([^\n]+)'
            ],
            StructureLevel.CHAPTER: [
                r'CHAPITRE\s+([IVXLCDM]+|\d+)\s*[:.–-]?\s*([^\n]+)',
                r'CHAPTER\s+([IVXLCDM]+|\d+)\s*[:.–-]?\s*([^\n]+)'
            ],
            StructureLevel.SECTION: [
                r'SECTION\s+([IVXLCDM]+|\d+)\s*[:.–-]?\s*([^\n]+)',
                r'Section\s+(\d+(?:\.\d+)?)\s*[:.–-]?\s*([^\n]+)'
            ],
            StructureLevel.SUBSECTION: [
                r'SOUS-SECTION\s+([IVXLCDM]+|\d+)\s*[:.–-]?\s*([^\n]+)',
                r'Sous-section\s+(\d+(?:\.\d+)?)\s*[:.–-]?\s*([^\n]+)',
                r'§\s*(\d+(?:\.\d+)?)\s*[:.–-]?\s*([^\n]+)'
            ],
            StructureLevel.ARTICLE: [
                r'Article\s+(\d+(?:[.,]\d+)?(?:\s*(?:bis|ter|quater))?)\s*[:.–-]?\s*([^\n]*)',
                r'Art\.\s*(\d+(?:[.,]\d+)?)\s*[:.–-]?\s*([^\n]*)'
            ]
        }
    
    def parse_document(self, text: str, metadata: Dict[str, Any] = None) -> DocumentNode:
        """Parse document into hierarchical structure."""
        # Create root node
        root = DocumentNode(
            level=StructureLevel.DOCUMENT,
            number="",
            title=metadata.get('file_name', 'Document') if metadata else 'Document',
            content="",
            metadata=metadata or {}
        )
        
        # Find all structural elements
        elements = self._find_all_elements(text)
        
        # Sort by position in text
        elements.sort(key=lambda x: x['position'])
        
        # Build hierarchy
        self._build_hierarchy(root, elements, text)
        
        # Extract content for each node
        self._extract_content(root, text, elements)
        
        return root
    
    def _find_all_elements(self, text: str) -> List[Dict[str, Any]]:
        """Find all structural elements in the text."""
        elements = []
        
        for level, patterns in self.structure_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    elements.append({
                        'level': level,
                        'number': match.group(1),
                        'title': match.group(2).strip() if len(match.groups()) >= 2 else '',
                        'position': match.start(),
                        'end_position': match.end(),
                        'full_match': match.group(0)
                    })
        
        return elements
    
    def _build_hierarchy(self, root: DocumentNode, elements: List[Dict], text: str):
        """Build the document hierarchy from found elements."""
        current_nodes = {level: root for level in StructureLevel}
        
        for element in elements:
            level = element['level']
            
            # Create node
            node = DocumentNode(
                level=level,
                number=self._normalize_number(element['number'], level),
                title=element['title'],
                content="",  # Will be filled later
                metadata={
                    'position': element['position'],
                    'end_position': element['end_position']
                }
            )
            
            # Find appropriate parent
            parent = self._find_parent_node(level, current_nodes)
            parent.add_child(node)
            
            # Update current nodes for this level and below
            current_nodes[level] = node
            for lower_level in StructureLevel:
                if lower_level.value > level.value:
                    current_nodes[lower_level] = node
    
    def _find_parent_node(self, level: StructureLevel, current_nodes: Dict[StructureLevel, DocumentNode]) -> DocumentNode:
        """Find the appropriate parent node for a given level."""
        # Look for the nearest higher level that has a node
        for parent_level in reversed(list(StructureLevel)):
            if parent_level.value < level.value and current_nodes.get(parent_level):
                return current_nodes[parent_level]
        
        # Default to root
        return current_nodes[StructureLevel.DOCUMENT]
    
    def _normalize_number(self, number: str, level: StructureLevel) -> str:
        """Normalize section/article numbers."""
        # Convert Roman numerals for titles and chapters
        if level in [StructureLevel.TITLE, StructureLevel.CHAPTER]:
            if re.match(r'^[IVXLCDM]+$', number.upper()):
                return number.upper()
        
        # Normalize decimal separators
        return number.replace(',', '.')
    
    def _extract_content(self, node: DocumentNode, full_text: str, all_elements: List[Dict]):
        """Extract content for each node in the hierarchy."""
        if node.level == StructureLevel.DOCUMENT:
            # For root, process children
            for child in node.children:
                self._extract_content(child, full_text, all_elements)
            return
        
        # Get position information
        start_pos = node.metadata.get('position', 0)
        
        # Find end position (start of next element at same or higher level)
        end_pos = len(full_text)
        current_index = next((i for i, e in enumerate(all_elements) 
                            if e['position'] == start_pos), -1)
        
        if current_index >= 0:
            for i in range(current_index + 1, len(all_elements)):
                next_element = all_elements[i]
                if next_element['level'].value <= node.level.value:
                    end_pos = next_element['position']
                    break
        
        # Extract content
        content = full_text[start_pos:end_pos].strip()
        
        # Remove the header line
        lines = content.split('\n', 1)
        if len(lines) > 1:
            node.content = lines[1].strip()
        else:
            node.content = ""
        
        # Process children
        for child in node.children:
            self._extract_content(child, full_text, all_elements)
    
    def export_structure(self, root: DocumentNode) -> Dict[str, Any]:
        """Export the hierarchical structure to a dictionary."""
        def node_to_dict(node: DocumentNode) -> Dict[str, Any]:
            return {
                'level': node.level.name,
                'number': node.number,
                'title': node.title,
                'content': node.content[:200] + '...' if len(node.content) > 200 else node.content,
                'full_path': node.get_full_path(),
                'children': [node_to_dict(child) for child in node.children]
            }
        
        return node_to_dict(root)
    
    def create_table_of_contents(self, root: DocumentNode) -> List[Dict[str, Any]]:
        """Create a table of contents from the document structure."""
        toc = []
        
        def process_node(node: DocumentNode, depth: int = 0):
            if node.level != StructureLevel.DOCUMENT:
                toc.append({
                    'level': node.level.name,
                    'number': node.number,
                    'title': node.title,
                    'depth': depth,
                    'path': node.get_full_path(),
                    'has_content': len(node.content) > 0,
                    'num_children': len(node.children)
                })
            
            for child in node.children:
                process_node(child, depth + 1)
        
        process_node(root)
        return toc
    
    def extract_structured_articles(self, root: DocumentNode) -> List[Dict[str, Any]]:
        """Extract articles with their full hierarchical context."""
        articles = []
        
        for article_node in root.find_articles():
            # Build context from hierarchy
            context = {
                'article_number': article_node.number,
                'article_title': article_node.title,
                'content': article_node.content,
                'hierarchy': {}
            }
            
            # Traverse up to get parent sections
            current = article_node.parent
            while current and current.level != StructureLevel.DOCUMENT:
                context['hierarchy'][current.level.name] = {
                    'number': current.number,
                    'title': current.title
                }
                current = current.parent
            
            # Add full path
            context['full_path'] = article_node.get_full_path()
            
            articles.append(context)
        
        return articles
    
    def find_related_articles(self, root: DocumentNode, search_terms: List[str]) -> List[DocumentNode]:
        """Find articles related to specific search terms."""
        related = []
        search_terms_lower = [term.lower() for term in search_terms]
        
        for article in root.find_articles():
            content_lower = article.content.lower()
            title_lower = article.title.lower()
            
            # Check if any search term appears in content or title
            if any(term in content_lower or term in title_lower for term in search_terms_lower):
                related.append(article)
        
        return related
    
    def get_section_summary(self, node: DocumentNode) -> Dict[str, Any]:
        """Get summary information for a section."""
        articles = node.find_articles()
        
        summary = {
            'level': node.level.name,
            'number': node.number,
            'title': node.title,
            'num_articles': len(articles),
            'num_subsections': len([c for c in node.children if c.level != StructureLevel.ARTICLE]),
            'total_content_length': len(node.content),
            'article_numbers': [a.number for a in articles]
        }
        
        # Analyze content themes
        if articles:
            all_content = ' '.join([a.content for a in articles])
            summary['key_themes'] = self._extract_key_themes(all_content)
        
        return summary
    
    def _extract_key_themes(self, content: str) -> List[str]:
        """Extract key themes from content."""
        theme_keywords = {
            'capital': ['capital', 'fonds propres', 'ratio de solvabilité'],
            'risk': ['risque', 'risk', 'exposition', 'exposure'],
            'governance': ['gouvernance', 'governance', 'conseil', 'board'],
            'compliance': ['conformité', 'compliance', 'respect', 'adhérence'],
            'reporting': ['rapport', 'reporting', 'déclaration', 'notification'],
            'penalties': ['sanction', 'pénalité', 'amende', 'penalty'],
            'liquidity': ['liquidité', 'liquidity', 'trésorerie', 'cash'],
            'audit': ['audit', 'contrôle', 'vérification', 'inspection']
        }
        
        found_themes = []
        content_lower = content.lower()
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                found_themes.append(theme)
        
        return found_themes


@dataclass
class CrossReference:
    """Represents a cross-reference between articles."""
    source_article: str
    target_article: str
    reference_type: str  # 'conformément à', 'voir', 'modifie', etc.
    reference_text: str  # The actual reference text
    context: str  # Surrounding text for context


class CrossReferenceResolver:
    """Resolve and analyze cross-references in regulatory documents."""
    
    def __init__(self):
        self.reference_patterns = {
            'fr': {
                'conformity': [
                    r'conformément (?:à|aux) (?:l\')?articles?\s+(\d+(?:[.,]\d+)?(?:\s*(?:bis|ter|quater))?(?:\s*(?:et|à)\s*\d+(?:[.,]\d+)?)*)',
                    r'en vertu de(?:s)? (?:l\')?articles?\s+(\d+(?:[.,]\d+)?(?:\s*(?:bis|ter|quater))?)',
                    r'selon (?:les dispositions de )?(?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'au sens de(?:s)? (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'prévu(?:e)?s?\s+(?:à|par|aux)\s+(?:l\')?articles?\s+(\d+(?:[.,]\d+)?)'
                ],
                'modification': [
                    r'modifie (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'remplace (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'abroge (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'complète (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)'
                ],
                'reference': [
                    r'voir (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'cf\.\s*(?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'se référer à (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'défini(?:e)?s?\s+(?:à|dans)\s+(?:l\')?articles?\s+(\d+(?:[.,]\d+)?)'
                ],
                'application': [
                    r'(?:en )?application de(?:s)? (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'pour l\'application de(?:s)? (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'aux fins de(?:s)? (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)'
                ],
                'exception': [
                    r'(?:par dérogation|nonobstant) (?:à |aux |les dispositions de )?(?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'sauf (?:disposition contraire de |dans )?(?:l\')?articles?\s+(\d+(?:[.,]\d+)?)',
                    r'à l\'exception de(?:s)? (?:l\')?articles?\s+(\d+(?:[.,]\d+)?)'
                ]
            },
            'en': {
                'conformity': [
                    r'in accordance with articles?\s+(\d+(?:\.\d+)?)',
                    r'pursuant to articles?\s+(\d+(?:\.\d+)?)',
                    r'as per articles?\s+(\d+(?:\.\d+)?)',
                    r'under articles?\s+(\d+(?:\.\d+)?)'
                ],
                'modification': [
                    r'amends articles?\s+(\d+(?:\.\d+)?)',
                    r'replaces articles?\s+(\d+(?:\.\d+)?)',
                    r'repeals articles?\s+(\d+(?:\.\d+)?)'
                ],
                'reference': [
                    r'see articles?\s+(\d+(?:\.\d+)?)',
                    r'refer to articles?\s+(\d+(?:\.\d+)?)',
                    r'as defined in articles?\s+(\d+(?:\.\d+)?)'
                ]
            }
        }
        
        # Initialize graph for reference network (if NetworkX is available)
        self.reference_graph = nx.DiGraph() if nx else None
    
    def extract_cross_references(self, articles: List[Dict[str, Any]], language: str = 'fr') -> List[CrossReference]:
        """Extract all cross-references from articles."""
        references = []
        
        for article in articles:
            article_num = article.get('number', '')
            content = article.get('content', '')
            
            # Add article to graph if NetworkX is available
            if self.reference_graph is not None:
                self.reference_graph.add_node(article_num, **article)
            
            # Extract references for each type
            for ref_type, patterns in self.reference_patterns.get(language, {}).items():
                for pattern in patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    
                    for match in matches:
                        # Extract referenced article numbers
                        ref_text = match.group(0)
                        referenced_nums = self._extract_article_numbers(match.group(1))
                        
                        # Get context (surrounding text)
                        start = max(0, match.start() - 50)
                        end = min(len(content), match.end() + 50)
                        context = content[start:end]
                        
                        # Create cross-reference for each referenced article
                        for ref_num in referenced_nums:
                            ref_num_normalized = self._normalize_article_number(ref_num)
                            
                            reference = CrossReference(
                                source_article=article_num,
                                target_article=ref_num_normalized,
                                reference_type=ref_type,
                                reference_text=ref_text,
                                context=context
                            )
                            references.append(reference)
                            
                            # Add edge to graph if NetworkX is available
                            if self.reference_graph is not None:
                                self.reference_graph.add_edge(
                                    article_num, 
                                    ref_num_normalized,
                                    type=ref_type,
                                    text=ref_text
                                )
        
        return references
    
    def _extract_article_numbers(self, text: str) -> List[str]:
        """Extract multiple article numbers from a reference."""
        numbers = []
        
        # Pattern for individual article numbers
        article_pattern = r'\d+(?:[.,]\d+)?(?:\s*(?:bis|ter|quater))?'
        
        # Handle ranges (e.g., "12 à 15")
        range_pattern = r'(\d+(?:[.,]\d+)?)\s*(?:à|to)\s*(\d+(?:[.,]\d+)?)'
        range_match = re.search(range_pattern, text)
        
        if range_match:
            start = int(range_match.group(1).replace(',', '.').split('.')[0])
            end = int(range_match.group(2).replace(',', '.').split('.')[0])
            numbers.extend([str(i) for i in range(start, end + 1)])
        else:
            # Extract individual numbers
            matches = re.findall(article_pattern, text)
            numbers.extend(matches)
        
        return numbers
    
    def _normalize_article_number(self, number: str) -> str:
        """Normalize article number format."""
        return f"Article {number.replace(',', '.')}"
    
    def analyze_reference_network(self) -> Dict[str, Any]:
        """Analyze the cross-reference network."""
        if not self.reference_graph or not nx:
            return {'error': 'NetworkX not available for network analysis'}
        
        analysis = {
            'total_articles': self.reference_graph.number_of_nodes(),
            'total_references': self.reference_graph.number_of_edges(),
            'most_referenced': self._get_most_referenced_articles(),
            'most_referencing': self._get_most_referencing_articles(),
            'isolated_articles': self._get_isolated_articles(),
            'reference_clusters': self._identify_clusters(),
            'circular_references': self._find_circular_references(),
            'reference_chains': self._find_reference_chains()
        }
        
        return analysis
    
    def _get_most_referenced_articles(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get the most referenced articles."""
        if not self.reference_graph:
            return []
        
        in_degrees = dict(self.reference_graph.in_degree())
        sorted_articles = sorted(in_degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_articles[:top_n]
    
    def _get_most_referencing_articles(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """Get articles that reference the most other articles."""
        if not self.reference_graph:
            return []
        
        out_degrees = dict(self.reference_graph.out_degree())
        sorted_articles = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)
        return sorted_articles[:top_n]
    
    def _get_isolated_articles(self) -> List[str]:
        """Get articles with no references in or out."""
        if not self.reference_graph:
            return []
        
        isolated = []
        for node in self.reference_graph.nodes():
            if self.reference_graph.in_degree(node) == 0 and self.reference_graph.out_degree(node) == 0:
                isolated.append(node)
        return isolated
    
    def _identify_clusters(self) -> List[Set[str]]:
        """Identify clusters of highly interconnected articles."""
        if not self.reference_graph or not nx:
            return []
        
        # Convert to undirected for community detection
        undirected = self.reference_graph.to_undirected()
        
        # Find connected components
        clusters = []
        for component in nx.connected_components(undirected):
            if len(component) > 1:  # Only include actual clusters
                clusters.append(component)
        
        return sorted(clusters, key=len, reverse=True)
    
    def _find_circular_references(self) -> List[List[str]]:
        """Find circular reference chains."""
        if not self.reference_graph or not nx:
            return []
        
        try:
            cycles = list(nx.simple_cycles(self.reference_graph))
            return [cycle for cycle in cycles if len(cycle) > 1]
        except:
            return []
    
    def _find_reference_chains(self, max_length: int = 5) -> List[List[str]]:
        """Find chains of references (A -> B -> C)."""
        if not self.reference_graph or not nx:
            return []
        
        chains = []
        
        for source in self.reference_graph.nodes():
            for target in self.reference_graph.nodes():
                if source != target:
                    try:
                        paths = list(nx.all_simple_paths(
                            self.reference_graph, 
                            source, 
                            target, 
                            cutoff=max_length
                        ))
                        for path in paths:
                            if len(path) >= 3:  # At least 3 nodes for a chain
                                chains.append(path)
                    except nx.NetworkXNoPath:
                        continue
        
        # Remove duplicate chains
        unique_chains = []
        seen = set()
        for chain in chains:
            chain_tuple = tuple(chain)
            if chain_tuple not in seen:
                seen.add(chain_tuple)
                unique_chains.append(chain)
        
        return sorted(unique_chains, key=len, reverse=True)
    
    def get_article_dependencies(self, article_number: str) -> Dict[str, List[str]]:
        """Get all dependencies for a specific article."""
        dependencies = {
            'references': [],  # Articles this one references
            'referenced_by': [],  # Articles that reference this one
            'indirect_references': [],  # Articles referenced through chains
            'indirect_referenced_by': []  # Articles that reference through chains
        }
        
        if not self.reference_graph or article_number not in self.reference_graph:
            return dependencies
        
        # Direct references
        dependencies['references'] = list(self.reference_graph.successors(article_number))
        dependencies['referenced_by'] = list(self.reference_graph.predecessors(article_number))
        
        # Indirect references (2+ hops)
        for node in self.reference_graph.nodes():
            if node != article_number:
                # Check if there's a path from article to node
                if nx.has_path(self.reference_graph, article_number, node):
                    path_length = nx.shortest_path_length(self.reference_graph, article_number, node)
                    if path_length > 1:
                        dependencies['indirect_references'].append(node)
                
                # Check if there's a path from node to article
                if nx.has_path(self.reference_graph, node, article_number):
                    path_length = nx.shortest_path_length(self.reference_graph, node, article_number)
                    if path_length > 1:
                        dependencies['indirect_referenced_by'].append(node)
        
        return dependencies
    
    def generate_reference_report(self, references: List[CrossReference]) -> Dict[str, Any]:
        """Generate a comprehensive reference report."""
        report = {
            'summary': {
                'total_references': len(references),
                'reference_types': {},
                'most_common_patterns': []
            },
            'by_type': {},
            'missing_targets': [],
            'bidirectional_references': []
        }
        
        # Count by type
        for ref in references:
            ref_type = ref.reference_type
            report['summary']['reference_types'][ref_type] = \
                report['summary']['reference_types'].get(ref_type, 0) + 1
            
            if ref_type not in report['by_type']:
                report['by_type'][ref_type] = []
            
            report['by_type'][ref_type].append({
                'from': ref.source_article,
                'to': ref.target_article,
                'text': ref.reference_text
            })
        
        # Find missing targets
        if self.reference_graph:
            all_articles = set(self.reference_graph.nodes())
            for ref in references:
                if ref.target_article not in all_articles:
                    report['missing_targets'].append({
                        'source': ref.source_article,
                        'missing_target': ref.target_article,
                        'reference_text': ref.reference_text
                    })
        
        # Find bidirectional references
        if self.reference_graph:
            for ref in references:
                # Check if there's a reverse reference
                if self.reference_graph.has_edge(ref.target_article, ref.source_article):
                    report['bidirectional_references'].append({
                        'article_1': ref.source_article,
                        'article_2': ref.target_article
                    })
        
        return report
    
    def visualize_reference_network(self, output_path: str = None) -> Dict[str, Any]:
        """Create visualization data for the reference network."""
        if not self.reference_graph:
            return {'error': 'No reference graph available'}
        
        # Prepare data for visualization (e.g., for D3.js or similar)
        nodes = []
        edges = []
        
        # Create nodes with metadata
        for node in self.reference_graph.nodes():
            node_data = {
                'id': node,
                'label': node,
                'in_degree': self.reference_graph.in_degree(node),
                'out_degree': self.reference_graph.out_degree(node),
                'total_degree': self.reference_graph.degree(node)
            }
            nodes.append(node_data)
        
        # Create edges with metadata
        for source, target, data in self.reference_graph.edges(data=True):
            edge_data = {
                'source': source,
                'target': target,
                'type': data.get('type', 'unknown'),
                'text': data.get('text', '')
            }
            edges.append(edge_data)
        
        visualization_data = {
            'nodes': nodes,
            'edges': edges,
            'statistics': {
                'total_nodes': len(nodes),
                'total_edges': len(edges),
                'average_degree': sum(n['total_degree'] for n in nodes) / len(nodes) if nodes else 0
            }
        }
        
        return visualization_data


class RegulatoryTableExtractor:
    """Enhanced table extraction for regulatory documents."""
    
    def __init__(self):
        self.table_patterns = {
            'capital_requirements': r'capital|fonds\s+propres|ratio',
            'penalties': r'sanctions|pénalités|amendes',
            'thresholds': r'seuils|limites|plafonds',
            'rates': r'taux|pourcentage|%',
            'deadlines': r'délais|échéances|dates\s+limites'
        }
    
    def extract_tables(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[str, List]:
        """Extract tables from PDF with categorization."""
        if not pd or not (camelot or tabula):
            return {'error': 'Table extraction libraries not available'}
        
        tables = {
            'raw_tables': [],
            'capital_tables': [],
            'penalty_tables': [],
            'threshold_tables': [],
            'rate_tables': [],
            'deadline_tables': [],
            'other_tables': []
        }
        
        # Try multiple extraction methods
        extracted_tables = []
        
        # Method 1: Camelot (best for bordered tables)
        if camelot:
            try:
                camelot_tables = self._extract_with_camelot(pdf_path, page_numbers)
                extracted_tables.extend(camelot_tables)
            except Exception as e:
                logger.debug(f"Camelot extraction failed: {e}")
        
        # Method 2: Tabula (good for borderless tables)
        if tabula:
            try:
                tabula_tables = self._extract_with_tabula(pdf_path, page_numbers)
                extracted_tables.extend(tabula_tables)
            except Exception as e:
                logger.debug(f"Tabula extraction failed: {e}")
        
        # Method 3: Custom regex-based extraction
        try:
            regex_tables = self._extract_with_regex(pdf_path, page_numbers)
            extracted_tables.extend(regex_tables)
        except Exception as e:
            logger.debug(f"Regex extraction failed: {e}")
        
        # Process and categorize tables
        for table_data in extracted_tables:
            df = table_data['dataframe']
            
            # Clean and standardize table
            df = self._clean_table(df)
            
            # Store raw table
            tables['raw_tables'].append({
                'dataframe': df,
                'page': table_data.get('page', 0),
                'method': table_data.get('method', 'unknown')
            })
            
            # Categorize table
            category = self._categorize_table(df)
            if category in tables:
                tables[category].append({
                    'dataframe': df,
                    'page': table_data.get('page', 0),
                    'category': category,
                    'headers': self._extract_headers(df),
                    'summary': self._generate_table_summary(df)
                })
        
        return tables
    
    def _extract_with_camelot(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[Dict]:
        """Extract tables using Camelot."""
        tables = []
        
        if page_numbers:
            pages = ','.join(map(str, page_numbers))
        else:
            pages = 'all'
        
        # Try lattice method first (for bordered tables)
        try:
            camelot_tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor='lattice',
                line_scale=40,
                shift_text=['']
            )
            
            for table in camelot_tables:
                tables.append({
                    'dataframe': table.df,
                    'page': table.page,
                    'method': 'camelot_lattice',
                    'accuracy': table.parsing_report.get('accuracy', 0)
                })
        except Exception as e:
            logger.debug(f"Camelot lattice method failed: {e}")
        
        # Try stream method (for borderless tables)
        try:
            camelot_tables = camelot.read_pdf(
                pdf_path,
                pages=pages,
                flavor='stream',
                edge_tol=50
            )
            
            for table in camelot_tables:
                tables.append({
                    'dataframe': table.df,
                    'page': table.page,
                    'method': 'camelot_stream',
                    'accuracy': table.parsing_report.get('accuracy', 0)
                })
        except Exception as e:
            logger.debug(f"Camelot stream method failed: {e}")
        
        return tables
    
    def _extract_with_tabula(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[Dict]:
        """Extract tables using Tabula."""
        tables = []
        
        if page_numbers:
            pages = page_numbers
        else:
            pages = 'all'
        
        # Extract tables
        try:
            dfs = tabula.read_pdf(
                pdf_path,
                pages=pages,
                multiple_tables=True,
                pandas_options={'header': None},
                lattice=True
            )
            
            for i, df in enumerate(dfs):
                tables.append({
                    'dataframe': df,
                    'page': page_numbers[i] if page_numbers and i < len(page_numbers) else 0,
                    'method': 'tabula'
                })
        except Exception as e:
            logger.debug(f"Tabula extraction failed: {e}")
        
        return tables
    
    def _extract_with_regex(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[Dict]:
        """Extract tables using regex patterns for structured text."""
        tables = []
        
        # This would require the text extraction first
        # Example pattern for capital requirement tables
        patterns = [
            # Pattern for ratio tables
            r'(?P<category>[^\n]+)\s+(?P<minimum>\d+[,.]?\d*)\s*%?\s+(?P<maximum>\d+[,.]?\d*)\s*%?',
            # Pattern for penalty tables
            r'(?P<violation>[^\n]+)\s+(?P<amount>[\d\s]+(?:000|millions?|milliards?))\s+(?P<currency>FCFA|EUR|USD)',
            # Pattern for deadline tables
            r'(?P<requirement>[^\n]+)\s+(?P<deadline>\d{1,2}/\d{1,2}/\d{4}|\d+ (?:jours?|mois|années?))'
        ]
        
        # Implementation would extract text and apply patterns
        # This is a placeholder for the actual implementation
        
        return tables
    
    def _clean_table(self, df) -> Any:
        """Clean and standardize extracted table."""
        if not pd:
            return df
        
        # Remove empty rows and columns
        df = df.dropna(how='all')
        df = df.dropna(axis=1, how='all')
        
        # Clean cell values
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].str.replace(r'\s+', ' ', regex=True)
        
        # Try to identify headers
        if df.shape[0] > 1:
            # Check if first row looks like headers
            first_row = df.iloc[0]
            if all(isinstance(val, str) and not str(val).isdigit() for val in first_row if pd.notna(val)):
                df.columns = first_row
                df = df[1:].reset_index(drop=True)
        
        return df
    
    def _categorize_table(self, df) -> str:
        """Categorize table based on content."""
        if not pd:
            return "other_tables"
        
        # Convert dataframe to string for pattern matching
        table_text = df.to_string().lower()
        
        # Check against patterns
        for category, pattern in self.table_patterns.items():
            if re.search(pattern, table_text):
                return f"{category}_tables"
        
        return "other_tables"
    
    def _extract_headers(self, df) -> List[str]:
        """Extract and clean table headers."""
        headers = []
        
        if hasattr(df, 'columns'):
            for col in df.columns:
                if isinstance(col, str):
                    headers.append(col)
                else:
                    headers.append(f"Column_{col}")
        
        return headers
    
    def _generate_table_summary(self, df) -> Dict[str, Any]:
        """Generate summary statistics for table."""
        if not pd:
            return {'error': 'Pandas not available'}
        
        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'numeric_columns': [],
            'text_columns': [],
            'has_totals': False,
            'has_percentages': False,
            'has_currency': False
        }
        
        for col in df.columns:
            # Check column type
            if df[col].dtype in ['int64', 'float64']:
                summary['numeric_columns'].append(str(col))
            else:
                summary['text_columns'].append(str(col))
                
                # Check for percentages
                if df[col].astype(str).str.contains('%').any():
                    summary['has_percentages'] = True
                
                # Check for currency
                if df[col].astype(str).str.contains('FCFA|EUR|USD', regex=True).any():
                    summary['has_currency'] = True
        
        # Check for total rows
        text_lower = df.to_string().lower()
        if 'total' in text_lower or 'somme' in text_lower:
            summary['has_totals'] = True
        
        return summary
    
    def export_tables_to_excel(self, tables: Dict[str, List], output_path: str):
        """Export categorized tables to Excel with formatting."""
        if not pd:
            logger.error("Pandas not available for Excel export")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for category, table_list in tables.items():
                if category != 'raw_tables' and table_list:
                    summary_data.append({
                        'Category': category.replace('_tables', '').title(),
                        'Count': len(table_list),
                        'Total Rows': sum(t['dataframe'].shape[0] for t in table_list),
                        'Pages': ', '.join(str(t.get('page', 0)) for t in table_list)
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Table_Summary', index=False)
            
            # Export each category
            for category, table_list in tables.items():
                if category == 'raw_tables' or not table_list:
                    continue
                
                # Create sheet for category
                sheet_name = category.replace('_tables', '').title()[:31]  # Excel sheet name limit
                
                # Combine tables with metadata
                current_row = 0
                for i, table_data in enumerate(table_list):
                    df = table_data['dataframe']
                    
                    # Write metadata
                    metadata_df = pd.DataFrame([{
                        'Table': f"Table {i+1}",
                        'Page': table_data.get('page', 'Unknown'),
                        'Rows': df.shape[0],
                        'Columns': df.shape[1]
                    }])
                    metadata_df.to_excel(writer, sheet_name=sheet_name, 
                                       startrow=current_row, index=False)
                    
                    # Write table
                    current_row += 3
                    df.to_excel(writer, sheet_name=sheet_name, 
                              startrow=current_row, index=False)
                    
                    current_row += df.shape[0] + 3  # Space between tables
    
    def find_regulatory_tables(self, text: str, article_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find and extract tables from article text."""
        tables = []
        
        # Pattern for structured data that might be tables
        table_indicators = [
            r'(?:^|\n)\s*\|',  # Pipe-delimited tables
            r'(?:^|\n)\s*[-–—]{3,}',  # Horizontal rules
            r'(?:^|\n)\s*\d+\s*[.)\-]\s*[^\n]+\s+\d+',  # Numbered lists with values
            r'(?:Tableau|Table)\s*\d*\s*:',  # Explicit table references
        ]
        
        for pattern in table_indicators:
            if re.search(pattern, text, re.MULTILINE):
                # Extract potential table content
                table_content = self._extract_table_from_text(text, pattern)
                if table_content:
                    tables.append({
                        'content': table_content,
                        'article': article_context.get('article_number', 'Unknown'),
                        'type': 'text_based',
                        'pattern': pattern
                    })
        
        return tables
    
    def _extract_table_from_text(self, text: str, pattern: str) -> Optional[str]:
        """Extract table-like content from text."""
        lines = text.split('\n')
        table_lines = []
        in_table = False
        
        for line in lines:
            if re.match(pattern, line):
                in_table = True
            elif in_table and line.strip() == '':
                # End of table
                break
            
            if in_table:
                table_lines.append(line)
        
        if len(table_lines) > 2:  # Minimum table size
            return '\n'.join(table_lines)
        
        return None


class COBACOCREnhancedParser:
    """
    Enhanced OCR parser specifically designed for COBAC regulatory documents.
    Handles common OCR errors and improves text extraction quality.
    """
    
    def __init__(self, confidence_threshold: float = 0.75):
        """
        Initialize the COBAC OCR Enhanced Parser.
        
        Args:
            confidence_threshold: Minimum confidence threshold for OCR results
        """
        self.confidence_threshold = confidence_threshold
        
        # COBAC-specific correction patterns
        self.cobac_corrections = {
            # Common OCR errors in French regulatory text
            r'\b0\b': 'O',  # Zero to letter O
            r'\bl\b': 'I',  # lowercase l to uppercase I
            r'rn\b': 'ni',  # rn to ni
            r'\bm\b': 'ni', # m to ni in context
            
            # COBAC-specific terms
            r'C0BAC': 'COBAC',
            r'CQBAC': 'COBAC',
            r'C0BAC': 'COBAC',
            r'CEMAO': 'CEMAC',
            r'CEMAG': 'CEMAC',
            
            # Banking terms
            r'banqu0': 'banque',
            r'créd1t': 'crédit',
            r'débit0ur': 'débiteur',
            r'crédit0ur': 'créditeur',
            r'établissem0nt': 'établissement',
            
            # Legal terms
            r'articl0': 'article',
            r'alln0a': 'alinéa',
            r'paragrap0e': 'paragraphe',
            r'chap1tre': 'chapitre',
            r't1tre': 'titre',
            
            # Numbers and symbols
            r'°': '°',  # Degree symbol
            r'0/00': '‰',  # Per mille
            r'0/0': '%',   # Percent
        }
        
        # French regulatory vocabulary for context validation
        self.regulatory_vocabulary = {
            'cobac', 'cemac', 'banque', 'crédit', 'établissement', 'financier',
            'article', 'alinéa', 'paragraphe', 'chapitre', 'titre', 'section',
            'règlement', 'instruction', 'circulaire', 'décision',
            'capital', 'fonds', 'réserve', 'provision', 'ratio', 'solvabilité',
            'liquidité', 'risque', 'créance', 'garantie', 'nantissement',
            'surveillance', 'contrôle', 'sanction', 'pénalité', 'amende'
        }
    
    def enhance_extracted_text(self, raw_text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhance OCR-extracted text with COBAC-specific corrections.
        
        Args:
            raw_text: Raw text from OCR extraction
            image_path: Optional path to source image for re-processing
            
        Returns:
            Dictionary with enhanced text and quality metrics
        """
        if not raw_text or not raw_text.strip():
            return {
                'enhanced_text': '',
                'confidence_score': 0.0,
                'corrections_made': 0,
                'quality_issues': ['Empty or no text extracted']
            }
        
        try:
            # Apply COBAC-specific corrections
            enhanced_text = self._apply_corrections(raw_text)
            
            # Validate and improve structure
            enhanced_text = self._fix_document_structure(enhanced_text)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(raw_text, enhanced_text)
            
            # Count corrections made
            corrections_made = self._count_corrections(raw_text, enhanced_text)
            
            # Identify quality issues
            quality_issues = self._identify_quality_issues(enhanced_text)
            
            # Attempt re-processing if confidence is too low and image is available
            if confidence_score < self.confidence_threshold and image_path and pytesseract:
                reprocessed_result = self._reprocess_with_enhanced_ocr(image_path)
                if reprocessed_result['confidence_score'] > confidence_score:
                    return reprocessed_result
            
            return {
                'enhanced_text': enhanced_text,
                'confidence_score': confidence_score,
                'corrections_made': corrections_made,
                'quality_issues': quality_issues,
                'processing_method': 'pattern_correction'
            }
            
        except Exception as e:
            logger.error(f"Error in OCR enhancement: {e}")
            return {
                'enhanced_text': raw_text,  # Return original on error
                'confidence_score': 0.5,  # Neutral confidence
                'corrections_made': 0,
                'quality_issues': [f'Processing error: {str(e)}'],
                'processing_method': 'fallback'
            }
    
    def _apply_corrections(self, text: str) -> str:
        """Apply COBAC-specific correction patterns."""
        corrected_text = text
        
        for pattern, replacement in self.cobac_corrections.items():
            corrected_text = re.sub(pattern, replacement, corrected_text, flags=re.IGNORECASE)
        
        return corrected_text
    
    def _fix_document_structure(self, text: str) -> str:
        """Fix common document structure issues from OCR."""
        # Fix broken article numbers
        text = re.sub(r'Artlcle\s+(\d+)', r'Article \1', text, flags=re.IGNORECASE)
        text = re.sub(r'Art[il]cle\s+(\d+)', r'Article \1', text, flags=re.IGNORECASE)
        
        # Fix broken section headers
        text = re.sub(r'SECT[I1]ON\s+([IVX\d]+)', r'SECTION \1', text, flags=re.IGNORECASE)
        text = re.sub(r'CHAP[IT]TRE\s+([IVX\d]+)', r'CHAPITRE \1', text, flags=re.IGNORECASE)
        text = re.sub(r'T[IT]TRE\s+([IVX\d]+)', r'TITRE \1', text, flags=re.IGNORECASE)
        
        # Fix paragraph breaks
        text = re.sub(r'([.!?])\s*([A-Z][a-z])', r'\1\n\n\2', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.;:!?])', r'\1', text)
        text = re.sub(r'([,.;:!?])([A-Za-z])', r'\1 \2', text)
        
        return text
    
    def _calculate_confidence(self, original: str, enhanced: str) -> float:
        """Calculate confidence score based on text quality indicators."""
        if not enhanced or not enhanced.strip():
            return 0.0
        
        score = 1.0
        
        # Check for regulatory vocabulary presence
        vocab_matches = sum(1 for word in self.regulatory_vocabulary 
                          if word.lower() in enhanced.lower())
        vocab_score = min(vocab_matches / 10.0, 1.0)  # Max 10 vocabulary words
        
        # Check for proper structure patterns
        structure_patterns = [
            r'Article\s+\d+',
            r'CHAPITRE\s+[IVX\d]+',
            r'SECTION\s+[IVX\d]+',
            r'TITRE\s+[IVX\d]+'
        ]
        
        structure_score = 0
        for pattern in structure_patterns:
            if re.search(pattern, enhanced, re.IGNORECASE):
                structure_score += 0.25
        
        # Check for proper French text characteristics
        french_score = 0.0
        if re.search(r'[àáâäèéêëìíîïòóôöùúûü]', enhanced):  # French accents
            french_score += 0.2
        if 'qu' in enhanced.lower():  # Common French digraph
            french_score += 0.1
        if any(word in enhanced.lower() for word in ['le', 'la', 'les', 'de', 'du', 'des']):
            french_score += 0.2
        
        # Penalty for obvious OCR errors
        error_patterns = [r'\b[0O]{2,}', r'\b[1Il]{3,}', r'[^\w\s\-.,;:!?()°%‰àáâäèéêëìíîïòóôöùúûü]']
        error_penalty = 0
        for pattern in error_patterns:
            matches = len(re.findall(pattern, enhanced))
            error_penalty += matches * 0.05
        
        # Calculate final score
        final_score = (0.4 * vocab_score + 0.3 * structure_score + 0.3 * french_score) - error_penalty
        return max(0.0, min(1.0, final_score))
    
    def _count_corrections(self, original: str, enhanced: str) -> int:
        """Count the number of corrections made."""
        if original == enhanced:
            return 0
        
        corrections = 0
        for pattern in self.cobac_corrections.keys():
            original_matches = len(re.findall(pattern, original, re.IGNORECASE))
            enhanced_matches = len(re.findall(pattern, enhanced, re.IGNORECASE))
            corrections += max(0, original_matches - enhanced_matches)
        
        return corrections
    
    def _identify_quality_issues(self, text: str) -> List[str]:
        """Identify potential quality issues in the text."""
        issues = []
        
        if len(text) < 100:
            issues.append("Text too short - possible extraction issue")
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^\w\s\-.,;:!?()°%‰àáâäèéêëìíîïòóôöùúûü]', text)) / max(len(text), 1)
        if special_char_ratio > 0.1:
            issues.append("High ratio of special characters detected")
        
        # Check for repeated characters (OCR artifacts)
        if re.search(r'(.)\1{4,}', text):
            issues.append("Repeated character sequences detected")
        
        # Check for missing regulatory structure
        if not re.search(r'Article\s+\d+', text, re.IGNORECASE):
            issues.append("No article structure detected")
        
        # Check for proper French text
        if not re.search(r'[àáâäèéêëìíîïòóôöùúûü]', text):
            issues.append("No French accented characters - possible encoding issue")
        
        return issues
    
    def _reprocess_with_enhanced_ocr(self, image_path: str) -> Dict[str, Any]:
        """Reprocess image with enhanced OCR settings."""
        if not pytesseract or not Image or not cv2 or not np:
            return {
                'enhanced_text': '',
                'confidence_score': 0.0,
                'corrections_made': 0,
                'quality_issues': ['OCR libraries not available'],
                'processing_method': 'enhanced_ocr_unavailable'
            }
        
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply noise reduction
            denoised = cv2.medianBlur(gray, 3)
            
            # Apply sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            # Adaptive thresholding
            thresh = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # OCR with French language model
            custom_config = r'--oem 3 --psm 6 -l fra'
            
            # Extract text with confidence scores
            data = pytesseract.image_to_data(thresh, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter low-confidence words
            filtered_text = []
            confidences = []
            
            for i, conf in enumerate(data['conf']):
                if int(conf) > 50:  # Confidence threshold
                    word = data['text'][i].strip()
                    if word:
                        filtered_text.append(word)
                        confidences.append(int(conf))
            
            enhanced_text = ' '.join(filtered_text)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Apply corrections to the enhanced OCR result
            final_text = self._apply_corrections(enhanced_text)
            final_text = self._fix_document_structure(final_text)
            
            return {
                'enhanced_text': final_text,
                'confidence_score': avg_confidence / 100.0,  # Convert to 0-1 scale
                'corrections_made': self._count_corrections(enhanced_text, final_text),
                'quality_issues': self._identify_quality_issues(final_text),
                'processing_method': 'enhanced_ocr_reprocessing'
            }
            
        except Exception as e:
            logger.error(f"Enhanced OCR reprocessing failed: {e}")
            return {
                'enhanced_text': '',
                'confidence_score': 0.0,
                'corrections_made': 0,
                'quality_issues': [f'Enhanced OCR failed: {str(e)}'],
                'processing_method': 'enhanced_ocr_failed'
            }


class AIValidator:
    """AI-powered document validation with Claude API."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-20250514", 
                 use_enhanced_prompts: bool = True):
        """
        Initialize the AI validator.
        
        Args:
            api_key: Anthropic API key (reads from env if not provided)
            model: Claude model to use
            use_enhanced_prompts: Whether to use enhanced prompt system
        """
        self.api_key = api_key
        self.model = model
        self.client = None
        self.use_enhanced_prompts = use_enhanced_prompts and HAS_ENHANCED_PROMPTS
        
        if api_key:
            try:
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Claude API: {e}")
        
        # Initialize enhanced prompts if available
        if self.use_enhanced_prompts:
            self.enhanced_prompts = EnhancedAIPrompts()
            logger.info("Enhanced AI prompts initialized")
        
        # Prompt templates (fallback for basic prompts)
        self.validation_prompt = self._load_validation_prompt()
        self.materiality_prompt = self._load_materiality_prompt()
        self.article_extraction_prompt = self._load_article_extraction_prompt()
        
        # Initialize hierarchical parser
        self.hierarchical_parser = HierarchicalParser()
        
        # Initialize cross-reference resolver
        self.cross_reference_resolver = CrossReferenceResolver()
        
        # Initialize table extractor
        self.table_extractor = RegulatoryTableExtractor()
        
        # Initialize OCR enhanced parser
        self.ocr_parser = COBACOCREnhancedParser()
    
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

Provide your analysis in the following JSON format. IMPORTANT: Always provide a complete, valid JSON response. Do not truncate or leave incomplete:

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
}

Example of a proper response:
{
    "completeness_score": 85,
    "completeness_issues": ["Some minor formatting issues"],
    "reliability_score": 92,
    "reliability_issues": [],
    "legal_structure_score": 78,
    "legal_structure_issues": ["Article numbering could be clearer"],
    "overall_score": 85,
    "recommendations": ["Review article numbering system"],
    "chunk_quality": "GOOD"
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

Provide your assessment in JSON format. IMPORTANT: Always provide a complete, valid JSON response. Do not truncate or leave incomplete:

{
    "materiality_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "reasoning": "Detailed explanation of the materiality assessment",
    "key_impacts": ["list of primary impact areas"],
    "compliance_urgency": "IMMEDIATE|SHORT_TERM|MEDIUM_TERM|LONG_TERM",
    "stakeholders_affected": ["list of affected stakeholder groups"]
}

Example response:
{
    "materiality_level": "HIGH",
    "reasoning": "This article establishes critical capital requirements that directly impact bank solvency and regulatory compliance.",
    "key_impacts": ["Capital adequacy", "Regulatory compliance", "Risk management"],
    "compliance_urgency": "IMMEDIATE",
    "stakeholders_affected": ["Bank management", "Regulators", "Shareholders"]
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

Format the output as JSON. IMPORTANT: Always provide a complete, valid JSON response. Do not truncate or leave incomplete:

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
}

Example response:
{
    "articles": [
        {
            "number": "1",
            "title": "Capital Requirements",
            "content": "Banks must maintain minimum capital adequacy ratios as specified in this regulation...",
            "references": ["Article 5", "Article 12"],
            "has_sub_articles": false
        }
    ],
    "extraction_confidence": "HIGH",
    "total_articles": 1,
    "regulation_type": "Règlement COBAC"
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
                max_tokens=4000,  # Increased to prevent truncation
                temperature=0.1,
                system="You are an expert regulatory document validator. CRITICAL: Always provide complete, properly formatted JSON responses. Start with { and end with }. Ensure all JSON objects are properly closed. Never truncate responses.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Log response details for debugging
            logger.debug(f"AI response length: {len(response.content[0].text)} characters")
            logger.debug(f"AI response preview: {response.content[0].text[:100]}...")
            
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
        
        # Then enhance with AI if available and working
        if self.client and len(text) < 50000:  # Limit for API calls
            # Only try AI enhancement if we haven't had too many recent failures
            error_manager = get_error_manager()
            recent_ai_errors = [
                err for err in error_manager.error_history[-10:]  # Last 10 errors
                if 'AI' in err.function_name and 
                   (datetime.now() - err.timestamp).total_seconds() < 300  # Last 5 minutes
            ]
            
            if len(recent_ai_errors) < 5:  # Don't try if too many recent AI failures
                try:
                    if self.use_enhanced_prompts:
                        # Use enhanced prompts for better extraction
                        enhanced_articles = safe_execute(
                            self._enhance_articles_with_enhanced_ai,
                            text, articles, document_metadata,
                            default_return=articles
                        )
                    else:
                        # Fallback to basic AI enhancement
                        enhanced_articles = safe_execute(
                            self._enhance_articles_with_ai,
                            text, articles, document_metadata,
                            default_return=articles
                        )
                    
                    if enhanced_articles and len(enhanced_articles) > len(articles):
                        articles = enhanced_articles
                        logger.info("AI enhancement improved article extraction")
                except Exception as e:
                    error_manager._record_error(e, "_enhance_articles_with_ai")
                    logger.debug(f"AI article enhancement failed: {e}")
            else:
                logger.debug("Skipping AI enhancement due to recent failures")
        
        logger.info(f"Extracted {len(articles)} articles from document")
        return articles
    
    def _extract_articles_regex(self, text: str, metadata: Dict[str, Any]) -> List[Article]:
        """Extract articles using improved regex patterns."""
        # Use enhanced version with normalized text
        return self._extract_articles_regex_enhanced(text, metadata)
    
    def _extract_articles_regex_enhanced(self, text: str, metadata: Dict[str, Any]) -> List[Article]:
        """Extract articles using comprehensive enhanced regex patterns."""
        articles = []
        seen_articles = set()  # Track already found articles
        
        logger.info(f"Starting regex article extraction on {len(text)} characters")
        
        # Additional processing for complex structures
        text_normalized = self._normalize_article_text(text)
        
        # Enhanced patterns for better article detection
        patterns = [
            # Standard formats
            r'Article\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|novies|decies))?)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Article\s+\d+)[\s\S]){20,5000})',
            
            # With parentheses or dashes
            r'Article\s+(\d+(?:[\.,]\d+)?)\s*\(([^)]+)\)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Article\s+\d+)[\s\S]){20,5000})',
            
            # Numbered sections that might be articles
            r'^\s*(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?)\s*[-–—]\s*([^\n]{0,200})?\n?((?:(?!^\s*\d+\s*[-–—])[\s\S]){20,3000})',
            
            # Article references in running text
            r"(?:En vertu de l'article|Conformément à l'article|Selon l'article|Au sens de l'article)\s+(\d+(?:[\.,]\d+)?)\s*,?\s*([^,\n]{0,200})",
            
            # COBAC/CEMAC specific formats
            r'Art\.\s*(\d+)\s*COBAC\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Art\.\s*\d+)[\s\S]){20,3000})',
            r'R\s*(\d+(?:[\.,]\d+)?)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!R\s*\d+)[\s\S]){20,3000})',  # R followed by number
            r'REGLE\s*(\d+(?:[\.,]\d+)?)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!REGLE\s*\d+)[\s\S]){20,3000})',  # REGLE format
            r'Règle\s*(\d+(?:[\.,]\d+)?)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Règle\s*\d+)[\s\S]){20,3000})',  # Règle format
            
            # Multi-line article headers
            r'ARTICLE\s+(\d+(?:[\.,]\d+)?)\s*\n\s*([^\n]{0,200})?\n?((?:(?!ARTICLE\s+\d+)[\s\S]){20,5000})',
            
            # Articles with subsections
            r'Article\s+(\d+(?:[\.,]\d+)?)\s*[:.-]?\s*\n\s*(\d+\.\d+[^\n]*\n(?:\s*\d+\.\d+[^\n]*\n)*)',
            
            # French ordinal numbers
            r'Article\s+(premier|deuxième|troisième|quatrième|cinquième|sixième|septième|huitième|neuvième|dixième)\s*[:.-]?\s*([^\n]{0,200})?\n?((?:(?!Article\s+(?:premier|deuxième))[\s\S]){20,3000})',
            
            # Account modification pattern (COBAC specific)
            r'Le\s+compte\s+divisionnaire\s+«\s*(\d+[-\s]*[^»]*)\s*»\s*([^\n]{0,200})?\n?((?:(?!Le\s+compte\s+divisionnaire)[\s\S]){20,3000})',
            
            # L'article format
            r"[Ll]['']article\s+(\d+(?:[\.,]\d+)?(?:\s*[a-zA-Z])?)\s+(?:stipule|dispose|prévoit|énonce|précise)?[:\s]+([^\n]{0,200})?\n?((?:(?![Ll]['']article\s+\d+)[\s\S]){20,3000})"
        ]
        
        for i, pattern in enumerate(patterns):
            matches = list(re.finditer(pattern, text_normalized, re.MULTILINE | re.IGNORECASE))
            if matches:
                logger.debug(f"Pattern {i} found {len(matches)} matches")
            for match in matches:
                article_number = self._normalize_article_number(match.group(1))
                
                if article_number in seen_articles:
                    continue
                seen_articles.add(article_number)
                
                # Extract complete article content using new helper method
                article_content = self._extract_complete_article_content(
                    text_normalized, 
                    match.start(), 
                    match.end(),
                    article_number
                )
                
                if article_content and len(article_content) > 50:
                    article = Article(
                        number=f"Article {article_number}",
                        title=self._extract_article_title(match),
                        content=article_content,
                        materiality=MaterialityLevel.MEDIUM,
                        materiality_reasoning="Pending AI assessment",
                        context=self._build_article_context(metadata, article_number)
                    )
                    articles.append(article)
        
        logger.info(f"Regex extraction found {len(articles)} articles before post-processing")
        return self._post_process_articles(articles)
    
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
    
    def _normalize_article_text(self, text: str) -> str:
        """Normalize text for better article extraction."""
        # Fix common OCR issues
        text = text.replace('Arlicle', 'Article')
        text = text.replace('Artide', 'Article')
        text = text.replace('Artiele', 'Article')
        
        # Normalize spaces and line breaks
        text = re.sub(r'\s*\n\s*', '\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text

    def _normalize_article_number(self, number_str: str) -> str:
        """Normalize article numbers to consistent format."""
        # Convert French ordinals to numbers
        ordinals = {
            'premier': '1', 'deuxième': '2', 'troisième': '3',
            'quatrième': '4', 'cinquième': '5', 'sixième': '6',
            'septième': '7', 'huitième': '8', 'neuvième': '9', 'dixième': '10'
        }
        
        number_str = number_str.lower().strip()
        if number_str in ordinals:
            return ordinals[number_str]
        
        # Normalize separators
        number_str = number_str.replace(',', '.')
        
        # Remove extra spaces
        number_str = re.sub(r'\s+', ' ', number_str)
        
        return number_str

    def _extract_complete_article_content(self, text: str, start: int, end: int, article_num: str) -> str:
        """Extract the complete content of an article, including subsections."""
        # Look for the next article to determine boundaries
        next_article_pattern = r'(?:Article|ARTICLE|Art\.)\s+\d+(?![\.,]\d)'
        
        # Find next article position
        next_match = re.search(next_article_pattern, text[end:], re.IGNORECASE)
        
        if next_match:
            content_end = end + next_match.start()
        else:
            # Look for other section markers
            section_markers = [
                r'^\s*TITRE\s+[IVX]+',
                r'^\s*CHAPITRE\s+[IVX]+',
                r'^\s*SECTION\s+[IVX]+',
                r'^\s*ANNEXE\s+[IVX]+',
            ]
            
            content_end = len(text)
            for marker in section_markers:
                marker_match = re.search(marker, text[end:], re.MULTILINE | re.IGNORECASE)
                if marker_match and end + marker_match.start() < content_end:
                    content_end = end + marker_match.start()
        
        # Extract and clean content
        content = text[start:content_end].strip()
        
        # Remove the article header from content
        content = re.sub(r'^Article\s+\d+[^\n]*\n', '', content, flags=re.IGNORECASE)
        
        return content.strip()

    def _extract_article_title(self, match) -> Optional[str]:
        """Extract article title from regex match."""
        # Try different group positions based on pattern
        for i in range(2, min(4, len(match.groups()) + 1)):
            if match.group(i) and len(match.group(i).strip()) > 5:
                title = match.group(i).strip()
                # Clean up title
                title = re.sub(r'^[:.-]\s*', '', title)
                title = re.sub(r'\s*[:.-]$', '', title)
                if len(title) > 5 and len(title) < 200:
                    return title
        return None

    def _build_article_context(self, metadata: Dict[str, Any], article_number: str) -> Dict[str, Any]:
        """Build comprehensive context for an article."""
        context = {
            'regulation_name': metadata.get('file_name', '').replace('.pdf', ''),
            'document_type': metadata.get('document_type', 'UNKNOWN'),
            'extraction_method': 'enhanced_regex',
            'article_number': article_number,
            'page_count': metadata.get('page_count', 0),
            'extraction_date': datetime.now().isoformat()
        }
        
        # Determine regulation type
        filename = metadata.get('file_name', '').upper()
        if 'COBAC' in filename:
            context['regulation_type'] = 'Règlement COBAC'
            context['authority'] = 'COBAC'
        elif 'CEMAC' in filename:
            context['regulation_type'] = 'Règlement CEMAC'
            context['authority'] = 'CEMAC'
        elif 'PENAL' in filename or 'PÉNAL' in filename:
            context['regulation_type'] = 'Code Pénal'
            context['authority'] = 'République Gabonaise'
        else:
            context['regulation_type'] = 'Unknown'
            context['authority'] = 'Unknown'
        
        return context

    def _post_process_articles(self, articles: List[Article]) -> List[Article]:
        """Post-process extracted articles for quality."""
        processed = []
        
        for article in articles:
            # Remove duplicate or near-duplicate articles
            is_duplicate = False
            for p_article in processed:
                similarity = self._calculate_similarity(article.content, p_article.content)
                if similarity > 0.9:  # 90% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                # Clean up article content
                article.content = self._clean_article_content(article.content)
                processed.append(article)
        
        # Sort by article number
        processed.sort(key=lambda x: self._parse_article_number_for_sort(x.number))
        
        return processed
    
    def _parse_article_number_for_sort(self, article_number: str) -> Tuple[int, int, str]:
        """
        Parse article number for sorting purposes.
        
        Returns:
            Tuple of (main_number, sub_number, original_string) for sorting
        """
        if not article_number:
            return (9999, 0, article_number)
        
        # Clean the article number
        clean_num = article_number.strip().upper()
        
        # Try to extract numeric components
        # Pattern: "Article 12" or "Art. 15" or "15" or "12-3" etc.
        
        # Remove common prefixes
        for prefix in ['ARTICLE', 'ART.', 'ART']:
            if clean_num.startswith(prefix):
                clean_num = clean_num[len(prefix):].strip()
                break
        
        # Handle formats like "12-3", "12.3", "12bis", "12ter"
        main_match = re.match(r'^(\d+)', clean_num)
        if main_match:
            main_num = int(main_match.group(1))
            
            # Look for sub-numbers
            remainder = clean_num[len(main_match.group(1)):].strip()
            
            # Handle "-3", ".3" patterns
            sub_match = re.match(r'^[.\-](\d+)', remainder)
            if sub_match:
                sub_num = int(sub_match.group(1))
                return (main_num, sub_num, article_number)
            
            # Handle "bis", "ter", "quater" patterns
            if remainder.startswith('BIS'):
                return (main_num, 1, article_number)
            elif remainder.startswith('TER'):
                return (main_num, 2, article_number)
            elif remainder.startswith('QUATER'):
                return (main_num, 3, article_number)
            
            return (main_num, 0, article_number)
        
        # If no numeric pattern found, sort at the end
        return (9999, 0, article_number)

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        # Simple character-based similarity
        if not text1 or not text2:
            return 0.0
        
        # Normalize texts
        text1 = ' '.join(text1.lower().split())
        text2 = ' '.join(text2.lower().split())
        
        # Use Jaccard similarity on words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _clean_article_content(self, content: str) -> str:
        """Clean article content for better readability."""
        # Remove excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r' {2,}', ' ', content)
        
        # Fix common formatting issues
        content = re.sub(r'(\d+)\s*°\s*', r'\1° ', content)  # Fix degree symbols
        content = re.sub(r'([a-z])\s*\)\s*', r'\1) ', content)  # Fix list items
        
        # Ensure proper spacing after punctuation
        content = re.sub(r'([.!?,:;])([A-Z])', r'\1 \2', content)
        
        return content.strip()
    
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
                max_tokens=4000,
                temperature=0.1,
                system="You are an expert in regulatory document analysis. Extract articles precisely. Always provide complete, properly formatted JSON responses. Ensure all JSON objects are properly closed.",
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
    
    def _enhance_articles_with_enhanced_ai(self, text: str, 
                                         regex_articles: List[Article], 
                                         metadata: Dict[str, Any]) -> List[Article]:
        """Enhance article extraction using enhanced AI prompts system."""
        if not self.client or not self.use_enhanced_prompts:
            return regex_articles
        
        try:
            # Get enhanced prompts for article extraction
            system_prompt, user_prompt = self.enhanced_prompts.get_prompt(
                PromptType.ARTICLE_EXTRACTION,
                document_text=text[:15000],  # Limit text length for API
                document_context={
                    'file_name': metadata.get('file_name', ''),
                    'document_type': metadata.get('document_type', ''),
                    'extraction_method': 'enhanced_ai'
                }
            )
            
            # Make API call with enhanced prompts
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4000,  # Higher token limit for enhanced extraction
                temperature=0.1,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Validate and parse the response
            validation_result = self.enhanced_prompts.validate_output(
                PromptType.ARTICLE_EXTRACTION,
                response.content[0].text
            )
            
            if not validation_result['is_valid']:
                logger.warning(f"Enhanced AI output validation failed: {validation_result.get('error', 'Unknown error')}")
                return regex_articles
            
            # Extract articles from validated response
            result = validation_result.get('parsed_output', {})
            ai_articles_data = result.get('articles', [])
            
            # Convert to Article objects
            enhanced_articles = []
            for ai_art in ai_articles_data:
                # Extract hierarchy information
                hierarchy = ai_art.get('hierarchie', {})
                
                article = Article(
                    number=ai_art.get('numero', ''),
                    title=ai_art.get('titre'),
                    content=ai_art.get('contenu', ''),
                    materiality=MaterialityLevel.MEDIUM,  # Will be assessed separately
                    materiality_reasoning="Pending enhanced assessment",
                    context={
                        'regulation_name': metadata.get('file_name', ''),
                        'document_type': metadata.get('document_type', ''),
                        'extraction_method': 'enhanced_ai',
                        'hierarchy': hierarchy,
                        'references': ai_art.get('references', []),
                        'position': ai_art.get('position', {}),
                        'validation_score': validation_result.get('validation_score', 0.8)
                    }
                )
                enhanced_articles.append(article)
            
            # Log enhancement success
            if enhanced_articles:
                logger.info(f"Enhanced AI extracted {len(enhanced_articles)} articles "
                           f"(vs {len(regex_articles)} from regex)")
                
                # Add document metadata if available
                doc_info = result.get('document_info', {})
                if doc_info:
                    logger.debug(f"Document info extracted: {doc_info}")
                
                # Add statistics if available
                stats = result.get('statistiques', {})
                if stats:
                    logger.debug(f"Extraction statistics: {stats}")
            
            return enhanced_articles if enhanced_articles else regex_articles
            
        except Exception as e:
            logger.error(f"Enhanced AI article extraction failed: {e}")
            # Fallback to basic AI if enhanced fails
            return self._enhance_articles_with_ai(text, regex_articles, metadata)
    
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
            # Use enhanced prompts if available
            if self.use_enhanced_prompts:
                return self._assess_materiality_enhanced(article)
            else:
                return self._assess_materiality_basic(article)
            
        except Exception as e:
            logger.error(f"AI materiality assessment failed: {e}")
            return self._assess_materiality_rules_based(article)
    
    def _assess_materiality_enhanced(self, article: Article) -> Article:
        """Enhanced materiality assessment using enhanced prompts."""
        try:
            # Get enhanced prompts for materiality assessment
            system_prompt, user_prompt = self.enhanced_prompts.get_prompt(
                PromptType.MATERIALITY_ASSESSMENT,
                article_content=article.content[:3000],  # Longer content for better assessment
                document_context={
                    'regulation_name': article.context.get('regulation_name', ''),
                    'document_type': article.context.get('document_type', ''),
                    'article_number': article.number,
                    'hierarchy': article.context.get('hierarchy', {}),
                    'references': article.context.get('references', [])
                }
            )
            
            # Make API call with enhanced prompts
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,  # Higher token limit for detailed assessment
                temperature=0.1,  # Lower temperature for consistency
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            # Validate and parse the response
            validation_result = self.enhanced_prompts.validate_output(
                PromptType.MATERIALITY_ASSESSMENT,
                response.content[0].text
            )
            
            if not validation_result['is_valid']:
                logger.warning(f"Enhanced materiality assessment validation failed: {validation_result.get('error', 'Unknown error')}")
                return self._assess_materiality_basic(article)
            
            # Extract materiality data from validated response
            result = validation_result.get('parsed_output', {})
            
            # Update article with enhanced assessment
            materiality_level = result.get('niveau_materialite', 'MEDIUM')
            if materiality_level in [level.value for level in MaterialityLevel]:
                article.materiality = MaterialityLevel(materiality_level)
            else:
                article.materiality = MaterialityLevel.MEDIUM
            
            article.materiality_reasoning = result.get('justification', 'Enhanced assessment completed')
            
            # Add enhanced assessment details to context
            article.context.update({
                'enhanced_assessment': True,
                'score_total': result.get('score_total', 0),
                'evaluation_details': result.get('evaluation', {}),
                'consequences_non_respect': result.get('consequences_non_respect', ''),
                'actions_requises': result.get('actions_requises', []),
                'priorite_implementation': result.get('priorite_implementation', ''),
                'validation_score': validation_result.get('validation_score', 0.8)
            })
            
            logger.debug(f"Enhanced materiality assessment completed for {article.number}: {materiality_level}")
            return article
            
        except Exception as e:
            logger.error(f"Enhanced materiality assessment failed: {e}")
            return self._assess_materiality_basic(article)
    
    def _assess_materiality_basic(self, article: Article) -> Article:
        """Basic materiality assessment using simple prompts."""
        try:
            prompt = self.materiality_prompt.format(
                article_content=article.content[:2000],  # Limit content
                regulation_name=article.context.get('regulation_name', ''),
                document_type=article.context.get('document_type', ''),
                article_number=article.number
            )
            
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                temperature=0.2,
                system="You are a regulatory compliance expert. Provide precise materiality assessments. Always provide complete, properly formatted JSON responses. Ensure all JSON objects are properly closed.",
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = self._parse_json_response(response.content[0].text)
            
            article.materiality = MaterialityLevel[result.get('materiality_level', 'MEDIUM')]
            article.materiality_reasoning = result.get('reasoning', '')
            
            return article
            
        except Exception as e:
            logger.error(f"Basic materiality assessment failed: {e}")
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
        
        # Handle the specific error pattern we're seeing: '\n    "completeness_score"'
        response_stripped = response_text.strip()
        if (response_stripped == '"completeness_score"' or 
            response_stripped == '\n    "completeness_score"' or 
            response_stripped.startswith('"completeness_score"') or
            (len(response_stripped) < 50 and 'completeness_score' in response_stripped)):
            logger.warning("Detected exact completeness_score truncation pattern - applying default scores")
            return {
                'completeness_score': 75.0,
                'reliability_score': 75.0,
                'legal_structure_score': 75.0,
                'overall_score': 75.0,
                'completeness_issues': [],
                'reliability_issues': [],
                'legal_structure_issues': [],
                'recommendations': ["Response was truncated, manual review recommended"],
                'is_truncated': True
            }
        
        # First, let's clean up common issues
        response_text = response_text.strip()
        
        # Check for the specific truncation patterns we're seeing
        if response_text.startswith('"') and not response_text.startswith('{"'):
            logger.warning(f"Detected malformed response starting with quote: {response_text[:50]}...")
            # This is likely a truncated response that started with a JSON key
            
            # Check if this is a validation response truncation
            if 'completeness_score' in response_text:
                logger.warning("Applying default validation scores for truncated response")
                return {
                    'completeness_score': 75.0,
                    'reliability_score': 75.0,
                    'legal_structure_score': 75.0,
                    'overall_score': 75.0,
                    'completeness_issues': [],
                    'reliability_issues': [],
                    'legal_structure_issues': [],
                    'recommendations': ["Response was truncated, manual review recommended"],
                    'chunk_quality': 'FAIR',
                    'method': 'truncated_response_fallback'
                }
            elif 'articles' in response_text:
                logger.warning("Applying empty articles for truncated response")
                return {
                    'articles': [],
                    'method': 'truncated_response_fallback'
                }
            elif 'materiality_level' in response_text:
                logger.warning("Applying default materiality for truncated response")
                return {
                    'materiality_level': 'MEDIUM',
                    'reasoning': 'Default due to truncated response - manual review recommended',
                    'key_impacts': [],
                    'compliance_urgency': 'MEDIUM_TERM',
                    'stakeholders_affected': [],
                    'method': 'truncated_response_fallback'
                }
        
        # Try multiple parsing strategies in order of preference
        parsing_strategies = [
            self._extract_markdown_json,
            self._extract_raw_json,
            self._extract_labeled_json,
            self._reconstruct_json_from_fragments,
            self._extract_key_values_fallback
        ]
        
        last_error = None
        for strategy in parsing_strategies:
            try:
                result = strategy(response_text)
                if result and self._validate_json_structure(result):
                    logger.debug(f"Successfully parsed JSON using {strategy.__name__}")
                    return result
            except Exception as e:
                last_error = e
                logger.debug(f"Parsing strategy {strategy.__name__} failed: {e}")
                continue
        
        logger.error(f"All JSON parsing strategies failed for response: {response_text[:500]}...")
        logger.error(f"Last error: {last_error}")
        
        # Return sensible defaults as last resort
        return {
            'completeness_score': 50.0,
            'reliability_score': 50.0,
            'legal_structure_score': 50.0,
            'overall_score': 50.0,
            'method': 'complete_fallback',
            'error': str(last_error) if last_error else 'Unknown parsing error'
        }
    
    def _extract_markdown_json(self, text: str) -> Dict[str, Any]:
        """Extract JSON from markdown code blocks."""
        patterns = [
            r'```(?:json)?\s*([\s\S]*?)\s*```',
            r'```\s*({\s*[\s\S]*?})\s*```',
            r'```\s*(\[[\s\S]*?\])\s*```'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                json_text = match.group(1).strip()
                cleaned_text = self._clean_json_text(json_text)
                if cleaned_text.startswith('{') or cleaned_text.startswith('['):
                    return json.loads(cleaned_text)
        raise ValueError("No markdown JSON found")
    
    def _extract_raw_json(self, text: str) -> Dict[str, Any]:
        """Extract raw JSON from text."""
        # Multiple patterns to find JSON structures
        patterns = [
            r'\{[\s\S]*?\}',  # Standard JSON object
            r'\[[\s\S]*?\]',  # JSON array
            r'(?:^|\n)\s*\{[\s\S]*?\}(?:\s*$|\n)',  # JSON object on its own line
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, text, re.MULTILINE)
            for match in matches:
                json_text = match.group(0).strip()
                try:
                    cleaned_text = self._clean_json_text(json_text)
                    if cleaned_text.startswith('{') or cleaned_text.startswith('['):
                        return json.loads(cleaned_text)
                except json.JSONDecodeError:
                    continue
        
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
        # Remove leading/trailing whitespace and newlines
        json_text = json_text.strip()
        
        # Remove trailing commas
        json_text = re.sub(r',\s*}', '}', json_text)
        json_text = re.sub(r',\s*]', ']', json_text)
        
        # Fix common quote issues
        json_text = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', json_text)  # Unquoted keys
        
        # Remove comments
        json_text = re.sub(r'//.*?$', '', json_text, flags=re.MULTILINE)
        json_text = re.sub(r'/\*.*?\*/', '', json_text, flags=re.DOTALL)
        
        # Handle malformed JSON that starts with partial content
        if not json_text.startswith(('{', '[')):
            # Look for the first { or [ 
            match = re.search(r'[{\[]', json_text)
            if match:
                json_text = json_text[match.start():]
        
        # Handle malformed JSON that ends with partial content
        if json_text.startswith('{'):
            # Find the last closing brace
            brace_count = 0
            last_valid_pos = -1
            for i, char in enumerate(json_text):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        last_valid_pos = i + 1
                        break
            if last_valid_pos > 0:
                json_text = json_text[:last_valid_pos]
        elif json_text.startswith('['):
            # Find the last closing bracket
            bracket_count = 0
            last_valid_pos = -1
            for i, char in enumerate(json_text):
                if char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                    if bracket_count == 0:
                        last_valid_pos = i + 1
                        break
            if last_valid_pos > 0:
                json_text = json_text[:last_valid_pos]
        
        return json_text.strip()
    
    def _reconstruct_json_from_fragments(self, text: str) -> Dict[str, Any]:
        """Reconstruct JSON from fragments - handles truncated responses."""
        result = {}
        
        # Handle cases where the response starts with a partial JSON key like '\n    "completeness_score"'
        if text.strip().startswith('"') and ':' not in text[:50]:
            # This is likely a truncated key, skip this strategy
            raise ValueError("Response appears to be truncated at the key level")
        
        # Look for key-value patterns in the text
        patterns = [
            r'"([^"]+)"\s*:\s*(\d+(?:\.\d+)?)',  # "key": number
            r'"([^"]+)"\s*:\s*"([^"]*)"',        # "key": "value"
            r'"([^"]+)"\s*:\s*\[([^\]]*)\]',     # "key": [array]
            r'"([^"]+)"\s*:\s*(true|false|null)', # "key": boolean/null
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                key = match[0]
                value = match[1]
                
                # Try to convert to appropriate type
                try:
                    if value.replace('.', '').replace('-', '').isdigit():
                        result[key] = float(value) if '.' in value else int(value)
                    elif value.lower() in ['true', 'false']:
                        result[key] = value.lower() == 'true'
                    elif value.lower() == 'null':
                        result[key] = None
                    elif key.endswith('_issues') or key.endswith('_affected') or key == 'key_impacts' or key == 'recommendations':
                        # These should be arrays
                        result[key] = []
                    else:
                        result[key] = value
                except ValueError:
                    result[key] = value
        
        if not result:
            raise ValueError("No valid key-value pairs found in fragments")
        
        return result
    
    def _extract_key_values_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback: extract key-value pairs manually."""
        result = {}
        
        # Handle the specific error pattern we're seeing: '\n    "completeness_score"'
        # This suggests the JSON is being truncated or malformed
        
        # Common patterns for validation scores with more flexible matching
        patterns = {
            'completeness_score': [
                r'["\']?completeness_score["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'["\']?completeness["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'completeness[_\s]*(?:score)?[:\s]*(\d+(?:\.\d+)?)',
            ],
            'reliability_score': [
                r'["\']?reliability_score["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'["\']?reliability["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'reliability[_\s]*(?:score)?[:\s]*(\d+(?:\.\d+)?)',
            ],
            'legal_structure_score': [
                r'["\']?legal_structure_score["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'["\']?legal_structure["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'legal[_\s]*structure[_\s]*(?:score)?[:\s]*(\d+(?:\.\d+)?)',
            ],
            'overall_score': [
                r'["\']?overall_score["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'["\']?overall["\']?\s*:\s*(\d+(?:\.\d+)?)',
                r'overall[_\s]*(?:score)?[:\s]*(\d+(?:\.\d+)?)',
            ]
        }
        
        # Special handling for materiality and articles
        materiality_patterns = [
            r'["\']?materiality_level["\']?\s*:\s*["\']?(\w+)["\']?',
            r'materiality[:\s]*["\']?(\w+)["\']?',
        ]
        
        article_patterns = [
            r'["\']?articles["\']?\s*:\s*\[([^\]]*)\]',
            r'articles[:\s]*\[([^\]]*)\]',
        ]
        
        # Try to extract validation scores
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    try:
                        result[key] = float(match.group(1))
                        break
                    except (ValueError, IndexError):
                        continue
        
        # Try to extract materiality
        for pattern in materiality_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result['materiality_level'] = match.group(1).upper()
                break
        
        # Try to extract articles list indicator
        for pattern in article_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                result['articles'] = []  # Indicate articles section was found
                break
        
        # If we found any scores, create a valid structure
        if any(key in result for key in ['completeness_score', 'reliability_score', 'legal_structure_score', 'overall_score']):
            return {
                'completeness_score': result.get('completeness_score', 75.0),
                'reliability_score': result.get('reliability_score', 75.0),
                'legal_structure_score': result.get('legal_structure_score', 75.0),
                'overall_score': result.get('overall_score', 75.0),
                'details': {
                    'method': 'fallback_extraction',
                    'timestamp': datetime.now().isoformat(),
                    'extracted_values': len(result)
                }
            }
        
        # If materiality was found
        if 'materiality_level' in result:
            return {
                'materiality_level': result['materiality_level'],
                'reasoning': 'Extracted from partial response'
            }
        
        # If articles indicator was found
        if 'articles' in result:
            return {
                'articles': [],
                'method': 'fallback_extraction'
            }
        
        raise ValueError("No extractable values found")
    
    def _validate_json_structure(self, data: Dict[str, Any]) -> bool:
        """Validate that the JSON has required fields."""
        if not isinstance(data, dict):
            return False
        
        # Check for validation score structure (with _score suffix)
        score_fields = ['completeness_score', 'reliability_score', 'legal_structure_score', 'overall_score']
        if any(field in data for field in score_fields):
            return True
        
        # Check for legacy validation score structure (without _score suffix)
        legacy_fields = ['completeness', 'reliability', 'legal_structure', 'overall']
        if any(field in data for field in legacy_fields):
            return True
        
        # Check if it's article extraction response
        if 'articles' in data:
            return True
        
        # Check if it's materiality assessment response
        if 'materiality_level' in data or 'materiality' in data:
            return True
        
        # Check if it has method indicator (from fallback)
        if 'method' in data:
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
        Extract comprehensive document structure using hierarchical parsing.
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            Dictionary containing comprehensive document structure information
        """
        # Use hierarchical parser for complete structure analysis
        document_tree = self.hierarchical_parser.parse_document(text, metadata)
        
        # Export hierarchical structure
        hierarchical_structure = self.hierarchical_parser.export_structure(document_tree)
        
        # Create table of contents
        table_of_contents = self.hierarchical_parser.create_table_of_contents(document_tree)
        
        # Extract structured articles with full context
        structured_articles = self.hierarchical_parser.extract_structured_articles(document_tree)
        
        # Legacy structure extraction (for backwards compatibility)
        legacy_structure = {
            'document_type': self._identify_document_type(text, metadata),
            'sections': self._extract_document_sections(text),
            'hierarchy': self._extract_document_hierarchy(text),
            'metadata_extracted': self._extract_document_metadata(text),
            'references': self._extract_regulatory_references(text)
        }
        
        # Comprehensive structure combining both approaches
        structure = {
            # Enhanced hierarchical structure
            'hierarchical_structure': hierarchical_structure,
            'table_of_contents': table_of_contents,
            'structured_articles': structured_articles,
            'document_tree_summary': {
                'total_titles': len([n for n in table_of_contents if n['level'] == 'TITLE']),
                'total_chapters': len([n for n in table_of_contents if n['level'] == 'CHAPTER']),
                'total_sections': len([n for n in table_of_contents if n['level'] == 'SECTION']),
                'total_subsections': len([n for n in table_of_contents if n['level'] == 'SUBSECTION']),
                'total_articles': len([n for n in table_of_contents if n['level'] == 'ARTICLE']),
                'max_depth': max([n['depth'] for n in table_of_contents], default=0)
            },
            
            # Legacy structure (backwards compatibility)
            'document_type': legacy_structure['document_type'],
            'sections': legacy_structure['sections'],
            'hierarchy': legacy_structure['hierarchy'],
            'metadata_extracted': legacy_structure['metadata_extracted'],
            'references': legacy_structure['references']
        }
        
        # Add thematic analysis
        if structured_articles:
            all_article_content = ' '.join([art['content'] for art in structured_articles])
            structure['thematic_analysis'] = self.hierarchical_parser._extract_key_themes(all_article_content)
        
        logger.info(f"Extracted comprehensive document structure: {structure['document_type']} with {structure['document_tree_summary']['total_articles']} articles across {structure['document_tree_summary']['max_depth']} levels")
        return structure
    
    def get_hierarchical_article_context(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get articles with their full hierarchical context.
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of articles with complete hierarchical context
        """
        document_tree = self.hierarchical_parser.parse_document(text, metadata)
        return self.hierarchical_parser.extract_structured_articles(document_tree)
    
    def find_articles_by_theme(self, text: str, metadata: Dict[str, Any], themes: List[str]) -> List[Dict[str, Any]]:
        """
        Find articles related to specific themes.
        
        Args:
            text: Full document text
            metadata: Document metadata
            themes: List of themes to search for
            
        Returns:
            List of articles related to the specified themes
        """
        document_tree = self.hierarchical_parser.parse_document(text, metadata)
        related_nodes = self.hierarchical_parser.find_related_articles(document_tree, themes)
        
        # Convert nodes to structured articles
        related_articles = []
        for node in related_nodes:
            context = {
                'article_number': node.number,
                'article_title': node.title,
                'content': node.content,
                'full_path': node.get_full_path(),
                'hierarchy': {}
            }
            
            # Build hierarchy context
            current = node.parent
            while current and current.level != StructureLevel.DOCUMENT:
                context['hierarchy'][current.level.name] = {
                    'number': current.number,
                    'title': current.title
                }
                current = current.parent
            
            related_articles.append(context)
        
        return related_articles
    
    def get_section_analysis(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get detailed analysis for each major section.
        
        Args:
            text: Full document text
            metadata: Document metadata
            
        Returns:
            List of section analyses
        """
        document_tree = self.hierarchical_parser.parse_document(text, metadata)
        analyses = []
        
        for child in document_tree.children:
            if child.level in [StructureLevel.TITLE, StructureLevel.CHAPTER, StructureLevel.SECTION]:
                analysis = self.hierarchical_parser.get_section_summary(child)
                analyses.append(analysis)
        
        return analyses
    
    def extract_cross_references(self, articles: List[Dict[str, Any]], language: str = 'fr') -> List[CrossReference]:
        """
        Extract all cross-references between articles.
        
        Args:
            articles: List of article dictionaries
            language: Language for reference patterns ('fr' or 'en')
            
        Returns:
            List of CrossReference objects
        """
        return self.cross_reference_resolver.extract_cross_references(articles, language)
    
    def analyze_article_network(self, articles: List[Dict[str, Any]], language: str = 'fr') -> Dict[str, Any]:
        """
        Perform comprehensive analysis of article cross-references.
        
        Args:
            articles: List of article dictionaries
            language: Language for reference patterns ('fr' or 'en')
            
        Returns:
            Dictionary containing network analysis results
        """
        # Extract cross-references
        references = self.cross_reference_resolver.extract_cross_references(articles, language)
        
        # Analyze the reference network
        network_analysis = self.cross_reference_resolver.analyze_reference_network()
        
        # Generate comprehensive report
        reference_report = self.cross_reference_resolver.generate_reference_report(references)
        
        # Combine all analyses
        analysis = {
            'total_cross_references': len(references),
            'reference_report': reference_report,
            'network_analysis': network_analysis,
            'cross_references': [
                {
                    'source': ref.source_article,
                    'target': ref.target_article,
                    'type': ref.reference_type,
                    'text': ref.reference_text,
                    'context': ref.context[:100] + '...' if len(ref.context) > 100 else ref.context
                }
                for ref in references
            ]
        }
        
        return analysis
    
    def get_article_dependencies(self, article_number: str, articles: List[Dict[str, Any]], language: str = 'fr') -> Dict[str, Any]:
        """
        Get comprehensive dependency analysis for a specific article.
        
        Args:
            article_number: Article number to analyze
            articles: List of article dictionaries
            language: Language for reference patterns ('fr' or 'en')
            
        Returns:
            Dictionary containing dependency information
        """
        # Extract cross-references first
        self.cross_reference_resolver.extract_cross_references(articles, language)
        
        # Get dependencies
        dependencies = self.cross_reference_resolver.get_article_dependencies(article_number)
        
        # Enhance with additional context
        enhanced_dependencies = {
            'article': article_number,
            'direct_dependencies': {
                'references_made': len(dependencies['references']),
                'referenced_by_count': len(dependencies['referenced_by']),
                'references': dependencies['references'],
                'referenced_by': dependencies['referenced_by']
            },
            'indirect_dependencies': {
                'indirect_references_count': len(dependencies['indirect_references']),
                'indirect_referenced_by_count': len(dependencies['indirect_referenced_by']),
                'indirect_references': dependencies['indirect_references'],
                'indirect_referenced_by': dependencies['indirect_referenced_by']
            },
            'dependency_score': self._calculate_dependency_score(dependencies),
            'centrality_metrics': self._get_article_centrality(article_number)
        }
        
        return enhanced_dependencies
    
    def find_regulatory_conflicts(self, articles: List[Dict[str, Any]], language: str = 'fr') -> List[Dict[str, Any]]:
        """
        Find potential conflicts in regulatory cross-references.
        
        Args:
            articles: List of article dictionaries
            language: Language for reference patterns ('fr' or 'en')
            
        Returns:
            List of potential conflicts
        """
        references = self.cross_reference_resolver.extract_cross_references(articles, language)
        conflicts = []
        
        # Group references by source and target
        ref_map = {}
        for ref in references:
            key = (ref.source_article, ref.target_article)
            if key not in ref_map:
                ref_map[key] = []
            ref_map[key].append(ref)
        
        # Find conflicting reference types
        for (source, target), refs in ref_map.items():
            if len(refs) > 1:
                ref_types = [ref.reference_type for ref in refs]
                unique_types = set(ref_types)
                
                # Check for conflicting types
                conflicting_pairs = [
                    ('modification', 'conformity'),
                    ('exception', 'conformity'),
                    ('modification', 'application')
                ]
                
                for type1, type2 in conflicting_pairs:
                    if type1 in unique_types and type2 in unique_types:
                        conflicts.append({
                            'source': source,
                            'target': target,
                            'conflict_type': f"{type1}_vs_{type2}",
                            'references': [
                                {
                                    'type': ref.reference_type,
                                    'text': ref.reference_text,
                                    'context': ref.context[:100]
                                }
                                for ref in refs
                            ]
                        })
        
        return conflicts
    
    def get_reference_visualization_data(self, articles: List[Dict[str, Any]], language: str = 'fr') -> Dict[str, Any]:
        """
        Get data for visualizing the cross-reference network.
        
        Args:
            articles: List of article dictionaries
            language: Language for reference patterns ('fr' or 'en')
            
        Returns:
            Visualization data for network graphs
        """
        # Extract cross-references
        self.cross_reference_resolver.extract_cross_references(articles, language)
        
        # Get visualization data
        return self.cross_reference_resolver.visualize_reference_network()
    
    def generate_regulatory_compliance_report(self, articles: List[Dict[str, Any]], language: str = 'fr') -> Dict[str, Any]:
        """
        Generate a comprehensive regulatory compliance report.
        
        Args:
            articles: List of article dictionaries
            language: Language for reference patterns ('fr' or 'en')
            
        Returns:
            Comprehensive compliance report
        """
        # Extract cross-references
        references = self.cross_reference_resolver.extract_cross_references(articles, language)
        
        # Analyze network
        network_analysis = self.cross_reference_resolver.analyze_reference_network()
        
        # Find conflicts
        conflicts = self.find_regulatory_conflicts(articles, language)
        
        # Generate report
        report = {
            'document_overview': {
                'total_articles': len(articles),
                'total_cross_references': len(references),
                'reference_density': len(references) / len(articles) if articles else 0
            },
            'compliance_indicators': {
                'well_referenced_articles': len([art for art, count in network_analysis.get('most_referenced', []) if count > 2]),
                'isolated_articles': len(network_analysis.get('isolated_articles', [])),
                'circular_references': len(network_analysis.get('circular_references', [])),
                'potential_conflicts': len(conflicts)
            },
            'reference_patterns': {
                'conformity_references': len([ref for ref in references if ref.reference_type == 'conformity']),
                'modification_references': len([ref for ref in references if ref.reference_type == 'modification']),
                'exception_references': len([ref for ref in references if ref.reference_type == 'exception']),
                'application_references': len([ref for ref in references if ref.reference_type == 'application'])
            },
            'quality_metrics': {
                'reference_completeness': self._calculate_reference_completeness(articles, references),
                'consistency_score': self._calculate_consistency_score(references),
                'complexity_score': self._calculate_complexity_score(network_analysis)
            },
            'recommendations': self._generate_compliance_recommendations(network_analysis, conflicts)
        }
        
        return report
    
    def _calculate_dependency_score(self, dependencies: Dict[str, List[str]]) -> float:
        """Calculate a dependency score for an article."""
        direct_refs = len(dependencies['references'])
        referenced_by = len(dependencies['referenced_by'])
        indirect_refs = len(dependencies['indirect_references'])
        indirect_ref_by = len(dependencies['indirect_referenced_by'])
        
        # Weighted score emphasizing being referenced (importance)
        score = (referenced_by * 3) + (indirect_ref_by * 1) + (direct_refs * 0.5) + (indirect_refs * 0.2)
        return min(score, 100.0)  # Cap at 100
    
    def _get_article_centrality(self, article_number: str) -> Dict[str, float]:
        """Get centrality metrics for an article."""
        if not self.cross_reference_resolver.reference_graph or not nx:
            return {'error': 'NetworkX not available'}
        
        graph = self.cross_reference_resolver.reference_graph
        
        if article_number not in graph:
            return {}
        
        try:
            centrality = {
                'in_degree_centrality': nx.in_degree_centrality(graph).get(article_number, 0),
                'out_degree_centrality': nx.out_degree_centrality(graph).get(article_number, 0),
                'betweenness_centrality': nx.betweenness_centrality(graph).get(article_number, 0),
                'closeness_centrality': nx.closeness_centrality(graph).get(article_number, 0)
            }
        except:
            centrality = {}
        
        return centrality
    
    def _calculate_reference_completeness(self, articles: List[Dict[str, Any]], references: List[CrossReference]) -> float:
        """Calculate how complete the cross-references are."""
        if not articles:
            return 0.0
        
        # Count articles with at least one reference
        articles_with_refs = set()
        for ref in references:
            articles_with_refs.add(ref.source_article)
            articles_with_refs.add(ref.target_article)
        
        return len(articles_with_refs) / len(articles) * 100
    
    def _calculate_consistency_score(self, references: List[CrossReference]) -> float:
        """Calculate consistency of reference patterns."""
        if not references:
            return 100.0
        
        # Check for consistent reference types between article pairs
        ref_pairs = {}
        for ref in references:
            key = (ref.source_article, ref.target_article)
            if key not in ref_pairs:
                ref_pairs[key] = set()
            ref_pairs[key].add(ref.reference_type)
        
        # Count pairs with consistent reference types
        consistent_pairs = sum(1 for types in ref_pairs.values() if len(types) == 1)
        
        return (consistent_pairs / len(ref_pairs)) * 100 if ref_pairs else 100.0
    
    def _calculate_complexity_score(self, network_analysis: Dict[str, Any]) -> float:
        """Calculate complexity score based on network structure."""
        if 'error' in network_analysis:
            return 0.0
        
        total_articles = network_analysis.get('total_articles', 0)
        total_references = network_analysis.get('total_references', 0)
        circular_refs = len(network_analysis.get('circular_references', []))
        reference_chains = len(network_analysis.get('reference_chains', []))
        
        if total_articles == 0:
            return 0.0
        
        # Complexity increases with reference density, circular references, and long chains
        density = total_references / total_articles
        complexity = (density * 20) + (circular_refs * 10) + (reference_chains * 5)
        
        return min(complexity, 100.0)  # Cap at 100
    
    def _generate_compliance_recommendations(self, network_analysis: Dict[str, Any], conflicts: List[Dict[str, Any]]) -> List[str]:
        """Generate compliance recommendations based on analysis."""
        recommendations = []
        
        if 'error' not in network_analysis:
            isolated_articles = network_analysis.get('isolated_articles', [])
            circular_refs = network_analysis.get('circular_references', [])
            
            if len(isolated_articles) > 0:
                recommendations.append(f"Review {len(isolated_articles)} isolated articles that have no cross-references")
            
            if len(circular_refs) > 0:
                recommendations.append(f"Resolve {len(circular_refs)} circular reference patterns that may cause ambiguity")
            
            if len(conflicts) > 0:
                recommendations.append(f"Address {len(conflicts)} potential conflicts in cross-reference patterns")
        
        return recommendations
    
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
    
    def extract_document_tables(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> Dict[str, List]:
        """
        Extract and categorize all tables from the document.
        
        Args:
            pdf_path: Path to the PDF file
            page_numbers: Specific pages to extract from (None for all pages)
            
        Returns:
            Dictionary with categorized tables and metadata
        """
        if not self.table_extractor:
            logger.warning("Table extractor not initialized")
            return {'tables': [], 'categories': {}, 'error': 'Table extractor not available'}
        
        try:
            logger.info(f"Extracting tables from document: {pdf_path}")
            tables_data = self.table_extractor.extract_tables(pdf_path, page_numbers)
            
            # Log extraction results
            total_tables = len(tables_data.get('tables', []))
            categories = tables_data.get('categories', {})
            logger.info(f"Extracted {total_tables} tables from {pdf_path}")
            
            if categories:
                for category, count in categories.items():
                    logger.info(f"  - {category}: {count} tables")
            
            return tables_data
            
        except Exception as e:
            logger.error(f"Failed to extract tables from {pdf_path}: {e}")
            return {'tables': [], 'categories': {}, 'error': str(e)}
    
    def analyze_article_tables(self, articles: List[Article]) -> Dict[str, Any]:
        """
        Find and extract tables from article content.
        
        Args:
            articles: List of Article objects to analyze
            
        Returns:
            Dictionary with table analysis results
        """
        if not self.table_extractor:
            logger.warning("Table extractor not initialized")
            return {'article_tables': [], 'total_tables': 0, 'error': 'Table extractor not available'}
        
        try:
            article_tables = []
            total_tables = 0
            
            for article in articles:
                # Create article context for table extraction
                article_context = {
                    'article_number': article.number,
                    'article_title': article.title or '',
                    'materiality': article.materiality.value,
                    'regulation_name': article.context.get('regulation_name', ''),
                    'document_type': article.context.get('document_type', '')
                }
                
                # Find regulatory tables in article content
                tables = self.table_extractor.find_regulatory_tables(
                    article.content, 
                    article_context
                )
                
                if tables:
                    article_tables.extend(tables)
                    total_tables += len(tables)
                    logger.info(f"Found {len(tables)} tables in Article {article.number}")
            
            logger.info(f"Total tables found in articles: {total_tables}")
            
            return {
                'article_tables': article_tables,
                'total_tables': total_tables,
                'articles_with_tables': len([art for art in articles 
                                           if any(table['article_number'] == art.number 
                                                for table in article_tables)])
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze article tables: {e}")
            return {'article_tables': [], 'total_tables': 0, 'error': str(e)}
    
    def export_tables_report(self, pdf_path: str, articles: Optional[List[Article]] = None, 
                           output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive table extraction report.
        
        Args:
            pdf_path: Path to the PDF file
            articles: Optional list of articles to analyze
            output_path: Optional output path for Excel report
            
        Returns:
            Path to the generated report file
        """
        if not self.table_extractor:
            logger.warning("Table extractor not initialized")
            return ""
        
        try:
            # Extract all document tables
            document_tables = self.extract_document_tables(pdf_path)
            
            # Analyze article tables if articles provided
            article_analysis = {}
            if articles:
                article_analysis = self.analyze_article_tables(articles)
            
            # Generate output path if not provided
            if not output_path:
                base_name = os.path.splitext(os.path.basename(pdf_path))[0]
                output_path = f"{base_name}_tables_report.xlsx"
            
            # Export to Excel using table extractor
            final_path = self.table_extractor.export_to_excel(
                document_tables.get('tables', []),
                output_path,
                {
                    'source_document': pdf_path,
                    'extraction_date': datetime.now().isoformat(),
                    'total_document_tables': len(document_tables.get('tables', [])),
                    'total_article_tables': article_analysis.get('total_tables', 0),
                    'articles_analyzed': len(articles) if articles else 0
                }
            )
            
            logger.info(f"Table extraction report exported to: {final_path}")
            return final_path
            
        except Exception as e:
            logger.error(f"Failed to generate tables report: {e}")
            return ""
    
    def enhance_text_with_ocr(self, raw_text: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Enhance text quality using COBAC OCR parser.
        
        Args:
            raw_text: Raw text from initial extraction
            image_path: Optional path to source image for re-processing
            
        Returns:
            Dictionary with enhanced text and processing metadata
        """
        if not self.ocr_parser:
            logger.warning("OCR parser not initialized")
            return {'enhanced_text': raw_text, 'confidence_score': 0.5, 'processing_method': 'no_enhancement'}
        
        try:
            logger.info("Enhancing text with COBAC OCR parser")
            enhancement_result = self.ocr_parser.enhance_extracted_text(raw_text, image_path)
            
            logger.info(f"OCR enhancement completed - Confidence: {enhancement_result['confidence_score']:.2f}, "
                       f"Corrections: {enhancement_result['corrections_made']}")
            
            if enhancement_result['quality_issues']:
                logger.warning(f"Quality issues detected: {', '.join(enhancement_result['quality_issues'])}")
            
            return enhancement_result
            
        except Exception as e:
            logger.error(f"OCR enhancement failed: {e}")
            return {
                'enhanced_text': raw_text,
                'confidence_score': 0.5,
                'corrections_made': 0,
                'quality_issues': [f'Enhancement failed: {str(e)}'],
                'processing_method': 'enhancement_error'
            }


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
        
        # Check if we should use AI based on recent error patterns
        error_manager = get_error_manager()
        should_use_ai = self._should_use_ai_validation(error_manager)
        
        # Validate each chunk (but limit AI usage if too many errors)
        chunks = document_data.get('chunks', [])
        for i, chunk in enumerate(chunks):
            if should_use_ai or i < 3:  # Always validate first 3 chunks, then check error rate
                score = self.validator.validate_chunk(chunk, document_data['metadata'])
            else:
                # Use default scoring if AI is having too many issues
                score = self.validator._get_default_validation_score()
                logger.info(f"Skipping AI validation for chunk {i} due to high error rate")
            
            chunk['validation_score'] = score.to_dict()
            validation_results['chunk_validations'].append(score)
        
        # Extract and validate articles (use AI only if it's working well)
        if should_use_ai:
            articles = self.validator.extract_articles(
                document_data.get('cleaned_text', ''),
                document_data['metadata']
            )
        else:
            # Fall back to regex-only extraction
            logger.info("Using regex-only article extraction due to AI errors")
            articles = safe_execute(
                self.validator._extract_articles_regex,
                document_data.get('cleaned_text', ''),
                document_data['metadata'],
                default_return=[]
            )
        
        # Extract document structure for COBAC documents
        document_structure = self.validator.extract_document_structure(
            document_data.get('cleaned_text', ''),
            document_data['metadata']
        )
        document_data['document_structure'] = document_structure
        
        # Assess materiality for each article (use AI only if it's working well)
        for article in articles:
            if should_use_ai:
                article = self.validator.assess_materiality(article)
            else:
                # Use rules-based assessment
                article = self.validator._assess_materiality_rules_based(article)
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
    
    def _should_use_ai_validation(self, error_manager) -> bool:
        """Determine if AI validation should be used based on recent error patterns."""
        # Get recent AI-related errors (last 5 minutes)
        recent_ai_errors = [
            err for err in error_manager.error_history[-20:]  # Check last 20 errors
            if 'AI' in err.function_name or 'validate_chunk' in err.function_name or 'extract_articles' in err.function_name
            and (datetime.now() - err.timestamp).total_seconds() < 300  # Last 5 minutes
        ]
        
        # If more than 10 AI errors in the last 5 minutes, skip AI
        if len(recent_ai_errors) > 10:
            logger.warning(f"High AI error rate detected ({len(recent_ai_errors)} errors in 5 min), switching to fallback methods")
            return False
        
        # Check for specific truncation errors
        truncation_errors = [
            err for err in recent_ai_errors
            if 'completeness_score' in str(err.error_message) or 'articles' in str(err.error_message)
        ]
        
        # If more than 5 truncation errors, skip AI
        if len(truncation_errors) > 5:
            logger.warning(f"High truncation error rate detected ({len(truncation_errors)} errors), switching to fallback methods")
            return False
        
        return True
    
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