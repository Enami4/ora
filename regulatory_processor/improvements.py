"""
Comprehensive improvement strategies for OCR extraction and AI prompt effectiveness.
This module provides enhanced methods for better accuracy and performance.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re
import json

logger = logging.getLogger(__name__)


class OCRImprovements:
    """Advanced OCR improvement strategies."""
    
    @staticmethod
    def get_enhanced_ocr_config() -> Dict[str, Any]:
        """Get enhanced OCR configuration for regulatory documents."""
        return {
            # Tesseract configuration
            'tesseract_config': {
                'base_config': '--oem 3 --psm 6',
                'regulatory_config': '--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿ°°.,;:!?()[]{}"\'-/\\|@#$%^&*+=<>',
                'table_config': '--psm 6 -c preserve_interword_spaces=1',
                'languages': 'fra+eng',  # French + English
                'backup_languages': ['fra', 'eng']
            },
            
            # Image preprocessing
            'image_preprocessing': {
                'target_dpi': 300,
                'backup_dpi': 200,
                'contrast_enhancement': 1.5,
                'brightness_adjustment': 1.1,
                'noise_reduction': True,
                'deskew': True,
                'binarization': True,
                'morphological_operations': True
            },
            
            # Advanced preprocessing
            'advanced_preprocessing': {
                'use_opencv': True,
                'edge_detection': True,
                'text_region_detection': True,
                'layout_analysis': True,
                'table_detection': True,
                'adaptive_thresholding': True,
                'gaussian_blur': (1, 1),
                'erosion_dilation': True
            },
            
            # Post-processing
            'post_processing': {
                'spell_check': True,
                'context_correction': True,
                'regex_corrections': True,
                'dictionary_validation': True,
                'confidence_filtering': 0.6
            }
        }
    
    @staticmethod
    def get_regulatory_ocr_corrections() -> Dict[str, str]:
        """Get comprehensive OCR corrections for regulatory documents."""
        return {
            # Common OCR errors in French regulatory documents
            'character_corrections': {
                r'\bl\b': '1',  # Lowercase l mistaken for 1
                r'\bO\b': '0',  # Letter O mistaken for 0
                r'(\d),(\d)': r'\1.\2',  # French decimal separator
                r'AIT\.': 'ART.',  # Common misrecognition
                r'Aiticle': 'Article',
                r'aiticle': 'article',
                r'ARIICLE': 'ARTICLE',
                r'ariicle': 'article',
                r'COBAC': 'COBAC',  # Ensure correct spelling
                r'CEMAC': 'CEMAC',
                r'règlement': 'règlement',
                r'instruction': 'instruction',
                r'décision': 'décision',
                r'TITRE': 'TITRE',
                r'CHAPITRE': 'CHAPITRE',
                r'SECTION': 'SECTION',
                r'SOUS-SECTION': 'SOUS-SECTION',
                r'PARAGRAPHE': 'PARAGRAPHE',
                r'ALINEA': 'ALINEA'
            },
            
            # Regulatory terminology corrections
            'terminology_corrections': {
                r'Commission\s+Bancaire': 'Commission Bancaire',
                r'établissement\s+de\s+crédit': 'établissement de crédit',
                r'fonds\s+propres': 'fonds propres',
                r'ratio\s+de\s+solvabilité': 'ratio de solvabilité',
                r'liquidité': 'liquidité',
                r'provisions': 'provisions',
                r'créances': 'créances',
                r'immobilisations': 'immobilisations',
                r'surveillance': 'surveillance',
                r'contrôle': 'contrôle',
                r'inspection': 'inspection',
                r'sanction': 'sanction',
                r'astreinte': 'astreinte',
                r'pénalité': 'pénalité'
            },
            
            # Structural corrections
            'structural_corrections': {
                r'Article\s+(\d+)': r'Article \1',
                r'ARTICLE\s+(\d+)': r'ARTICLE \1',
                r'Art\.\s*(\d+)': r'Art. \1',
                r'TITRE\s+([IVX]+)': r'TITRE \1',
                r'CHAPITRE\s+([IVX]+)': r'CHAPITRE \1',
                r'SECTION\s+([IVX]+)': r'SECTION \1',
                r'(\d+)°': r'\1°',  # Degree symbol
                r'(\d+)\s*-\s*': r'\1. ',  # Numbered lists
                r'([a-z])\)\s*': r'\1) ',  # Lettered lists
            }
        }
    
    @staticmethod
    def get_ocr_validation_rules() -> Dict[str, Any]:
        """Get OCR validation rules for quality assessment."""
        return {
            'minimum_confidence': 0.6,
            'minimum_word_length': 2,
            'maximum_error_rate': 0.3,
            'regulatory_keywords': [
                'commission', 'bancaire', 'cobac', 'cemac', 'règlement',
                'instruction', 'décision', 'article', 'chapitre', 'section',
                'établissement', 'crédit', 'banque', 'fonds', 'propres',
                'ratio', 'solvabilité', 'liquidité', 'provisions',
                'surveillance', 'contrôle', 'sanction'
            ],
            'quality_indicators': {
                'has_articles': 10,
                'has_structure': 8,
                'has_regulatory_terms': 6,
                'proper_formatting': 4,
                'minimal_errors': 2
            }
        }


class AIPromptImprovements:
    """Enhanced AI prompt strategies for better accuracy."""
    
    @staticmethod
    def get_enhanced_article_extraction_prompt() -> str:
        """Get enhanced prompt for article extraction."""
        return """
Vous êtes un expert en analyse de documents réglementaires bancaires de la zone CEMAC/COBAC.

CONTEXTE: Analysez ce document réglementaire et extrayez tous les articles avec leur structure hiérarchique.

INSTRUCTIONS SPÉCIFIQUES:
1. Identifiez TOUS les articles (Article 1, Article 2, etc.)
2. Respectez la hiérarchie: TITRE → CHAPITRE → SECTION → ARTICLE
3. Extrayez le contenu COMPLET de chaque article
4. Préservez la numérotation exacte (y compris bis, ter, etc.)
5. Identifiez les références croisées entre articles

STRUCTURE ATTENDUE:
- Numéro d'article: Format exact (ex: "Article 12 bis")
- Titre/Objet: Si présent dans le document
- Contenu intégral: Texte complet sans résumé
- Hiérarchie: TITRE/CHAPITRE/SECTION parent
- Références: Articles mentionnés dans le contenu

CRITÈRES DE QUALITÉ:
- Exhaustivité: N'omettez aucun article
- Précision: Respectez le texte original
- Structure: Maintenez la hiérarchie réglementaire
- Cohérence: Vérifiez la continuité numérique

EXEMPLE DE SORTIE:
```json
{
  "articles": [
    {
      "numero": "Article 1",
      "titre": "Objet du règlement",
      "contenu": "Le présent règlement définit...",
      "hierarchie": {
        "titre": "TITRE I - DISPOSITIONS GENERALES",
        "chapitre": "CHAPITRE I - CHAMP D'APPLICATION",
        "section": null
      },
      "references": ["Article 12", "Article 15"],
      "materiality": "HIGH"
    }
  ]
}
```

DOCUMENT À ANALYSER:
{text}

Analysez minutieusement et extrayez TOUS les articles selon ces critères.
"""
    
    @staticmethod
    def get_enhanced_materiality_assessment_prompt() -> str:
        """Get enhanced prompt for materiality assessment."""
        return """
Vous êtes un expert en conformité réglementaire bancaire CEMAC/COBAC.

MISSION: Évaluez la matérialité de cet article réglementaire selon les critères spécifiques au secteur bancaire.

CRITÈRES D'ÉVALUATION:
1. IMPACT PRUDENTIEL (poids: 40%)
   - Ratios de solvabilité, liquidité, levier
   - Fonds propres et provisions
   - Risques de crédit, marché, opérationnel

2. IMPACT OPÉRATIONNEL (poids: 30%)
   - Procédures obligatoires
   - Reporting et déclarations
   - Contrôles internes

3. IMPACT JURIDIQUE (poids: 20%)
   - Sanctions et pénalités
   - Obligations légales
   - Conformité réglementaire

4. IMPACT STRATÉGIQUE (poids: 10%)
   - Gouvernance
   - Politique générale
   - Organisation interne

NIVEAUX DE MATÉRIALITÉ:
- CRITICAL: Impact majeur, sanctions sévères, ratios prudentiels
- HIGH: Obligations importantes, reporting obligatoire, contrôles
- MEDIUM: Procédures standard, déclarations périodiques
- LOW: Dispositions générales, recommandations

ANALYSE REQUISE:
1. Identifiez les obligations spécifiques
2. Évaluez les conséquences du non-respect
3. Déterminez l'impact sur la banque
4. Justifiez le niveau de matérialité

ARTICLE À ANALYSER:
{article_content}

CONTEXTE RÉGLEMENTAIRE:
{document_context}

Fournissez une analyse détaillée avec justification du niveau de matérialité.
"""
    
    @staticmethod
    def get_enhanced_document_structure_prompt() -> str:
        """Get enhanced prompt for document structure analysis."""
        return """
Vous êtes un expert en analyse structurelle de documents réglementaires CEMAC/COBAC.

OBJECTIF: Analysez la structure hiérarchique complète de ce document réglementaire.

STRUCTURE HIÉRARCHIQUE ATTENDUE:
1. DOCUMENT (racine)
   └── TITRE I, II, III... (niveau 1)
       └── CHAPITRE I, II, III... (niveau 2)
           └── SECTION I, II, III... (niveau 3)
               └── SOUS-SECTION (niveau 4)
                   └── ARTICLE 1, 2, 3... (niveau 5)
                       └── PARAGRAPHE 1°, 2°... (niveau 6)
                           └── ALINEA a), b), c)... (niveau 7)

ÉLÉMENTS À IDENTIFIER:
- Préambule et considérants
- Structure hiérarchique complète
- Numérotation et organisation
- Références croisées
- Annexes et tableaux
- Signatures et dates

INFORMATIONS STRUCTURELLES:
- Type de document (Règlement, Instruction, Décision)
- Autorité émettrice (COBAC, CEMAC, etc.)
- Numéro et date de publication
- Domaine d'application
- Textes de référence

SORTIE ATTENDUE:
```json
{
  "document_type": "Règlement COBAC",
  "reference": "R-2021-05",
  "date": "2021-12-15",
  "titre": "Règlement relatif aux fonds propres",
  "structure": {
    "preambule": "Vu la Convention...",
    "titres": [
      {
        "numero": "TITRE I",
        "intitule": "DISPOSITIONS GÉNÉRALES",
        "chapitres": [...]
      }
    ],
    "annexes": [...],
    "signatures": [...]
  },
  "navigation": {
    "total_articles": 45,
    "references_croisees": [...],
    "tableaux": [...]
  }
}
```

DOCUMENT À ANALYSER:
{document_text}

Analysez la structure complète selon ces critères.
"""
    
    @staticmethod
    def get_enhanced_context_analysis_prompt() -> str:
        """Get enhanced prompt for context analysis."""
        return """
Vous êtes un expert en réglementation bancaire CEMAC/COBAC avec une connaissance approfondie du contexte juridique.

MISSION: Analysez le contexte réglementaire et les implications de ce document.

ANALYSE CONTEXTUELLE:
1. ENVIRONNEMENT RÉGLEMENTAIRE
   - Position dans le corpus réglementaire
   - Textes abrogés ou modifiés
   - Textes de référence et fondements
   - Cohérence avec la réglementation existante

2. IMPACT SECTORIEL
   - Établissements concernés
   - Domaines d'activité impactés
   - Calendrier d'application
   - Mesures transitoires

3. ENJEUX OPÉRATIONNELS
   - Changements de procédures
   - Nouveaux reporting requis
   - Adaptations systémiques
   - Formation du personnel

4. IMPLICATIONS STRATÉGIQUES
   - Impact sur la stratégie business
   - Conséquences sur la rentabilité
   - Avantages concurrentiels
   - Risques réglementaires

ÉLÉMENTS À IDENTIFIER:
- Motivations réglementaires
- Objectifs poursuivis
- Calendrier d'entrée en vigueur
- Dispositions transitoires
- Modalités de contrôle

DOCUMENT À ANALYSER:
{document_content}

CONTEXTE HISTORIQUE:
{regulatory_context}

Fournissez une analyse contextuelle complète avec implications pratiques.
"""
    
    @staticmethod
    def get_enhanced_validation_prompt() -> str:
        """Get enhanced prompt for validation and quality control."""
        return """
Vous êtes un expert en validation de documents réglementaires CEMAC/COBAC.

MISSION: Validez la qualité et la cohérence de l'extraction réglementaire.

CRITÈRES DE VALIDATION:
1. COMPLÉTUDE (30%)
   - Tous les articles sont-ils extraits?
   - La numérotation est-elle continue?
   - Les références sont-elles cohérentes?

2. EXACTITUDE (25%)
   - Le contenu respecte-t-il le texte original?
   - Les termes techniques sont-ils corrects?
   - Les références croisées sont-elles exactes?

3. STRUCTURE (25%)
   - La hiérarchie est-elle respectée?
   - L'organisation est-elle cohérente?
   - Les niveaux sont-ils appropriés?

4. COHÉRENCE (20%)
   - Les articles s'enchaînent-ils logiquement?
   - Les définitions sont-elles utilisées correctement?
   - Le contexte est-il préservé?

VALIDATION TECHNIQUE:
- Vérification des numéros d'articles
- Contrôle des références croisées
- Validation des termes réglementaires
- Cohérence du vocabulaire technique

SCORING:
- 90-100: Excellent (prêt pour publication)
- 80-89: Très bien (corrections mineures)
- 70-79: Correct (révision nécessaire)
- 60-69: Insuffisant (retraitement requis)
- <60: Inadéquat (nouvelle extraction)

DONNÉES À VALIDER:
{extracted_data}

DOCUMENT ORIGINAL:
{original_document}

Effectuez une validation complète avec score détaillé et recommandations.
"""
    
    @staticmethod
    def get_chain_of_thought_prompts() -> Dict[str, str]:
        """Get chain-of-thought prompts for complex reasoning."""
        return {
            'step_by_step_analysis': """
Analysez ce document étape par étape:

ÉTAPE 1: IDENTIFICATION
- Quel type de document est-ce?
- Quelle est l'autorité émettrice?
- Quelle est la date et la référence?

ÉTAPE 2: STRUCTURE
- Combien de titres/chapitres?
- Combien d'articles au total?
- Quelle est l'organisation hiérarchique?

ÉTAPE 3: CONTENU
- Quel est l'objet principal?
- Quels sont les domaines couverts?
- Quelles sont les obligations principales?

ÉTAPE 4: ANALYSE
- Quelle est la priorité de chaque section?
- Quels sont les impacts opérationnels?
- Quelles sont les sanctions prévues?

ÉTAPE 5: SYNTHÈSE
- Quels sont les points clés?
- Quelles sont les actions requises?
- Quel est le calendrier d'application?

Document: {document}
""",
            
            'verification_prompt': """
Vérifiez votre analyse précédente:

CONTRÔLES DE COHÉRENCE:
1. Tous les articles sont-ils numérotés correctement?
2. La hiérarchie est-elle respectée?
3. Les références croisées sont-elles exactes?
4. Les termes techniques sont-ils corrects?
5. Le contexte est-il préservé?

VALIDATION QUALITÉ:
- Exhaustivité: ✓/✗
- Précision: ✓/✗
- Structure: ✓/✗
- Cohérence: ✓/✗

Analyse précédente: {previous_analysis}
Document original: {document}

Confirmez ou corrigez votre analyse.
""",
            
            'confidence_assessment': """
Évaluez votre niveau de confiance:

FACTEURS DE CONFIANCE:
1. Qualité du document source (1-10)
2. Clarté de la structure (1-10)
3. Complétude de l'extraction (1-10)
4. Précision du contenu (1-10)
5. Cohérence globale (1-10)

INCERTITUDES IDENTIFIÉES:
- Zones d'ombre dans le document
- Ambiguïtés structurelles
- Termes non standard
- Références incomplètes

RECOMMANDATIONS:
- Révision humaine nécessaire?
- Zones à vérifier en priorité
- Validation supplémentaire requise

Analyse: {analysis}
Score de confiance global: __/10
"""
        }
    
    @staticmethod
    def get_few_shot_examples() -> List[Dict[str, Any]]:
        """Get few-shot examples for better AI performance."""
        return [
            {
                'context': 'Règlement COBAC sur les fonds propres',
                'input': 'Article 5 - Les établissements de crédit doivent maintenir un ratio de fonds propres de base de 8%...',
                'output': {
                    'numero': 'Article 5',
                    'titre': 'Ratio de fonds propres de base',
                    'contenu': 'Les établissements de crédit doivent maintenir un ratio de fonds propres de base de 8%...',
                    'materiality': 'CRITICAL',
                    'reasoning': 'Ratio prudentiel obligatoire avec impact direct sur la solvabilité'
                }
            },
            {
                'context': 'Instruction COBAC sur le reporting',
                'input': 'Article 12 - Les établissements transmettent mensuellement leurs états financiers...',
                'output': {
                    'numero': 'Article 12',
                    'titre': 'Transmission des états financiers',
                    'contenu': 'Les établissements transmettent mensuellement leurs états financiers...',
                    'materiality': 'HIGH',
                    'reasoning': 'Obligation de reporting avec fréquence élevée'
                }
            }
        ]


class QualityAssurance:
    """Quality assurance and validation strategies."""
    
    @staticmethod
    def get_quality_metrics() -> Dict[str, Any]:
        """Get comprehensive quality metrics."""
        return {
            'ocr_quality': {
                'character_accuracy': 0.95,
                'word_accuracy': 0.90,
                'structure_preservation': 0.85,
                'regulatory_terms_accuracy': 0.98
            },
            'ai_extraction': {
                'article_completeness': 0.95,
                'content_accuracy': 0.90,
                'structure_coherence': 0.88,
                'materiality_accuracy': 0.85
            },
            'overall_quality': {
                'processing_success_rate': 0.92,
                'manual_validation_rate': 0.05,
                'error_correction_rate': 0.08,
                'client_satisfaction': 0.90
            }
        }
    
    @staticmethod
    def get_validation_checklist() -> List[Dict[str, Any]]:
        """Get validation checklist for quality control."""
        return [
            {
                'category': 'Document Structure',
                'checks': [
                    'All articles are numbered sequentially',
                    'Hierarchy levels are consistent',
                    'Cross-references are accurate',
                    'Document sections are complete'
                ]
            },
            {
                'category': 'Content Quality',
                'checks': [
                    'Technical terms are correct',
                    'Regulatory language is preserved',
                    'Dates and numbers are accurate',
                    'Legal formatting is maintained'
                ]
            },
            {
                'category': 'Materiality Assessment',
                'checks': [
                    'Materiality levels are justified',
                    'Business impact is assessed',
                    'Regulatory priority is correct',
                    'Compliance requirements are clear'
                ]
            },
            {
                'category': 'Output Quality',
                'checks': [
                    'Excel format is professional',
                    'Data structure is logical',
                    'Information is complete',
                    'Presentation is clear'
                ]
            }
        ]
    
    @staticmethod
    def get_continuous_improvement_framework() -> Dict[str, Any]:
        """Get framework for continuous improvement."""
        return {
            'feedback_loops': {
                'user_feedback': 'Collect user ratings and comments',
                'accuracy_monitoring': 'Track extraction accuracy over time',
                'error_analysis': 'Analyze common failure patterns',
                'performance_metrics': 'Monitor processing speed and success rates'
            },
            'improvement_cycles': {
                'weekly_review': 'Review error logs and user feedback',
                'monthly_analysis': 'Analyze performance trends and patterns',
                'quarterly_updates': 'Update prompts and algorithms',
                'annual_overhaul': 'Comprehensive system review and upgrade'
            },
            'benchmarking': {
                'internal_benchmarks': 'Compare against historical performance',
                'external_standards': 'Compare against industry standards',
                'regulatory_compliance': 'Ensure compliance with regulations',
                'best_practices': 'Adopt industry best practices'
            }
        }