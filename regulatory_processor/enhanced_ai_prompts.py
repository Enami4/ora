"""
Enhanced AI prompt engineering for better regulatory document analysis.
"""

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from .improvements import AIPromptImprovements

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Types of AI prompts for different analysis tasks."""
    ARTICLE_EXTRACTION = "article_extraction"
    MATERIALITY_ASSESSMENT = "materiality_assessment"
    DOCUMENT_STRUCTURE = "document_structure"
    CONTEXT_ANALYSIS = "context_analysis"
    VALIDATION = "validation"
    CHAIN_OF_THOUGHT = "chain_of_thought"


@dataclass
class PromptTemplate:
    """Template for AI prompts with context and examples."""
    system_prompt: str
    user_prompt: str
    few_shot_examples: List[Dict[str, Any]]
    validation_criteria: Dict[str, Any]
    expected_output_format: str


class EnhancedAIPrompts:
    """Enhanced AI prompt engineering system."""
    
    def __init__(self):
        self.improvements = AIPromptImprovements()
        self.templates = self._initialize_templates()
    
    def _initialize_templates(self) -> Dict[PromptType, PromptTemplate]:
        """Initialize prompt templates."""
        return {
            PromptType.ARTICLE_EXTRACTION: self._create_article_extraction_template(),
            PromptType.MATERIALITY_ASSESSMENT: self._create_materiality_template(),
            PromptType.DOCUMENT_STRUCTURE: self._create_structure_template(),
            PromptType.CONTEXT_ANALYSIS: self._create_context_template(),
            PromptType.VALIDATION: self._create_validation_template(),
            PromptType.CHAIN_OF_THOUGHT: self._create_chain_of_thought_template()
        }
    
    def _create_article_extraction_template(self) -> PromptTemplate:
        """Create enhanced article extraction template."""
        return PromptTemplate(
            system_prompt="""Vous êtes un expert en analyse de documents réglementaires bancaires CEMAC/COBAC avec 15 ans d'expérience.

EXPERTISE: Analyse exhaustive des textes réglementaires bancaires, extraction structurée des articles, respect de la hiérarchie juridique.

OBJECTIF: Extraire TOUS les articles avec leur structure hiérarchique complète et leur contenu intégral.

MÉTHODOLOGIE:
1. Lecture séquentielle du document
2. Identification des structures hiérarchiques
3. Extraction exhaustive des articles
4. Validation de la continuité numérique
5. Vérification des références croisées

CRITÈRES DE QUALITÉ:
- Exhaustivité: 100% des articles extraits
- Précision: Texte original préservé
- Structure: Hiérarchie respectée
- Cohérence: Numérotation validée""",
            
            user_prompt="""DOCUMENT RÉGLEMENTAIRE À ANALYSER:

{document_text}

INSTRUCTIONS SPÉCIFIQUES:
1. Identifiez TOUS les articles (ne pas omettre d'articles)
2. Respectez la hiérarchie: TITRE → CHAPITRE → SECTION → ARTICLE
3. Extrayez le contenu COMPLET (pas de résumé)
4. Préservez la numérotation exacte (bis, ter, quater)
5. Identifiez les références croisées

STRUCTURE DE SORTIE REQUISE:
```json
{
  "document_info": {
    "type": "Règlement/Instruction/Décision",
    "reference": "R-XXXX-XX",
    "date": "YYYY-MM-DD",
    "titre": "Titre complet du document"
  },
  "articles": [
    {
      "numero": "Article X",
      "titre": "Titre/objet si présent",
      "contenu": "Contenu intégral de l'article",
      "hierarchie": {
        "titre": "TITRE X - Description",
        "chapitre": "CHAPITRE X - Description",
        "section": "SECTION X - Description"
      },
      "references": ["Article Y", "Article Z"],
      "position": {
        "ligne_debut": 123,
        "ligne_fin": 145
      }
    }
  ],
  "structure_hierarchique": {
    "titres": [...],
    "chapitres": [...],
    "sections": [...]
  },
  "statistiques": {
    "total_articles": 45,
    "articles_avec_references": 12,
    "niveaux_hierarchiques": 4
  }
}
```

VALIDATION:
- Vérifiez que tous les articles sont extraits
- Confirmez la continuité numérique
- Validez les références croisées
- Assurez-vous de la cohérence structurelle

Procédez à l'analyse exhaustive maintenant.""",
            
            few_shot_examples=self.improvements.get_few_shot_examples(),
            
            validation_criteria={
                'completeness': 0.95,
                'accuracy': 0.90,
                'structure_coherence': 0.88,
                'reference_accuracy': 0.85
            },
            
            expected_output_format="JSON with articles, hierarchy, and metadata"
        )
    
    def _create_materiality_template(self) -> PromptTemplate:
        """Create enhanced materiality assessment template."""
        return PromptTemplate(
            system_prompt="""Vous êtes un expert en conformité réglementaire bancaire CEMAC/COBAC avec une spécialisation en évaluation des risques.

EXPERTISE: Évaluation de la matérialité des exigences réglementaires, impact sur les établissements bancaires, priorisation des obligations de conformité.

CADRE D'ÉVALUATION:
- Risques prudentiels (ratios, fonds propres)
- Obligations opérationnelles (procédures, reporting)
- Conséquences juridiques (sanctions, pénalités)
- Impact stratégique (gouvernance, organisation)

MÉTHODOLOGIE:
1. Analyse du contenu réglementaire
2. Évaluation des impacts multidimensionnels
3. Quantification des risques
4. Classification par niveau de matérialité
5. Justification détaillée""",
            
            user_prompt="""ARTICLE RÉGLEMENTAIRE À ÉVALUER:

{article_content}

CONTEXTE DOCUMENTAIRE:
{document_context}

GRILLE D'ÉVALUATION:

1. IMPACT PRUDENTIEL (40%)
   - Ratios de solvabilité/liquidité: □ Aucun □ Faible □ Moyen □ Fort □ Critique
   - Fonds propres/provisions: □ Aucun □ Faible □ Moyen □ Fort □ Critique
   - Risques (crédit/marché/opérationnel): □ Aucun □ Faible □ Moyen □ Fort □ Critique

2. IMPACT OPÉRATIONNEL (30%)
   - Procédures obligatoires: □ Aucun □ Faible □ Moyen □ Fort □ Critique
   - Reporting/déclarations: □ Aucun □ Faible □ Moyen □ Fort □ Critique
   - Contrôles internes: □ Aucun □ Faible □ Moyen □ Fort □ Critique

3. IMPACT JURIDIQUE (20%)
   - Sanctions/pénalités: □ Aucunes □ Faibles □ Moyennes □ Fortes □ Critiques
   - Obligations légales: □ Aucunes □ Faibles □ Moyennes □ Fortes □ Critiques
   - Conformité réglementaire: □ Aucune □ Faible □ Moyenne □ Forte □ Critique

4. IMPACT STRATÉGIQUE (10%)
   - Gouvernance: □ Aucun □ Faible □ Moyen □ Fort □ Critique
   - Organisation: □ Aucun □ Faible □ Moyen □ Fort □ Critique
   - Politique générale: □ Aucun □ Faible □ Moyen □ Fort □ Critique

NIVEAUX DE MATÉRIALITÉ:
- CRITICAL (>90): Impact majeur, sanctions sévères, ratios prudentiels
- HIGH (70-90): Obligations importantes, reporting obligatoire
- MEDIUM (50-70): Procédures standard, déclarations périodiques
- LOW (<50): Dispositions générales, recommandations

ANALYSE REQUISE:
```json
{
  "evaluation": {
    "impact_prudentiel": {
      "score": 0-100,
      "details": "Analyse détaillée",
      "elements": ["ratio X", "provision Y"]
    },
    "impact_operationnel": {
      "score": 0-100,
      "details": "Analyse détaillée",
      "elements": ["procédure X", "reporting Y"]
    },
    "impact_juridique": {
      "score": 0-100,
      "details": "Analyse détaillée",
      "elements": ["sanction X", "obligation Y"]
    },
    "impact_strategique": {
      "score": 0-100,
      "details": "Analyse détaillée",
      "elements": ["gouvernance X", "politique Y"]
    }
  },
  "score_total": 0-100,
  "niveau_materialite": "CRITICAL/HIGH/MEDIUM/LOW",
  "justification": "Justification détaillée du niveau assigné",
  "consequences_non_respect": "Conséquences du non-respect",
  "actions_requises": ["Action 1", "Action 2"],
  "priorite_implementation": "Immédiate/30 jours/3 mois/6 mois",
  "ressources_necessaires": "Estimation des ressources"
}
```

Procédez à l'évaluation complète.""",
            
            few_shot_examples=[
                {
                    'article': 'Article 5 - Ratio de solvabilité',
                    'content': 'Les établissements maintiennent un ratio de fonds propres de base d\'au moins 8%',
                    'assessment': {
                        'niveau_materialite': 'CRITICAL',
                        'score_total': 95,
                        'justification': 'Ratio prudentiel fondamental avec impact direct sur l\'autorisation d\'exercer'
                    }
                }
            ],
            
            validation_criteria={
                'scoring_accuracy': 0.85,
                'justification_quality': 0.80,
                'consistency': 0.90
            },
            
            expected_output_format="JSON with detailed materiality assessment"
        )
    
    def _create_structure_template(self) -> PromptTemplate:
        """Create enhanced document structure template."""
        return PromptTemplate(
            system_prompt="""Vous êtes un expert en analyse structurelle de documents juridiques réglementaires.

EXPERTISE: Identification des structures hiérarchiques, organisation documentaire, analyse de la cohérence juridique.

STRUCTURES RÉGLEMENTAIRES STANDARDS:
- Préambule (Vu, Considérant)
- Corps du texte (TITRE → CHAPITRE → SECTION → ARTICLE)
- Dispositions finales
- Annexes éventuelles

MÉTHODOLOGIE:
1. Identification du type de document
2. Analyse de la structure hiérarchique
3. Cartographie des sections
4. Validation de la cohérence
5. Extraction des métadonnées""",
            
            user_prompt="""DOCUMENT À ANALYSER STRUCTURELLEMENT:

{document_text}

ANALYSE STRUCTURELLE REQUISE:

1. IDENTIFICATION DU DOCUMENT
   - Type: Règlement/Instruction/Décision
   - Autorité: COBAC/CEMAC/Autre
   - Référence: Numéro et date
   - Objet: Domaine réglementaire

2. STRUCTURE HIÉRARCHIQUE
   - Préambule (Vu, Considérant)
   - Dispositions générales
   - TITRES (niveau 1)
   - CHAPITRES (niveau 2)
   - SECTIONS (niveau 3)
   - ARTICLES (niveau 4)
   - PARAGRAPHES (niveau 5)

3. ORGANISATION DOCUMENTAIRE
   - Logique d'organisation
   - Cohérence structurelle
   - Références internes
   - Annexes et tableaux

FORMAT DE SORTIE:
```json
{
  "document_metadata": {
    "type": "Règlement COBAC",
    "reference": "R-2021-05",
    "date": "2021-12-15",
    "titre": "Règlement relatif aux...",
    "autorite": "COBAC",
    "statut": "En vigueur"
  },
  "structure_hierarchique": {
    "preambule": {
      "vu": ["Texte 1", "Texte 2"],
      "considerant": ["Motif 1", "Motif 2"]
    },
    "corps_du_texte": {
      "titres": [
        {
          "numero": "TITRE I",
          "intitule": "DISPOSITIONS GÉNÉRALES",
          "chapitres": [
            {
              "numero": "CHAPITRE I",
              "intitule": "CHAMP D'APPLICATION",
              "sections": [
                {
                  "numero": "SECTION I",
                  "intitule": "ÉTABLISSEMENTS CONCERNÉS",
                  "articles": ["Article 1", "Article 2"]
                }
              ]
            }
          ]
        }
      ]
    },
    "dispositions_finales": {
      "entree_en_vigueur": "Article X",
      "abrogation": "Article Y",
      "publication": "Article Z"
    },
    "annexes": [
      {
        "numero": "Annexe I",
        "titre": "Modèles de déclaration",
        "contenu": "Description"
      }
    ]
  },
  "navigation": {
    "total_articles": 45,
    "references_croisees": [
      {"de": "Article 5", "vers": "Article 12"},
      {"de": "Article 8", "vers": "Annexe I"}
    ],
    "tableaux_figures": [
      {"type": "tableau", "titre": "Ratios prudentiels", "position": "Article 10"}
    ]
  },
  "qualite_structurelle": {
    "coherence": 0.95,
    "completude": 0.90,
    "logique": 0.85,
    "conformite": 0.92
  }
}
```

Procédez à l'analyse structurelle complète.""",
            
            few_shot_examples=[],
            
            validation_criteria={
                'structure_completeness': 0.90,
                'hierarchy_accuracy': 0.85,
                'metadata_quality': 0.80
            },
            
            expected_output_format="JSON with complete document structure"
        )
    
    def _create_context_template(self) -> PromptTemplate:
        """Create enhanced context analysis template."""
        return PromptTemplate(
            system_prompt="""Vous êtes un expert en réglementation bancaire CEMAC/COBAC avec une connaissance approfondie de l'évolution réglementaire.

EXPERTISE: Analyse contextuelle, impact sectoriel, implications stratégiques, évolution réglementaire.

DIMENSIONS D'ANALYSE:
- Environnement réglementaire
- Impact sectoriel
- Enjeux opérationnels
- Implications stratégiques
- Évolution historique""",
            
            user_prompt="""DOCUMENT À ANALYSER CONTEXTUELLEMENT:

{document_content}

CONTEXTE RÉGLEMENTAIRE DISPONIBLE:
{regulatory_context}

ANALYSE CONTEXTUELLE REQUISE:

1. ENVIRONNEMENT RÉGLEMENTAIRE
   - Position dans le corpus réglementaire
   - Textes de référence et fondements
   - Textes abrogés ou modifiés
   - Cohérence avec l'existant

2. IMPACT SECTORIEL
   - Établissements concernés
   - Domaines d'activité impactés
   - Calendrier d'application
   - Mesures transitoires

3. ENJEUX OPÉRATIONNELS
   - Changements de procédures
   - Nouveaux reporting requis
   - Adaptations systémiques
   - Formation nécessaire

4. IMPLICATIONS STRATÉGIQUES
   - Impact sur la stratégie
   - Conséquences financières
   - Avantages concurrentiels
   - Risques réglementaires

FORMAT DE SORTIE:
```json
{
  "contexte_reglementaire": {
    "position_corpus": "Description de la position dans le corpus",
    "textes_reference": ["Texte 1", "Texte 2"],
    "textes_abroges": ["Ancien texte 1"],
    "coherence_existant": 0.95
  },
  "impact_sectoriel": {
    "etablissements_concernes": ["Banques", "EMF"],
    "domaines_activite": ["Crédit", "Dépôts"],
    "calendrier_application": "2024-01-01",
    "mesures_transitoires": "Description"
  },
  "enjeux_operationnels": {
    "changements_procedures": ["Procédure 1", "Procédure 2"],
    "nouveaux_reporting": ["État 1", "État 2"],
    "adaptations_systemes": ["Système 1", "Système 2"],
    "formation_requise": "Description"
  },
  "implications_strategiques": {
    "impact_strategie": "Description",
    "consequences_financieres": "Estimation",
    "avantages_concurrentiels": "Description",
    "risques_reglementaires": "Évaluation"
  },
  "recommandations": {
    "actions_immediates": ["Action 1", "Action 2"],
    "planification_moyen_terme": ["Plan 1", "Plan 2"],
    "surveillance_continue": ["Élément 1", "Élément 2"]
  }
}
```

Procédez à l'analyse contextuelle complète.""",
            
            few_shot_examples=[],
            
            validation_criteria={
                'context_relevance': 0.85,
                'impact_accuracy': 0.80,
                'strategic_insight': 0.75
            },
            
            expected_output_format="JSON with comprehensive context analysis"
        )
    
    def _create_validation_template(self) -> PromptTemplate:
        """Create enhanced validation template."""
        return PromptTemplate(
            system_prompt="""Vous êtes un expert en validation et contrôle qualité de documents réglementaires.

EXPERTISE: Validation de l'exactitude, vérification de la cohérence, contrôle qualité, détection d'erreurs.

CRITÈRES DE VALIDATION:
- Complétude (exhaustivité)
- Exactitude (fidélité au texte)
- Cohérence (logique interne)
- Qualité (présentation)""",
            
            user_prompt="""DONNÉES À VALIDER:

{extracted_data}

DOCUMENT ORIGINAL:
{original_document}

VALIDATION REQUISE:

1. COMPLÉTUDE (30%)
   - Tous les articles extraits?
   - Numérotation continue?
   - Références cohérentes?
   - Sections complètes?

2. EXACTITUDE (25%)
   - Fidélité au texte original?
   - Termes techniques corrects?
   - Dates et chiffres exacts?
   - Références précises?

3. COHÉRENCE (25%)
   - Logique structurelle?
   - Enchaînement des articles?
   - Définitions utilisées?
   - Contexte préservé?

4. QUALITÉ (20%)
   - Présentation claire?
   - Format standardisé?
   - Informations complètes?
   - Lisibilité optimale?

FORMAT DE VALIDATION:
```json
{
  "validation_results": {
    "completude": {
      "score": 0-100,
      "details": "Analyse détaillée",
      "elements_manquants": ["Élément 1"],
      "recommendations": ["Action 1"]
    },
    "exactitude": {
      "score": 0-100,
      "details": "Analyse détaillée",
      "erreurs_detectees": ["Erreur 1"],
      "corrections_requises": ["Correction 1"]
    },
    "coherence": {
      "score": 0-100,
      "details": "Analyse détaillée",
      "incoherences": ["Incohérence 1"],
      "ameliorations": ["Amélioration 1"]
    },
    "qualite": {
      "score": 0-100,
      "details": "Analyse détaillée",
      "defauts_presentation": ["Défaut 1"],
      "optimisations": ["Optimisation 1"]
    }
  },
  "score_global": 0-100,
  "niveau_qualite": "EXCELLENT/TRES_BIEN/CORRECT/INSUFFISANT",
  "certification": "VALIDÉ/RÉVISION_REQUISE/RETRAITEMENT_NÉCESSAIRE",
  "actions_correctives": ["Action 1", "Action 2"],
  "prochaines_etapes": ["Étape 1", "Étape 2"]
}
```

Procédez à la validation complète.""",
            
            few_shot_examples=[],
            
            validation_criteria={
                'validation_accuracy': 0.90,
                'error_detection': 0.85,
                'quality_assessment': 0.80
            },
            
            expected_output_format="JSON with detailed validation results"
        )
    
    def _create_chain_of_thought_template(self) -> PromptTemplate:
        """Create chain-of-thought template for complex reasoning."""
        return PromptTemplate(
            system_prompt="""Vous êtes un expert en analyse réglementaire avec une approche méthodique et structurée.

MÉTHODOLOGIE: Raisonnement étape par étape, validation continue, auto-correction.

PROCESSUS:
1. Observation initiale
2. Analyse étape par étape
3. Validation intermédiaire
4. Synthèse finale
5. Auto-évaluation""",
            
            user_prompt="""DOCUMENT À ANALYSER MÉTHODIQUEMENT:

{document}

RAISONNEMENT ÉTAPE PAR ÉTAPE:

ÉTAPE 1: OBSERVATION INITIALE
- Première impression du document
- Type et structure générale
- Complexité apparente
- Défis potentiels

ÉTAPE 2: ANALYSE STRUCTURELLE
- Identification des composants
- Hiérarchie documentaire
- Organisation logique
- Cohérence interne

ÉTAPE 3: EXTRACTION SYSTÉMATIQUE
- Méthode d'extraction
- Validation en cours
- Ajustements nécessaires
- Vérifications croisées

ÉTAPE 4: VALIDATION INTERMÉDIAIRE
- Contrôle de cohérence
- Vérification d'exhaustivité
- Détection d'erreurs
- Corrections appliquées

ÉTAPE 5: SYNTHÈSE FINALE
- Résultats consolidés
- Qualité globale
- Recommandations
- Prochaines étapes

ÉTAPE 6: AUTO-ÉVALUATION
- Niveau de confiance
- Zones d'incertitude
- Validation supplémentaire
- Amélioration continue

Procédez à l'analyse méthodique étape par étape.""",
            
            few_shot_examples=[],
            
            validation_criteria={
                'reasoning_quality': 0.85,
                'step_coherence': 0.80,
                'self_validation': 0.75
            },
            
            expected_output_format="Step-by-step reasoning with validation"
        )
    
    def get_prompt(self, prompt_type: PromptType, **kwargs) -> Tuple[str, str]:
        """Get formatted prompt for specific task."""
        template = self.templates.get(prompt_type)
        if not template:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
        
        try:
            # Format the user prompt with provided arguments
            formatted_user_prompt = template.user_prompt.format(**kwargs)
            
            return template.system_prompt, formatted_user_prompt
            
        except KeyError as e:
            raise ValueError(f"Missing required parameter for {prompt_type}: {e}")
    
    def get_few_shot_examples(self, prompt_type: PromptType) -> List[Dict[str, Any]]:
        """Get few-shot examples for specific prompt type."""
        template = self.templates.get(prompt_type)
        return template.few_shot_examples if template else []
    
    def validate_output(self, prompt_type: PromptType, output: str) -> Dict[str, Any]:
        """Validate AI output against expected criteria."""
        template = self.templates.get(prompt_type)
        if not template:
            return {'is_valid': False, 'error': 'Unknown prompt type'}
        
        try:
            # Try to parse JSON output
            if template.expected_output_format.lower().startswith('json'):
                parsed_output = json.loads(output)
                return {
                    'is_valid': True,
                    'parsed_output': parsed_output,
                    'validation_score': self._calculate_validation_score(
                        prompt_type, parsed_output
                    )
                }
            else:
                return {
                    'is_valid': True,
                    'output': output,
                    'validation_score': 0.8  # Default score for non-JSON
                }
                
        except json.JSONDecodeError as e:
            return {
                'is_valid': False,
                'error': f'Invalid JSON format: {e}',
                'raw_output': output
            }
    
    def _calculate_validation_score(self, prompt_type: PromptType, 
                                   output: Dict[str, Any]) -> float:
        """Calculate validation score based on output quality."""
        template = self.templates.get(prompt_type)
        if not template:
            return 0.0
        
        criteria = template.validation_criteria
        score = 0.0
        total_weight = 0.0
        
        # Generic validation checks
        if prompt_type == PromptType.ARTICLE_EXTRACTION:
            if 'articles' in output and len(output['articles']) > 0:
                score += 0.3
            if 'document_info' in output:
                score += 0.2
            if 'structure_hierarchique' in output:
                score += 0.2
            if 'statistiques' in output:
                score += 0.1
        
        elif prompt_type == PromptType.MATERIALITY_ASSESSMENT:
            if 'evaluation' in output:
                score += 0.3
            if 'niveau_materialite' in output:
                score += 0.2
            if 'justification' in output:
                score += 0.2
            if 'actions_requises' in output:
                score += 0.1
        
        # Add more specific validation logic as needed
        
        return min(score, 1.0)
    
    def get_adaptive_prompt(self, prompt_type: PromptType, 
                           previous_attempts: List[Dict[str, Any]], 
                           **kwargs) -> Tuple[str, str]:
        """Get adaptive prompt based on previous attempts."""
        base_system, base_user = self.get_prompt(prompt_type, **kwargs)
        
        if not previous_attempts:
            return base_system, base_user
        
        # Analyze previous failures
        failure_patterns = self._analyze_failure_patterns(previous_attempts)
        
        # Enhance prompt based on failure patterns
        enhanced_system = self._enhance_system_prompt(base_system, failure_patterns)
        enhanced_user = self._enhance_user_prompt(base_user, failure_patterns)
        
        return enhanced_system, enhanced_user
    
    def _analyze_failure_patterns(self, attempts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze failure patterns from previous attempts."""
        patterns = {
            'common_errors': [],
            'missing_elements': [],
            'format_issues': [],
            'content_problems': []
        }
        
        for attempt in attempts:
            if not attempt.get('success', False):
                error = attempt.get('error', '')
                if 'json' in error.lower():
                    patterns['format_issues'].append('JSON formatting')
                elif 'missing' in error.lower():
                    patterns['missing_elements'].append('Missing required elements')
                elif 'article' in error.lower():
                    patterns['content_problems'].append('Article extraction issues')
        
        return patterns
    
    def _enhance_system_prompt(self, base_prompt: str, 
                              failure_patterns: Dict[str, Any]) -> str:
        """Enhance system prompt based on failure patterns."""
        enhancements = []
        
        if failure_patterns['format_issues']:
            enhancements.append("""
ATTENTION SPÉCIALE AU FORMAT:
- Respectez STRICTEMENT le format JSON demandé
- Vérifiez la syntaxe JSON avant de répondre
- Utilisez des guillemets doubles pour les chaînes
- Évitez les caractères spéciaux non échappés""")
        
        if failure_patterns['missing_elements']:
            enhancements.append("""
EXHAUSTIVITÉ REQUISE:
- N'omettez AUCUN élément demandé
- Vérifiez que tous les champs requis sont présents
- Complétez tous les attributs même si informations partielles""")
        
        if failure_patterns['content_problems']:
            enhancements.append("""
PRÉCISION DU CONTENU:
- Lisez attentivement TOUT le document
- Extrayez TOUS les articles sans exception
- Préservez le texte original exactement
- Validez la numérotation séquentielle""")
        
        if enhancements:
            return base_prompt + "\n\n" + "\n".join(enhancements)
        
        return base_prompt
    
    def _enhance_user_prompt(self, base_prompt: str, 
                            failure_patterns: Dict[str, Any]) -> str:
        """Enhance user prompt based on failure patterns."""
        if failure_patterns['format_issues']:
            base_prompt += "\n\nIMPORTANT: Respectez EXACTEMENT le format JSON demandé."
        
        if failure_patterns['missing_elements']:
            base_prompt += "\n\nATTENTION: Assurez-vous que TOUS les éléments requis sont présents."
        
        return base_prompt