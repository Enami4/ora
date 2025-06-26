#!/usr/bin/env python3
"""
PyQt GUI for Regulatory Document Processor - JABE Branded Version
User-friendly interface with JABE branding and design consistency.
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QGridLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QProgressBar,
    QGroupBox, QCheckBox, QComboBox, QFileDialog, QMessageBox, QTabWidget,
    QTableWidget, QTableWidgetItem, QHeaderView, QSplitter, QFrame,
    QScrollArea, QSpinBox, QDoubleSpinBox, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QSettings, QSize
from PyQt5.QtGui import QFont, QIcon, QPixmap, QPalette, QColor, QPainter, QBrush

from regulatory_processor import RegulatoryDocumentProcessor, ProcessorConfig


# JABE Brand Colors (extracted from logo)
class JABEColors:
    PRIMARY_BLUE = "#1A8FD1"      # Main blue from logo
    DARK_NAVY = "#2C3E50"         # Dark blue/navy from logo
    LIGHT_BLUE = "#74B9FF"        # Light blue accent
    LIGHTER_BLUE = "#A4CDFF"      # Even lighter blue for backgrounds
    WHITE = "#FFFFFF"
    LIGHT_GRAY = "#F8F9FA"
    MEDIUM_GRAY = "#E9ECEF"
    DARK_GRAY = "#495057"
    SUCCESS_GREEN = "#28A745"
    WARNING_ORANGE = "#FFC107"
    ERROR_RED = "#DC3545"


class ProcessingThread(QThread):
    """Background thread for document processing to keep UI responsive."""
    
    progress_update = pyqtSignal(int, str)  # (percentage, status_message)
    document_processed = pyqtSignal(str, dict)  # (filename, result_summary)
    processing_complete = pyqtSignal(str, dict)  # (output_file, final_summary)
    error_occurred = pyqtSignal(str)  # (error_message)
    log_message = pyqtSignal(str)  # (log_message)
    
    def __init__(self, config: ProcessorConfig, document_paths: List[str], 
                 output_path: str, user_info: Dict[str, str]):
        super().__init__()
        self.config = config
        self.document_paths = document_paths
        self.output_path = output_path
        self.user_info = user_info
        self.is_cancelled = False
    
    def run(self):
        """Main processing logic running in background thread."""
        try:
            self.log_message.emit(f"Starting processing with user: {self.user_info['name']} {self.user_info['surname']}")
            
            # Initialize processor
            processor = RegulatoryDocumentProcessor(self.config)
            
            total_files = len(self.document_paths)
            processed_count = 0
            
            self.progress_update.emit(0, f"Initializing processor... Found {total_files} documents")
            
            for i, doc_path in enumerate(self.document_paths):
                if self.is_cancelled:
                    self.log_message.emit("Processing cancelled by user")
                    return
                
                filename = os.path.basename(doc_path)
                self.progress_update.emit(
                    int((i / total_files) * 100), 
                    f"Processing {filename}..."
                )
                self.log_message.emit(f"Processing: {filename}")
                
                try:
                    if os.path.isfile(doc_path):
                        result = processor.process_document(doc_path)
                        if result:
                            processed_count += 1
                            summary = self._create_summary(result, filename)
                            self.document_processed.emit(filename, summary)
                            self.log_message.emit(f"✓ Successfully processed: {filename}")
                        else:
                            self.log_message.emit(f"✗ Failed to process: {filename}")
                    elif os.path.isdir(doc_path):
                        # Process directory - this automatically adds documents to processor.processed_documents
                        results = processor.process_directory(doc_path, recursive=True)
                        processed_count += len(results)
                        
                        # Emit signals for each processed file in the directory
                        for result in results:
                            file_name = result['metadata'].get('file_name', 'unknown')
                            summary = self._create_summary(result, file_name)
                            self.document_processed.emit(file_name, summary)
                        
                        self.log_message.emit(f"✓ Successfully processed directory {filename}: {len(results)} files")
                    else:
                        self.log_message.emit(f"Skipping invalid path: {doc_path}")
                        continue
                
                except Exception as e:
                    self.log_message.emit(f"✗ Error processing {filename}: {str(e)}")
                    continue
            
            # Export results
            self.progress_update.emit(90, "Generating Excel export...")
            self.log_message.emit("Exporting results to Excel...")
            self.log_message.emit(f"Processor has {len(processor.processed_documents)} documents ready for export")
            self.log_message.emit(f"Export path: {self.output_path}")
            
            processor.export_results(
                self.output_path,
                format='excel',
                include_validation=self.config.enable_ai_validation,
                include_articles=self.config.extract_articles,
                include_full_text=True,
                user_info=self.user_info
            )
            
            # Create final summary
            final_summary = processor.get_summary()
            final_summary.update({
                'output_file': self.output_path,
                'processed_count': processed_count,
                'total_files': total_files,
                'user_info': self.user_info,
                'processing_time': datetime.now().isoformat()
            })
            
            self.progress_update.emit(100, "Processing complete!")
            self.log_message.emit(f"✅ Processing complete! Output saved to: {self.output_path}")
            self.processing_complete.emit(self.output_path, final_summary)
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(f"❌ Critical error: {str(e)}")
            self.error_occurred.emit(error_msg)
    
    def _create_summary(self, result: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Create summary for a processed document."""
        return {
            'type': 'single',
            'chunks': result.get('statistics', {}).get('total_chunks', 0),
            'words': result.get('statistics', {}).get('total_words', 0),
            'articles': len(result.get('articles', [])),
            'validation_score': result.get('validation_results', {}).get('document_validation', {}).get('overall_score', 0)
        }
    
    def cancel(self):
        """Cancel the processing."""
        self.is_cancelled = True


class JABERegulatoryGUI(QMainWindow):
    """Main GUI application with JABE branding and design."""
    
    def __init__(self):
        super().__init__()
        self.settings = QSettings('JABE', 'RegulatoryProcessor')
        self.processing_thread = None
        self.processed_documents = []
        
        self.init_ui()
        self.load_settings()
    
    def init_ui(self):
        """Initialize the user interface with JABE branding."""
        self.setWindowTitle("JABE Regulatory Document Processor")
        self.setMinimumSize(1400, 800)
        self.setStyleSheet(self._get_jabe_stylesheet())
        
        # Set window icon
        self.set_window_icon()
        
        # Central widget with tabs
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header with logo
        header = self._create_jabe_header()
        layout.addWidget(header)
        
        # Main content area
        content_widget = QWidget()
        content_widget.setStyleSheet(f"background-color: {JABEColors.LIGHT_GRAY};")
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(20, 20, 20, 20)
        
        # Main tabs
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet(self._get_tab_stylesheet())
        self.tab_widget.addTab(self._create_input_tab(), "📋 Saisie Documents")
        self.tab_widget.addTab(self._create_config_tab(), "⚙️ Configuration")
        self.tab_widget.addTab(self._create_processing_tab(), "🔄 Traitement")
        self.tab_widget.addTab(self._create_results_tab(), "📊 Résultats")
        
        content_layout.addWidget(self.tab_widget)
        layout.addWidget(content_widget)
        
        # Status bar with JABE styling
        self.statusBar().setStyleSheet(f"""
            QStatusBar {{
                background-color: {JABEColors.DARK_NAVY};
                color: white;
                border: none;
                font-weight: 500;
            }}
        """)
        self.statusBar().showMessage("Prêt à traiter les documents réglementaires")
    
    def set_window_icon(self):
        """Set window icon for both window and taskbar display."""
        logo_path = os.path.join(os.path.dirname(__file__), "static", "img", "JABE_LOGO_02.png")
        if os.path.exists(logo_path):
            icon = QIcon(logo_path)
            # Set window icon
            self.setWindowIcon(icon)
            # Set application icon for taskbar
            app = QApplication.instance()
            if app:
                app.setWindowIcon(icon)
            # For Windows taskbar grouping
            try:
                import ctypes
                myappid = 'jabe.regulatory.processor.1.0'
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
            except:
                pass  # Not on Windows or ctypes not available
        else:
            # Fallback icon
            self.setWindowIcon(self.style().standardIcon(self.style().SP_ComputerIcon))
    
    def _create_jabe_header(self) -> QWidget:
        """Create the JABE branded header."""
        header_frame = QFrame()
        header_frame.setFixedHeight(140)
        header_frame.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {JABEColors.PRIMARY_BLUE}, stop:1 {JABEColors.DARK_NAVY});
                border: none;
            }}
        """)
        
        layout = QHBoxLayout(header_frame)
        layout.setContentsMargins(40, 20, 40, 20)
        
        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "static", "img", "JABE_LOGO_02.png")
        if os.path.exists(logo_path):
            pixmap = QPixmap(logo_path)
            # Scale logo proportionally to fit header with better sizing
            scaled_pixmap = pixmap.scaled(150, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            logo_label.setPixmap(scaled_pixmap)
        else:
            # Fallback if logo not found
            logo_label.setText("JABE")
            logo_label.setStyleSheet(f"""
                font-size: 32px; 
                font-weight: bold; 
                color: white;
                background: transparent;
            """)
        
        layout.addWidget(logo_label)
        
        # Spacer
        layout.addStretch()
        
        # Title section with flexible layout
        title_widget = QWidget()
        title_widget.setStyleSheet("background: transparent;")
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(30)
        
        # Main title - takes full width needed
        title = QLabel("Processeur de Documents Réglementaires")
        title.setFont(QFont("Segoe UI", 24, QFont.Bold))
        title.setStyleSheet("color: white; background: transparent; padding: 2px;")
        title.setWordWrap(False)  # Keep on one line if possible
        title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        title.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        title.setMinimumWidth(400)  # Ensure minimum width for readability
        title_layout.addWidget(title)
        
        # Subtitle - flexible width
        subtitle = QLabel("Validation IA & Extraction d'Articles - Solution Professionnelle")
        subtitle.setFont(QFont("Segoe UI", 13))
        subtitle.setStyleSheet(f"color: {JABEColors.LIGHTER_BLUE}; background: transparent; padding: 2px;")
        subtitle.setWordWrap(True)  # Allow wrapping for subtitle
        subtitle.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        subtitle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        subtitle.setMinimumWidth(300)  # Minimum width to prevent over-compression
        title_layout.addWidget(subtitle)
        
        layout.addWidget(title_widget)
        
        return header_frame
    
    def _create_input_tab(self) -> QWidget:
        """Create the document input tab with JABE styling."""
        widget = QWidget()
        widget.setStyleSheet(f"background-color: {JABEColors.LIGHT_GRAY};")
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # User Information Group
        user_group = self._create_styled_group("👤 Informations Utilisateur")
        user_layout = QGridLayout()
        user_layout.setSpacing(15)
        
        # First name
        user_layout.addWidget(self._create_label("Prénom:"), 0, 0)
        self.name_input = self._create_input("Entrez votre prénom")
        user_layout.addWidget(self.name_input, 0, 1)
        
        # Last name
        user_layout.addWidget(self._create_label("Nom:"), 1, 0)
        self.surname_input = self._create_input("Entrez votre nom de famille")
        user_layout.addWidget(self.surname_input, 1, 1)
        
        user_group.setLayout(user_layout)
        layout.addWidget(user_group)
        
        # Document Paths Group
        paths_group = self._create_styled_group("📁 Localisation des Documents")
        paths_layout = QVBoxLayout()
        paths_layout.setSpacing(15)
        
        # Document paths list
        paths_layout.addWidget(self._create_label("Chemins sélectionnés:"))
        self.paths_list = QTextEdit()
        self.paths_list.setMaximumHeight(120)
        self.paths_list.setPlaceholderText("Les chemins des documents apparaîtront ici...")
        self.paths_list.setStyleSheet(self._get_input_stylesheet())
        paths_layout.addWidget(self.paths_list)
        
        # Path selection buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(10)
        
        self.add_files_btn = self._create_primary_button("📄 Ajouter Fichiers PDF")
        self.add_files_btn.clicked.connect(self.add_pdf_files)
        buttons_layout.addWidget(self.add_files_btn)
        
        self.add_folder_btn = self._create_primary_button("📂 Ajouter Dossier")
        self.add_folder_btn.clicked.connect(self.add_folder)
        buttons_layout.addWidget(self.add_folder_btn)
        
        self.clear_paths_btn = self._create_secondary_button("🗑️ Tout Effacer")
        self.clear_paths_btn.clicked.connect(self.clear_paths)
        buttons_layout.addWidget(self.clear_paths_btn)
        
        paths_layout.addLayout(buttons_layout)
        paths_group.setLayout(paths_layout)
        layout.addWidget(paths_group)
        
        # Output Settings Group
        output_group = self._create_styled_group("💾 Paramètres de Sortie")
        output_layout = QGridLayout()
        output_layout.setSpacing(15)
        
        # Output directory
        output_layout.addWidget(self._create_label("Répertoire de sortie:"), 0, 0)
        self.output_dir_input = self._create_input("Sélectionnez le répertoire de sortie...")
        output_layout.addWidget(self.output_dir_input, 0, 1)
        
        self.browse_output_btn = self._create_secondary_button("📂 Parcourir")
        self.browse_output_btn.clicked.connect(self.browse_output_directory)
        output_layout.addWidget(self.browse_output_btn, 0, 2)
        
        # Filename
        output_layout.addWidget(self._create_label("Nom du fichier:"), 1, 0)
        self.output_filename_input = self._create_input("Nom de base (info utilisateur et horodatage ajoutés automatiquement)")
        self.output_filename_input.setText("analyse_reglementaire")
        output_layout.addWidget(self.output_filename_input, 1, 1, 1, 2)
        
        # Preview
        output_layout.addWidget(self._create_label("Aperçu:"), 2, 0)
        self.filename_preview = QLabel()
        self.filename_preview.setStyleSheet(f"""
            color: {JABEColors.DARK_GRAY}; 
            font-style: italic; 
            font-size: 11px;
            background: transparent;
            padding: 5px;
        """)
        self.update_filename_preview()
        output_layout.addWidget(self.filename_preview, 2, 1, 1, 2)
        
        # Connect signals
        self.name_input.textChanged.connect(self.update_filename_preview)
        self.surname_input.textChanged.connect(self.update_filename_preview)
        self.output_filename_input.textChanged.connect(self.update_filename_preview)
        
        output_group.setLayout(output_layout)
        layout.addWidget(output_group)
        
        # Action Buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()
        
        self.validate_btn = self._create_success_button("✅ Valider les Entrées")
        self.validate_btn.clicked.connect(self.validate_inputs)
        action_layout.addWidget(self.validate_btn)
        
        self.start_processing_btn = self._create_primary_button("🚀 Démarrer le Traitement")
        self.start_processing_btn.clicked.connect(self.start_processing)
        self.start_processing_btn.setEnabled(False)
        action_layout.addWidget(self.start_processing_btn)
        
        layout.addLayout(action_layout)
        layout.addStretch()
        
        return widget
    
    def _create_config_tab(self) -> QWidget:
        """Create the configuration tab with JABE styling."""
        widget = QWidget()
        widget.setStyleSheet(f"background-color: {JABEColors.LIGHT_GRAY};")
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # Processing Settings
        processing_group = self._create_styled_group("⚙️ Configuration du Traitement")
        processing_layout = QGridLayout()
        processing_layout.setSpacing(15)
        
        processing_layout.addWidget(self._create_label("Taille des segments:"), 0, 0)
        self.chunk_size_input = self._create_spinbox(200, 2000, 1000, " caractères")
        processing_layout.addWidget(self.chunk_size_input, 0, 1)
        
        processing_layout.addWidget(self._create_label("Chevauchement:"), 1, 0)
        self.chunk_overlap_input = self._create_spinbox(50, 500, 200, " caractères")
        processing_layout.addWidget(self.chunk_overlap_input, 1, 1)
        
        processing_layout.addWidget(self._create_label("Taille max fichier:"), 2, 0)
        self.max_file_size_input = self._create_spinbox(10, 500, 100, " MB")
        processing_layout.addWidget(self.max_file_size_input, 2, 1)
        
        processing_group.setLayout(processing_layout)
        layout.addWidget(processing_group)
        
        # AI Settings
        ai_group = self._create_styled_group("🤖 Paramètres de Validation IA")
        ai_layout = QGridLayout()
        ai_layout.setSpacing(15)
        
        self.enable_ai_checkbox = self._create_checkbox("Activer la Validation IA")
        self.enable_ai_checkbox.setChecked(True)
        self.enable_ai_checkbox.toggled.connect(self.toggle_ai_settings)
        ai_layout.addWidget(self.enable_ai_checkbox, 0, 0, 1, 2)
        
        ai_layout.addWidget(self._create_label("Clé API:"), 1, 0)
        self.api_key_input = self._create_input("Clé API Anthropic ou variable d'environnement ANTHROPIC_API_KEY")
        self.api_key_input.setEchoMode(QLineEdit.Password)
        ai_layout.addWidget(self.api_key_input, 1, 1)
        
        ai_layout.addWidget(self._create_label("Modèle IA:"), 2, 0)
        self.ai_model_combo = self._create_combobox([
            "claude-3-haiku-20240307",
            "claude-3-sonnet-20240229", 
            "claude-3-opus-20240229"
        ])
        ai_layout.addWidget(self.ai_model_combo, 2, 1)
        
        ai_group.setLayout(ai_layout)
        layout.addWidget(ai_group)
        
        # Feature Settings
        features_group = self._create_styled_group("🎯 Paramètres des Fonctionnalités")
        features_layout = QVBoxLayout()
        features_layout.setSpacing(10)
        
        self.extract_articles_checkbox = self._create_checkbox("Extraire les Articles Réglementaires")
        self.extract_articles_checkbox.setChecked(True)
        features_layout.addWidget(self.extract_articles_checkbox)
        
        self.assess_materiality_checkbox = self._create_checkbox("Évaluer la Matérialité des Articles")
        self.assess_materiality_checkbox.setChecked(True)
        features_layout.addWidget(self.assess_materiality_checkbox)
        
        self.clean_text_checkbox = self._create_checkbox("Nettoyer le Texte Extrait")
        self.clean_text_checkbox.setChecked(True)
        features_layout.addWidget(self.clean_text_checkbox)
        
        features_group.setLayout(features_layout)
        layout.addWidget(features_group)
        
        # Preset buttons
        preset_layout = QHBoxLayout()
        
        self.basic_preset_btn = self._create_secondary_button("Traitement Basique")
        self.basic_preset_btn.clicked.connect(self.load_basic_preset)
        preset_layout.addWidget(self.basic_preset_btn)
        
        self.ai_preset_btn = self._create_primary_button("IA Avancée")
        self.ai_preset_btn.clicked.connect(self.load_ai_preset)
        preset_layout.addWidget(self.ai_preset_btn)
        
        self.custom_preset_btn = self._create_secondary_button("Sauvegarder Preset")
        self.custom_preset_btn.clicked.connect(self.save_preset)
        preset_layout.addWidget(self.custom_preset_btn)
        
        layout.addLayout(preset_layout)
        layout.addStretch()
        
        return widget
    
    def _create_processing_tab(self) -> QWidget:
        """Create the processing tab with JABE styling."""
        widget = QWidget()
        widget.setStyleSheet(f"background-color: {JABEColors.LIGHT_GRAY};")
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # Progress Section
        progress_group = self._create_styled_group("📊 Progression du Traitement")
        progress_layout = QVBoxLayout()
        progress_layout.setSpacing(15)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 2px solid {JABEColors.MEDIUM_GRAY};
                border-radius: 8px;
                text-align: center;
                font-weight: bold;
                font-size: 12px;
                background-color: white;
                height: 25px;
            }}
            QProgressBar::chunk {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 {JABEColors.PRIMARY_BLUE}, stop:1 {JABEColors.LIGHT_BLUE});
                border-radius: 6px;
            }}
        """)
        progress_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Prêt à démarrer le traitement...")
        self.status_label.setStyleSheet(f"""
            color: {JABEColors.DARK_GRAY};
            font-size: 14px;
            font-weight: 500;
            background: transparent;
            padding: 5px;
        """)
        progress_layout.addWidget(self.status_label)
        
        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)
        
        # Control Buttons
        control_layout = QHBoxLayout()
        
        self.pause_btn = self._create_secondary_button("⏸️ Pause")
        self.pause_btn.setEnabled(False)
        control_layout.addWidget(self.pause_btn)
        
        self.cancel_btn = self._create_warning_button("⏹️ Annuler")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        control_layout.addWidget(self.cancel_btn)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Processing Log
        log_group = self._create_styled_group("📝 Journal de Traitement")
        log_layout = QVBoxLayout()
        log_layout.setSpacing(10)
        
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setFont(QFont("Consolas", 9))
        self.log_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {JABEColors.DARK_NAVY};
                color: {JABEColors.LIGHTER_BLUE};
                border: 2px solid {JABEColors.MEDIUM_GRAY};
                border-radius: 8px;
                padding: 10px;
                font-family: 'Consolas', 'Monaco', monospace;
            }}
        """)
        log_layout.addWidget(self.log_display)
        
        log_controls = QHBoxLayout()
        
        self.clear_log_btn = self._create_secondary_button("🗑️ Effacer Journal")
        self.clear_log_btn.clicked.connect(self.clear_log)
        log_controls.addWidget(self.clear_log_btn)
        
        self.save_log_btn = self._create_secondary_button("💾 Sauvegarder Journal")
        self.save_log_btn.clicked.connect(self.save_log)
        log_controls.addWidget(self.save_log_btn)
        
        log_controls.addStretch()
        log_layout.addLayout(log_controls)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        return widget
    
    def _create_results_tab(self) -> QWidget:
        """Create the results tab with JABE styling."""
        widget = QWidget()
        widget.setStyleSheet(f"background-color: {JABEColors.LIGHT_GRAY};")
        layout = QVBoxLayout(widget)
        layout.setSpacing(20)
        
        # Results Summary
        summary_group = self._create_styled_group("📈 Résumé du Traitement")
        summary_layout = QGridLayout()
        summary_layout.setSpacing(15)
        
        self.summary_labels = {}
        labels = [
            ("Documents Total:", "total_docs"),
            ("Réussis:", "successful"),
            ("Échoués:", "failed"), 
            ("Articles Total:", "articles"),
            ("Score Moyen:", "avg_score")
        ]
        
        for i, (label, key) in enumerate(labels):
            summary_layout.addWidget(self._create_label(label), i // 3, (i % 3) * 2)
            value_label = QLabel("0")
            value_label.setStyleSheet(f"""
                font-weight: bold; 
                font-size: 16px;
                color: {JABEColors.PRIMARY_BLUE};
                background: transparent;
            """)
            self.summary_labels[key] = value_label
            summary_layout.addWidget(value_label, i // 3, (i % 3) * 2 + 1)
        
        summary_group.setLayout(summary_layout)
        layout.addWidget(summary_group)
        
        # Document Results Table
        table_group = self._create_styled_group("📋 Détails des Documents")
        table_layout = QVBoxLayout()
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Document", "Type", "Articles", "Segments", "Score Validation", "Statut"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.setStyleSheet(f"""
            QTableWidget {{
                background-color: white;
                border: 2px solid {JABEColors.MEDIUM_GRAY};
                border-radius: 8px;
                gridline-color: {JABEColors.MEDIUM_GRAY};
            }}
            QHeaderView::section {{
                background-color: {JABEColors.PRIMARY_BLUE};
                color: white;
                font-weight: bold;
                border: none;
                padding: 8px;
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {JABEColors.MEDIUM_GRAY};
            }}
        """)
        table_layout.addWidget(self.results_table)
        
        table_group.setLayout(table_layout)
        layout.addWidget(table_group)
        
        # Download Section
        download_group = self._create_styled_group("⬇️ Télécharger les Résultats")
        download_layout = QHBoxLayout()
        download_layout.setSpacing(15)
        
        self.output_path_label = QLabel("Aucun fichier de sortie généré")
        self.output_path_label.setStyleSheet(f"""
            color: {JABEColors.DARK_GRAY};
            font-size: 12px;
            background: transparent;
        """)
        download_layout.addWidget(self.output_path_label)
        
        download_layout.addStretch()
        
        self.open_file_btn = self._create_success_button("📂 Ouvrir Fichier")
        self.open_file_btn.clicked.connect(self.open_output_file)
        self.open_file_btn.setEnabled(False)
        download_layout.addWidget(self.open_file_btn)
        
        self.open_folder_btn = self._create_primary_button("📁 Ouvrir Dossier")
        self.open_folder_btn.clicked.connect(self.open_output_folder)
        self.open_folder_btn.setEnabled(False)
        download_layout.addWidget(self.open_folder_btn)
        
        download_group.setLayout(download_layout)
        layout.addWidget(download_group)
        
        return widget
    
    def _get_jabe_stylesheet(self) -> str:
        """Get the JABE-branded application stylesheet."""
        return f"""
            QMainWindow {{
                background-color: {JABEColors.LIGHT_GRAY};
            }}
            QWidget {{
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }}
        """
    
    def _get_tab_stylesheet(self) -> str:
        """Get the tab widget stylesheet."""
        return f"""
            QTabWidget::pane {{
                border: 2px solid {JABEColors.MEDIUM_GRAY};
                background-color: white;
                border-radius: 8px;
                top: -2px;
            }}
            QTabBar::tab {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {JABEColors.MEDIUM_GRAY}, stop:1 {JABEColors.LIGHT_GRAY});
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                font-weight: 500;
                font-size: 13px;
                color: {JABEColors.DARK_GRAY};
            }}
            QTabBar::tab:selected {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {JABEColors.PRIMARY_BLUE}, stop:1 {JABEColors.LIGHT_BLUE});
                color: white;
                font-weight: bold;
            }}
            QTabBar::tab:hover:!selected {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {JABEColors.LIGHTER_BLUE}, stop:1 {JABEColors.LIGHT_GRAY});
            }}
        """
    
    def _get_input_stylesheet(self) -> str:
        """Get the input field stylesheet."""
        return f"""
            QLineEdit, QTextEdit, QSpinBox, QComboBox {{
                border: 2px solid {JABEColors.MEDIUM_GRAY};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
                background-color: white;
                color: {JABEColors.DARK_GRAY};
            }}
            QLineEdit:focus, QTextEdit:focus, QSpinBox:focus, QComboBox:focus {{
                border-color: {JABEColors.PRIMARY_BLUE};
                outline: none;
            }}
        """
    
    def _create_styled_group(self, title: str) -> QGroupBox:
        """Create a styled group box."""
        group = QGroupBox(title)
        group.setStyleSheet(f"""
            QGroupBox {{
                font-weight: bold;
                font-size: 14px;
                border: 2px solid {JABEColors.MEDIUM_GRAY};
                border-radius: 10px;
                margin-top: 12px;
                padding-top: 15px;
                background-color: white;
                color: {JABEColors.DARK_NAVY};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 10px 0 10px;
                color: {JABEColors.PRIMARY_BLUE};
                background-color: white;
            }}
        """)
        return group
    
    def _create_label(self, text: str) -> QLabel:
        """Create a styled label."""
        label = QLabel(text)
        label.setStyleSheet(f"""
            color: {JABEColors.DARK_GRAY};
            font-weight: 500;
            font-size: 13px;
            background: transparent;
        """)
        return label
    
    def _create_input(self, placeholder: str) -> QLineEdit:
        """Create a styled input field."""
        input_field = QLineEdit()
        input_field.setPlaceholderText(placeholder)
        input_field.setStyleSheet(self._get_input_stylesheet())
        return input_field
    
    def _create_spinbox(self, min_val: int, max_val: int, default: int, suffix: str) -> QSpinBox:
        """Create a styled spinbox."""
        spinbox = QSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(default)
        spinbox.setSuffix(suffix)
        spinbox.setStyleSheet(self._get_input_stylesheet())
        return spinbox
    
    def _create_combobox(self, items: List[str]) -> QComboBox:
        """Create a styled combobox."""
        combo = QComboBox()
        combo.addItems(items)
        combo.setStyleSheet(self._get_input_stylesheet())
        return combo
    
    def _create_checkbox(self, text: str) -> QCheckBox:
        """Create a styled checkbox."""
        checkbox = QCheckBox(text)
        checkbox.setStyleSheet(f"""
            QCheckBox {{
                font-size: 13px;
                color: {JABEColors.DARK_GRAY};
                background: transparent;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 18px;
                height: 18px;
                border-radius: 3px;
                border: 2px solid {JABEColors.MEDIUM_GRAY};
                background-color: white;
            }}
            QCheckBox::indicator:checked {{
                background-color: {JABEColors.PRIMARY_BLUE};
                border-color: {JABEColors.PRIMARY_BLUE};
                image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTIiIGhlaWdodD0iOSIgdmlld0JveD0iMCAwIDEyIDkiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik0xIDQuNUw0LjUgOEwxMSAxIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIgc3Ryb2tlLWxpbmVqb2luPSJyb3VuZCIvPgo8L3N2Zz4K);
            }}
        """)
        return checkbox
    
    def _create_primary_button(self, text: str) -> QPushButton:
        """Create a primary styled button."""
        button = QPushButton(text)
        button.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {JABEColors.PRIMARY_BLUE}, stop:1 {JABEColors.LIGHT_BLUE});
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {JABEColors.LIGHT_BLUE}, stop:1 {JABEColors.PRIMARY_BLUE});
            }}
            QPushButton:pressed {{
                background-color: {JABEColors.DARK_NAVY};
            }}
            QPushButton:disabled {{
                background-color: {JABEColors.MEDIUM_GRAY};
                color: {JABEColors.DARK_GRAY};
            }}
        """)
        return button
    
    def _create_secondary_button(self, text: str) -> QPushButton:
        """Create a secondary styled button."""
        button = QPushButton(text)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: white;
                color: {JABEColors.PRIMARY_BLUE};
                border: 2px solid {JABEColors.PRIMARY_BLUE};
                padding: 10px 18px;
                border-radius: 8px;
                font-weight: 500;
                font-size: 13px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: {JABEColors.LIGHTER_BLUE};
                color: white;
                border-color: {JABEColors.LIGHTER_BLUE};
            }}
            QPushButton:pressed {{
                background-color: {JABEColors.PRIMARY_BLUE};
                color: white;
            }}
            QPushButton:disabled {{
                background-color: {JABEColors.MEDIUM_GRAY};
                color: {JABEColors.DARK_GRAY};
                border-color: {JABEColors.MEDIUM_GRAY};
            }}
        """)
        return button
    
    def _create_success_button(self, text: str) -> QPushButton:
        """Create a success styled button."""
        button = QPushButton(text)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {JABEColors.SUCCESS_GREEN};
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #218838;
            }}
            QPushButton:pressed {{
                background-color: #1e7e34;
            }}
            QPushButton:disabled {{
                background-color: {JABEColors.MEDIUM_GRAY};
                color: {JABEColors.DARK_GRAY};
            }}
        """)
        return button
    
    def _create_warning_button(self, text: str) -> QPushButton:
        """Create a warning styled button."""
        button = QPushButton(text)
        button.setStyleSheet(f"""
            QPushButton {{
                background-color: {JABEColors.ERROR_RED};
                color: white;
                border: none;
                padding: 12px 20px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 13px;
                min-width: 120px;
            }}
            QPushButton:hover {{
                background-color: #c82333;
            }}
            QPushButton:pressed {{
                background-color: #bd2130;
            }}
            QPushButton:disabled {{
                background-color: {JABEColors.MEDIUM_GRAY};
                color: {JABEColors.DARK_GRAY};
            }}
        """)
        return button
    
    # Include all the existing methods from the original class
    # (add_pdf_files, add_folder, validate_inputs, start_processing, etc.)
    # ... [All other methods remain the same as in the original regulatory_gui.py]
    
    def add_pdf_files(self):
        """Add PDF files to processing list."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Sélectionner les Fichiers PDF", "", "Fichiers PDF (*.pdf)"
        )
        if files:
            current_text = self.paths_list.toPlainText()
            new_paths = []
            if current_text:
                new_paths.append(current_text)
            new_paths.extend(files)
            self.paths_list.setPlainText("\n".join(new_paths))
    
    def add_folder(self):
        """Add folder to processing list."""
        folder = QFileDialog.getExistingDirectory(self, "Sélectionner un Dossier")
        if folder:
            current_text = self.paths_list.toPlainText()
            new_paths = []
            if current_text:
                new_paths.append(current_text)
            new_paths.append(folder)
            self.paths_list.setPlainText("\n".join(new_paths))
    
    def clear_paths(self):
        """Clear all paths."""
        self.paths_list.clear()
    
    def browse_output_directory(self):
        """Browse for output directory."""
        directory = QFileDialog.getExistingDirectory(self, "Sélectionner le Répertoire de Sortie")
        if directory:
            self.output_dir_input.setText(directory)
    
    def update_filename_preview(self):
        """Update the filename preview."""
        base_name = self.output_filename_input.text().strip() or "analyse_reglementaire"
        
        # Get user info
        first_name = self.name_input.text().strip()
        last_name = self.surname_input.text().strip()
        
        if first_name and last_name:
            user_part = f"{first_name}_{last_name}".replace(' ', '_').replace('-', '_')
            preview = f"{base_name}_{user_part}_AAAAMMJJ_HHMMSS.xlsx"
        elif first_name or last_name:
            user_part = (first_name or last_name).replace(' ', '_').replace('-', '_')
            preview = f"{base_name}_{user_part}_AAAAMMJJ_HHMMSS.xlsx"
        else:
            preview = f"{base_name}_AAAAMMJJ_HHMMSS.xlsx"
        
        self.filename_preview.setText(preview)
    
    def toggle_ai_settings(self, enabled: bool):
        """Toggle AI-related settings."""
        self.api_key_input.setEnabled(enabled)
        self.ai_model_combo.setEnabled(enabled)
        self.assess_materiality_checkbox.setEnabled(enabled)
    
    def load_basic_preset(self):
        """Load basic processing preset."""
        self.chunk_size_input.setValue(800)
        self.chunk_overlap_input.setValue(150)
        self.enable_ai_checkbox.setChecked(False)
        self.extract_articles_checkbox.setChecked(True)
        self.assess_materiality_checkbox.setChecked(False)
        self.clean_text_checkbox.setChecked(True)
    
    def load_ai_preset(self):
        """Load AI-enhanced preset."""
        self.chunk_size_input.setValue(1000)
        self.chunk_overlap_input.setValue(200)
        self.enable_ai_checkbox.setChecked(True)
        self.extract_articles_checkbox.setChecked(True)
        self.assess_materiality_checkbox.setChecked(True)
        self.clean_text_checkbox.setChecked(True)
    
    def save_preset(self):
        """Save current configuration as preset."""
        self.save_settings()
        QMessageBox.information(self, "Preset Sauvegardé", "Configuration actuelle sauvegardée comme preset par défaut.")
    
    def validate_inputs(self) -> bool:
        """Validate user inputs."""
        errors = []
        
        # Check user info
        if not self.name_input.text().strip():
            errors.append("Le prénom est requis")
        if not self.surname_input.text().strip():
            errors.append("Le nom de famille est requis")
        
        # Check document paths
        paths = [p.strip() for p in self.paths_list.toPlainText().split('\n') if p.strip()]
        if not paths:
            errors.append("Au moins un chemin de document est requis")
        else:
            for path in paths:
                if not os.path.exists(path):
                    errors.append(f"Le chemin n'existe pas: {path}")
        
        # Check output settings
        if not self.output_dir_input.text().strip():
            errors.append("Le répertoire de sortie est requis")
        elif not os.path.exists(self.output_dir_input.text()):
            errors.append("Le répertoire de sortie n'existe pas")
        
        if not self.output_filename_input.text().strip():
            errors.append("Le nom de fichier de sortie est requis")
        
        # Check AI settings
        if self.enable_ai_checkbox.isChecked():
            api_key = self.api_key_input.text().strip() or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                errors.append("La clé API est requise pour la validation IA")
        
        if errors:
            QMessageBox.warning(self, "Erreurs de Validation", "\n".join(f"• {error}" for error in errors))
            return False
        
        self.start_processing_btn.setEnabled(True)
        QMessageBox.information(self, "Validation Réussie", "Toutes les entrées sont valides. Prêt à démarrer le traitement!")
        return True
    
    def start_processing(self):
        """Start the document processing."""
        if not self.validate_inputs():
            return
        
        # Create configuration
        config = ProcessorConfig(
            chunk_size=self.chunk_size_input.value(),
            chunk_overlap=self.chunk_overlap_input.value(),
            max_file_size_mb=self.max_file_size_input.value(),
            enable_ai_validation=self.enable_ai_checkbox.isChecked(),
            extract_articles=self.extract_articles_checkbox.isChecked(),
            assess_materiality=self.assess_materiality_checkbox.isChecked(),
            clean_text=self.clean_text_checkbox.isChecked(),
            anthropic_api_key=self.api_key_input.text().strip() or None,
            ai_model=self.ai_model_combo.currentText(),
            log_level="INFO"
        )
        
        # Get document paths
        paths = [p.strip() for p in self.paths_list.toPlainText().split('\n') if p.strip()]
        
        # Create output path with user info
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        user_name = f"{self.name_input.text().strip()}_{self.surname_input.text().strip()}"
        user_name = user_name.replace(' ', '_').replace('-', '_')  # Clean filename
        
        base_filename = self.output_filename_input.text().strip() or "analyse_reglementaire"
        filename = f"{base_filename}_{user_name}_{timestamp}.xlsx"
        output_path = os.path.join(self.output_dir_input.text(), filename)
        
        # User info
        user_info = {
            'name': self.name_input.text().strip(),
            'surname': self.surname_input.text().strip()
        }
        
        # Start processing thread
        self.processing_thread = ProcessingThread(config, paths, output_path, user_info)
        self.processing_thread.progress_update.connect(self.update_progress)
        self.processing_thread.document_processed.connect(self.document_processed)
        self.processing_thread.processing_complete.connect(self.processing_complete)
        self.processing_thread.error_occurred.connect(self.processing_error)
        self.processing_thread.log_message.connect(self.add_log_message)
        
        self.processing_thread.start()
        
        # Update UI state
        self.start_processing_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.tab_widget.setCurrentIndex(2)  # Switch to processing tab
        
        self.add_log_message("🚀 Traitement démarré...")
    
    def update_progress(self, percentage: int, message: str):
        """Update progress bar and status."""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(message)
        self.statusBar().showMessage(message)
    
    def document_processed(self, filename: str, summary: Dict[str, Any]):
        """Handle document processing completion."""
        row = self.results_table.rowCount()
        self.results_table.insertRow(row)
        
        items = [
            filename,
            summary.get('type', 'unknown'),
            str(summary.get('articles', 0)),
            str(summary.get('chunks', 0)),
            f"{summary.get('validation_score', 0):.1f}",
            "✅ Succès"
        ]
        
        for col, item in enumerate(items):
            self.results_table.setItem(row, col, QTableWidgetItem(item))
        
        self.processed_documents.append(summary)
    
    def processing_complete(self, output_file: str, summary: Dict[str, Any]):
        """Handle processing completion."""
        self.start_processing_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        # Update summary
        self.summary_labels['total_docs'].setText(str(summary.get('total_documents', 0)))
        self.summary_labels['successful'].setText(str(summary.get('successful_extractions', 0)))
        self.summary_labels['failed'].setText(str(summary.get('total_errors', 0)))
        
        total_articles = sum(doc.get('articles', 0) for doc in self.processed_documents)
        self.summary_labels['articles'].setText(str(total_articles))
        
        scores = [doc.get('validation_score', 0) for doc in self.processed_documents if doc.get('validation_score', 0) > 0]
        avg_score = sum(scores) / len(scores) if scores else 0
        self.summary_labels['avg_score'].setText(f"{avg_score:.1f}")
        
        # Update output path
        self.output_path_label.setText(f"📄 {os.path.basename(output_file)}")
        self.open_file_btn.setEnabled(True)
        self.open_folder_btn.setEnabled(True)
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(3)
        
        # Show completion message
        QMessageBox.information(
            self, 
            "Traitement Terminé", 
            f"Traitement réussi de {summary.get('processed_count', 0)} documents.\n\n"
            f"Fichier de sortie: {output_file}\n\n"
            f"Total articles extraits: {total_articles}\n"
            f"Score de validation moyen: {avg_score:.1f}/100"
        )
        
        self.add_log_message(f"🎉 Traitement terminé avec succès!")
    
    def processing_error(self, error_message: str):
        """Handle processing errors."""
        self.start_processing_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        
        QMessageBox.critical(self, "Erreur de Traitement", error_message)
        self.add_log_message(f"❌ Échec du traitement: {error_message}")
    
    def cancel_processing(self):
        """Cancel the current processing."""
        if self.processing_thread and self.processing_thread.isRunning():
            self.processing_thread.cancel()
            self.processing_thread.wait()
        
        self.start_processing_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.status_label.setText("Traitement annulé")
        self.add_log_message("⏹️ Traitement annulé par l'utilisateur")
    
    def add_log_message(self, message: str):
        """Add message to processing log."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
        self.log_display.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_log(self):
        """Clear the processing log."""
        self.log_display.clear()
    
    def save_log(self):
        """Save the processing log to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Sauvegarder Journal", f"journal_traitement_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Fichiers Texte (*.txt)"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(self.log_display.toPlainText())
                QMessageBox.information(self, "Journal Sauvegardé", f"Journal sauvegardé dans: {filename}")
            except Exception as e:
                QMessageBox.warning(self, "Erreur de Sauvegarde", f"Échec de la sauvegarde du journal: {str(e)}")
    
    def open_output_file(self):
        """Open the output Excel file."""
        if hasattr(self, 'processing_thread') and self.processing_thread:
            output_file = self.processing_thread.output_path
            if os.path.exists(output_file):
                os.startfile(output_file)  # Windows
    
    def open_output_folder(self):
        """Open the output folder."""
        if hasattr(self, 'processing_thread') and self.processing_thread:
            output_dir = os.path.dirname(self.processing_thread.output_path)
            if os.path.exists(output_dir):
                os.startfile(output_dir)  # Windows
    
    def save_settings(self):
        """Save application settings."""
        self.settings.setValue('name', self.name_input.text())
        self.settings.setValue('surname', self.surname_input.text())
        self.settings.setValue('output_dir', self.output_dir_input.text())
        self.settings.setValue('chunk_size', self.chunk_size_input.value())
        self.settings.setValue('chunk_overlap', self.chunk_overlap_input.value())
        self.settings.setValue('enable_ai', self.enable_ai_checkbox.isChecked())
        self.settings.setValue('ai_model', self.ai_model_combo.currentText())
    
    def load_settings(self):
        """Load application settings."""
        self.name_input.setText(self.settings.value('name', ''))
        self.surname_input.setText(self.settings.value('surname', ''))
        self.output_dir_input.setText(self.settings.value('output_dir', ''))
        self.chunk_size_input.setValue(int(self.settings.value('chunk_size', 1000)))
        self.chunk_overlap_input.setValue(int(self.settings.value('chunk_overlap', 200)))
        self.enable_ai_checkbox.setChecked(self.settings.value('enable_ai', True, type=bool))
        
        ai_model = self.settings.value('ai_model', 'claude-3-haiku-20240307')
        index = self.ai_model_combo.findText(ai_model)
        if index >= 0:
            self.ai_model_combo.setCurrentIndex(index)
    
    def closeEvent(self, event):
        """Handle application close."""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self, 'Fermer l\'Application',
                'Le traitement est toujours en cours. Voulez-vous annuler et quitter?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_processing()
                self.save_settings()
                event.accept()
            else:
                event.ignore()
        else:
            self.save_settings()
            event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("JABE Regulatory Document Processor")
    app.setApplicationVersion("2.0")
    
    window = JABERegulatoryGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()