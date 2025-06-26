"""
JABE Regulatory Processor Test Suite
====================================

Comprehensive testing framework ensuring application consistency and reliability.

Test Structure:
- unit/: Unit tests for individual modules
- integration/: Integration tests for complete workflows
- gui/: Automated GUI testing
- performance/: Performance and load testing
- fixtures/: Test data and mock objects
- reports/: Test execution reports
"""

import sys
import os
from pathlib import Path

# Add the parent directory to Python path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

__version__ = "1.0.0"
__author__ = "JABE Test Team"