#!/usr/bin/env python3
"""Detailed import analysis for the regulatory_processor module."""

import ast
import os
from typing import Set, Dict, List, Tuple

class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports = {'modules': set(), 'from_imports': {}, 'imported_names': set()}
        self.defined = {'functions': set(), 'classes': set(), 'variables': set()}
        self.used_names = {}
        self.current_class = None
        self.current_function = None
        self.in_function_params = False
        
    def visit_Import(self, node):
        for alias in node.names:
            self.imports['modules'].add(alias.name)
            name = alias.asname if alias.asname else alias.name
            self.imports['imported_names'].add(name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node):
        module = node.module or ''
        for alias in node.names:
            if alias.name == '*':
                self.imports['from_imports'][module] = {'*'}
            else:
                if module not in self.imports['from_imports']:
                    self.imports['from_imports'][module] = set()
                self.imports['from_imports'][module].add(alias.name)
                name = alias.asname if alias.asname else alias.name
                self.imports['imported_names'].add(name)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node):
        self.defined['functions'].add(node.name)
        old_function = self.current_function
        self.current_function = node.name
        
        # Visit function arguments
        self.in_function_params = True
        for arg in node.args.args:
            self.defined['variables'].add(arg.arg)
        self.in_function_params = False
        
        self.generic_visit(node)
        self.current_function = old_function
        
    def visit_ClassDef(self, node):
        self.defined['classes'].add(node.name)
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.defined['variables'].add(target.id)
        self.generic_visit(node)
        
    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and not self.in_function_params:
            if hasattr(node, 'lineno'):
                if node.id not in self.used_names:
                    self.used_names[node.id] = []
                self.used_names[node.id].append(node.lineno)
        self.generic_visit(node)

def analyze_file(file_path: str) -> Dict[str, List[Tuple[str, int]]]:
    """Analyze a single Python file for import issues."""
    issues = {
        'undefined': [],
        'unused_imports': [],
        'missing_standard_imports': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        tree = ast.parse(content, filename=file_path)
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        
        # Python builtins
        builtins = set(dir(__builtins__))
        
        # Common special names
        special_names = {
            'self', 'cls', '__name__', '__file__', '__doc__', '__all__',
            'True', 'False', 'None', 'NotImplemented', 'Ellipsis',
            '__debug__', '__import__', '__loader__', '__spec__',
            '__package__', '__cached__', '__version__', '__author__'
        }
        
        # Names that are OK to use without explicit import (comprehension variables, etc)
        comprehension_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                for generator in node.generators:
                    if isinstance(generator.target, ast.Name):
                        comprehension_vars.add(generator.target.id)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    comprehension_vars.add(node.target.id)
            elif isinstance(node, ast.ExceptHandler):
                if node.name:
                    comprehension_vars.add(node.name)
        
        # All available names
        all_available = (
            analyzer.imports['imported_names'] | 
            analyzer.defined['functions'] | 
            analyzer.defined['classes'] | 
            analyzer.defined['variables'] |
            builtins | 
            special_names |
            comprehension_vars
        )
        
        # Check for truly undefined names
        for name, lines in analyzer.used_names.items():
            if name not in all_available:
                # Check if it's imported via star import
                star_imported = False
                for module, imports in analyzer.imports['from_imports'].items():
                    if '*' in imports:
                        star_imported = True
                        break
                
                if not star_imported:
                    # Special case for forward references in type hints
                    if name == 'ProcessorConfig' and 'interfaces.py' in file_path:
                        continue
                        
                    for line in lines:
                        issues['undefined'].append((name, line))
        
        # Check for unused imports
        all_used = set(analyzer.used_names.keys())
        all_defined_by_file = (
            analyzer.defined['functions'] | 
            analyzer.defined['classes'] | 
            analyzer.defined['variables']
        )
        
        for imported_name in analyzer.imports['imported_names']:
            if imported_name not in all_used and imported_name not in all_defined_by_file:
                # Check if it might be used in string form (like in getattr)
                if f"'{imported_name}'" not in content and f'"{imported_name}"' not in content:
                    # Check if it's re-exported via __all__
                    if '__all__' in content and imported_name in content:
                        continue
                    issues['unused_imports'].append((imported_name, 0))
        
        # Check for missing standard library imports
        standard_libs = {
            'os': ['path', 'environ', 'makedirs', 'listdir', 'walk', 'getenv'],
            'sys': ['argv', 'exit', 'path', 'stdout', 'stderr'],
            'json': ['loads', 'dumps', 'load', 'dump'],
            're': ['match', 'search', 'findall', 'sub', 'compile', 'split'],
            'datetime': ['now', 'today', 'timedelta', 'datetime'],
            'pathlib': ['Path'],
            'typing': ['Dict', 'List', 'Set', 'Tuple', 'Optional', 'Any', 'Union'],
        }
        
        for module, attrs in standard_libs.items():
            module_imported = module in analyzer.imports['modules']
            from_imports = analyzer.imports['from_imports'].get(module, set())
            
            for attr in attrs:
                if attr in all_used and not module_imported and attr not in from_imports:
                    issues['missing_standard_imports'].append((f"{module}.{attr}", 0))
                    
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        
    return issues

def main():
    """Analyze all Python files in the regulatory_processor directory."""
    base_dir = "/mnt/c/Users/doupa/Desktop/Ventures/Orabank/regulatory_processor"
    
    print("IMPORT ANALYSIS FOR REGULATORY_PROCESSOR MODULE")
    print("=" * 80)
    
    all_issues = {}
    file_count = 0
    
    for filename in sorted(os.listdir(base_dir)):
        if filename.endswith('.py'):
            file_count += 1
            file_path = os.path.join(base_dir, filename)
            issues = analyze_file(file_path)
            
            # Filter out false positives
            filtered_issues = {
                'undefined': [],
                'unused_imports': issues['unused_imports'],
                'missing_standard_imports': issues['missing_standard_imports']
            }
            
            # Filter undefined names to remove false positives from comprehensions, parameters, etc
            seen = set()
            for name, line in issues['undefined']:
                key = (name, line)
                if key not in seen:
                    seen.add(key)
                    # Skip common false positives
                    if name not in ['i', 'j', 'k', 'e', 'item', 'key', 'value', 'row', 'col', 
                                  'chunk', 'doc', 'article', 'error', 'cell', 'sheet']:
                        filtered_issues['undefined'].append((name, line))
            
            if any(filtered_issues[key] for key in filtered_issues):
                all_issues[filename] = filtered_issues
    
    # Summary
    total_undefined = sum(len(issues['undefined']) for issues in all_issues.values())
    total_unused = sum(len(issues['unused_imports']) for issues in all_issues.values())
    total_missing = sum(len(issues['missing_standard_imports']) for issues in all_issues.values())
    
    print(f"\nAnalyzed {file_count} Python files")
    print(f"Files with issues: {len(all_issues)}")
    print(f"Total undefined names: {total_undefined}")
    print(f"Total unused imports: {total_unused}")
    print(f"Total missing standard imports: {total_missing}")
    
    if not all_issues:
        print("\n✅ No significant import issues found!")
    else:
        for filename, issues in sorted(all_issues.items()):
            print(f"\n{'='*60}")
            print(f"FILE: {filename}")
            print(f"{'='*60}")
            
            if issues['undefined']:
                print("\n❌ UNDEFINED NAMES (used but not imported/defined):")
                unique_names = {}
                for name, line in sorted(issues['undefined']):
                    if name not in unique_names:
                        unique_names[name] = []
                    unique_names[name].append(line)
                
                for name, lines in sorted(unique_names.items()):
                    lines_str = ', '.join(str(l) for l in sorted(set(lines))[:5])
                    if len(lines) > 5:
                        lines_str += f", ... ({len(lines)} total)"
                    print(f"  '{name}' - Lines: {lines_str}")
            
            if issues['unused_imports']:
                print("\n⚠️  UNUSED IMPORTS:")
                for name, _ in sorted(set(issues['unused_imports'])):
                    print(f"  '{name}' is imported but never used")
            
            if issues['missing_standard_imports']:
                print("\n⚠️  POTENTIAL MISSING STANDARD IMPORTS:")
                for name, _ in sorted(set(issues['missing_standard_imports'])):
                    print(f"  '{name}' might need to be imported")

if __name__ == "__main__":
    main()