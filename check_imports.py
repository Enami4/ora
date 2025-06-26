#!/usr/bin/env python3
"""Script to check for missing imports and undefined names in Python files."""

import ast
import os
import sys
from typing import Set, Dict, List, Tuple

def get_imports(tree: ast.AST) -> Dict[str, Set[str]]:
    """Extract all imports from an AST."""
    imports = {'modules': set(), 'names': set(), 'from_imports': {}}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                import_name = alias.asname if alias.asname else alias.name
                imports['modules'].add(module_name)
                imports['names'].add(import_name.split('.')[0])
                
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            for alias in node.names:
                import_name = alias.asname if alias.asname else alias.name
                imports['names'].add(import_name)
                if module not in imports['from_imports']:
                    imports['from_imports'][module] = set()
                imports['from_imports'][module].add(import_name)
    
    return imports

def get_defined_names(tree: ast.AST) -> Set[str]:
    """Extract all defined names (functions, classes, variables) from an AST."""
    defined = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined.add(node.name)
        elif isinstance(node, ast.ClassDef):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
    
    return defined

def get_used_names(tree: ast.AST) -> Dict[str, List[int]]:
    """Extract all used names and their line numbers from an AST."""
    used = {}
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if hasattr(node, 'lineno'):
                if node.id not in used:
                    used[node.id] = []
                used[node.id].append(node.lineno)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Name) and hasattr(node.value, 'lineno'):
                if node.value.id not in used:
                    used[node.value.id] = []
                used[node.value.id].append(node.value.lineno)
    
    return used

def check_file(file_path: str) -> Dict[str, List[Tuple[str, int]]]:
    """Check a Python file for import issues."""
    issues = {
        'undefined': [],
        'unused_imports': [],
        'missing_imports': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=file_path)
        
        imports = get_imports(tree)
        defined = get_defined_names(tree)
        used = get_used_names(tree)
        
        # Python builtins and common names
        builtins = set(dir(__builtins__))
        common_names = {
            'self', 'cls', '__name__', '__file__', '__doc__', '__all__',
            'True', 'False', 'None', 'NotImplemented', 'Ellipsis',
            '__debug__', '__import__', '__loader__', '__spec__',
            '__package__', '__cached__', 'ProcessorConfig'  # Forward reference in interfaces.py
        }
        
        # Check for undefined names
        all_available = imports['names'] | defined | builtins | common_names
        
        for name, lines in used.items():
            if name not in all_available:
                for line in lines:
                    issues['undefined'].append((name, line))
        
        # Check for unused imports
        all_used_names = set(used.keys())
        for imported_name in imports['names']:
            if imported_name not in all_used_names and imported_name not in defined:
                issues['unused_imports'].append((imported_name, 0))
        
        # Check for common missing imports based on usage patterns
        common_modules = {
            'os': ['path', 'environ', 'makedirs', 'listdir', 'walk'],
            'sys': ['argv', 'exit', 'path', 'stdout', 'stderr'],
            'json': ['loads', 'dumps', 'load', 'dump'],
            're': ['match', 'search', 'findall', 'sub', 'compile'],
            'logging': ['getLogger', 'basicConfig', 'info', 'error', 'warning'],
            'datetime': ['now', 'today', 'timedelta'],
            'pathlib': ['Path'],
        }
        
        for module, attrs in common_modules.items():
            for attr in attrs:
                if attr in all_used_names and module not in imports['modules']:
                    issues['missing_imports'].append((f"{module}.{attr}", 0))
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    
    return issues

def main():
    """Main function to check all Python files in the regulatory_processor directory."""
    base_dir = "/mnt/c/Users/doupa/Desktop/Ventures/Orabank/regulatory_processor"
    
    all_issues = {}
    
    for filename in os.listdir(base_dir):
        if filename.endswith('.py'):
            file_path = os.path.join(base_dir, filename)
            issues = check_file(file_path)
            
            if any(issues[key] for key in issues):
                all_issues[filename] = issues
    
    # Print results
    if not all_issues:
        print("No import issues found in any files!")
    else:
        for filename, issues in sorted(all_issues.items()):
            print(f"\n{'='*60}")
            print(f"FILE: {filename}")
            print(f"{'='*60}")
            
            if issues['undefined']:
                print("\nUNDEFINED NAMES (used but not imported/defined):")
                for name, line in sorted(set(issues['undefined'])):
                    print(f"  Line {line}: '{name}' is not defined")
            
            if issues['unused_imports']:
                print("\nUNUSED IMPORTS:")
                for name, _ in sorted(set(issues['unused_imports'])):
                    print(f"  '{name}' is imported but never used")
            
            if issues['missing_imports']:
                print("\nPOTENTIAL MISSING IMPORTS:")
                for name, _ in sorted(set(issues['missing_imports'])):
                    print(f"  '{name}' might need to be imported")

if __name__ == "__main__":
    main()