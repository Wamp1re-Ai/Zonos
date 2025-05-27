#!/usr/bin/env python3
"""
Simple notebook validator for Google Colab compatibility
"""

import json
import sys
from pathlib import Path

def validate_colab_notebook(notebook_path):
    """
    Quick validation check for Colab compatibility
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # Basic structure checks
        required_fields = ['cells', 'metadata', 'nbformat']
        missing = [field for field in required_fields if field not in notebook]
        
        if missing:
            print(f"❌ {notebook_path}: Missing fields: {missing}")
            return False
        
        cell_count = len(notebook.get('cells', []))
        nbformat = notebook.get('nbformat', 0)
        
        print(f"✅ {notebook_path}: {cell_count} cells, nbformat {nbformat}")
        return True
        
    except json.JSONDecodeError as e:
        print(f"❌ {notebook_path}: JSON error - {e}")
        return False
    except Exception as e:
        print(f"❌ {notebook_path}: Error - {e}")
        return False

def main():
    """
    Validate all notebook files in the current directory
    """
    notebook_files = list(Path('.').glob('*.ipynb'))
    
    if not notebook_files:
        print("No .ipynb files found in current directory")
        return
    
    print("🔍 Validating Jupyter notebooks for Google Colab compatibility...\n")
    
    all_valid = True
    for notebook_file in sorted(notebook_files):
        if not validate_colab_notebook(notebook_file):
            all_valid = False
    
    print(f"\n{'✅ All notebooks are valid!' if all_valid else '❌ Some notebooks have issues'}")

if __name__ == "__main__":
    main()
