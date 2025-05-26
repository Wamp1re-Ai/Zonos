#!/usr/bin/env python3
"""
Validate that the Enhanced Voice Cloning Colab notebook is properly formatted.
"""

import json
import sys

def validate_notebook():
    """Validate the notebook JSON structure and content."""
    
    print("🔍 Validating Enhanced Voice Cloning Colab Notebook")
    print("=" * 55)
    
    try:
        # Test 1: Load and parse JSON
        print("\n1. Testing JSON structure...")
        with open('Enhanced_Voice_Cloning_Colab.ipynb', 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        print("✅ JSON structure is valid")
        
        # Test 2: Check notebook format
        print("\n2. Checking notebook format...")
        assert 'cells' in notebook, "Missing 'cells' key"
        assert 'metadata' in notebook, "Missing 'metadata' key"
        assert 'nbformat' in notebook, "Missing 'nbformat' key"
        print("✅ Notebook format is correct")
        
        # Test 3: Check cell count and types
        print("\n3. Analyzing cells...")
        cells = notebook['cells']
        print(f"Total cells: {len(cells)}")
        
        cell_types = {}
        for i, cell in enumerate(cells):
            cell_type = cell.get('cell_type', 'unknown')
            cell_types[cell_type] = cell_types.get(cell_type, 0) + 1
            
            # Check for duplicate Cell 1
            if cell_type == 'code' and 'source' in cell:
                source = ''.join(cell['source'])
                if '#@title 1.' in source:
                    print(f"  Cell {i+1}: {cell_type} - Setup cell")
        
        print(f"Cell types: {cell_types}")
        print("✅ Cell structure looks good")
        
        # Test 4: Check for UV installation
        print("\n4. Checking for UV installation...")
        uv_found = False
        for cell in cells:
            if cell.get('cell_type') == 'code' and 'source' in cell:
                source = ''.join(cell['source'])
                if 'uv pip install' in source and 'Ultra-Fast' in source:
                    uv_found = True
                    print("✅ UV ultra-fast installation found")
                    break
        
        if not uv_found:
            print("⚠️ UV installation not found")
        
        # Test 5: Check for enhanced fallback functions
        print("\n5. Checking for enhanced fallback functions...")
        fallback_found = False
        for cell in cells:
            if cell.get('cell_type') == 'code' and 'source' in cell:
                source = ''.join(cell['source'])
                if 'simple_enhanced_clone_voice' in source and 'fallback enhanced' in source:
                    fallback_found = True
                    print("✅ Enhanced fallback functions found")
                    break
        
        if not fallback_found:
            print("⚠️ Enhanced fallback functions not found")
        
        # Test 6: Check for duplicate cells
        print("\n6. Checking for duplicate cells...")
        titles = []
        duplicates = []
        
        for i, cell in enumerate(cells):
            if cell.get('cell_type') == 'code' and 'source' in cell:
                source = ''.join(cell['source'])
                for line in source.split('\n'):
                    if line.strip().startswith('#@title'):
                        title = line.strip()
                        if title in titles:
                            duplicates.append((i+1, title))
                        else:
                            titles.append(title)
                        break
        
        if duplicates:
            print(f"⚠️ Found duplicate cells: {duplicates}")
        else:
            print("✅ No duplicate cells found")
        
        print(f"\nFound {len(titles)} titled cells:")
        for title in titles:
            print(f"  - {title}")
        
        # Test 7: Summary
        print(f"\n📊 Validation Summary:")
        print(f"  - JSON structure: ✅ Valid")
        print(f"  - Notebook format: ✅ Correct")
        print(f"  - Total cells: {len(cells)}")
        print(f"  - Code cells: {cell_types.get('code', 0)}")
        print(f"  - Markdown cells: {cell_types.get('markdown', 0)}")
        print(f"  - UV installation: {'✅' if uv_found else '⚠️'}")
        print(f"  - Enhanced fallbacks: {'✅' if fallback_found else '⚠️'}")
        print(f"  - Duplicate cells: {'⚠️' if duplicates else '✅ None'}")
        
        if uv_found and fallback_found and not duplicates:
            print(f"\n🎉 Notebook validation PASSED!")
            print(f"The Enhanced Voice Cloning Colab notebook is ready to use!")
            return True
        else:
            print(f"\n⚠️ Notebook validation completed with warnings.")
            return False
            
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return False
    except Exception as e:
        print(f"❌ Validation error: {e}")
        return False

if __name__ == "__main__":
    success = validate_notebook()
    sys.exit(0 if success else 1)
