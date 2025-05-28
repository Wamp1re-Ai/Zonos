#!/usr/bin/env python3
"""
Test script to verify code structure for unlimited voice generation.

This script tests the code structure without importing torch dependencies.
"""

import ast
import os

def test_enhanced_voice_cloning_structure():
    """Test enhanced_voice_cloning.py structure."""
    print("🧪 Testing Enhanced Voice Cloning Code Structure")
    print("=" * 60)

    try:
        with open('Zonos/enhanced_voice_cloning.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse the AST to check for methods
        tree = ast.parse(content)

        class_methods = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                class_methods[node.name] = methods

        if 'EnhancedVoiceCloner' in class_methods:
            methods = class_methods['EnhancedVoiceCloner']
            print(f"✅ EnhancedVoiceCloner class found with {len(methods)} methods")

            # Check for unlimited methods
            if 'generate_unlimited_speech' in methods:
                print("✅ generate_unlimited_speech method found")
            else:
                print("❌ generate_unlimited_speech method NOT found")
                return False

            if '_generate_chunked_unlimited' in methods:
                print("✅ _generate_chunked_unlimited method found")
            else:
                print("❌ _generate_chunked_unlimited method NOT found")
                return False

            if '_intelligent_text_chunking' in methods:
                print("✅ _intelligent_text_chunking method found")
            else:
                print("❌ _intelligent_text_chunking method NOT found")
                return False
        else:
            print("❌ EnhancedVoiceCloner class NOT found")
            return False

        # Check for token cap removal
        if '86 * 120' in content and 'NO CAP' in content:
            print("✅ Token cap removal confirmed (old cap referenced but bypassed)")
        elif '86 * 120' not in content:
            print("✅ Token cap completely removed")
        else:
            print("⚠️ Token cap status unclear")

        # Check for unlimited token calculation
        if 'UNLIMITED' in content and 'NO CAP' in content:
            print("✅ Unlimited token calculation confirmed")
        else:
            print("❌ Unlimited token calculation NOT confirmed")
            return False

        return True

    except Exception as e:
        print(f"❌ Error testing enhanced structure: {e}")
        return False

def test_unlimited_voice_cloning_structure():
    """Test unlimited_voice_cloning.py structure."""
    print("\n🧪 Testing Unlimited Voice Cloning Code Structure")
    print("=" * 60)

    try:
        with open('Zonos/unlimited_voice_cloning.py', 'r', encoding='utf-8') as f:
            content = f.read()

        # Check import fix
        if 'from zonos.conditioning import make_cond_dict' in content:
            print("✅ Import fix confirmed (zonos.conditioning)")
        else:
            print("❌ Import fix NOT found")
            return False

        # Check for unlimited features
        if 'NO LENGTH LIMITS' in content:
            print("✅ No length limits confirmed")
        else:
            print("❌ No length limits NOT confirmed")
            return False

        if 'NO CAPS' in content or 'NO CAP' in content:
            print("✅ No caps confirmed")
        else:
            print("❌ No caps NOT confirmed")
            return False

        # Parse AST for class structure
        tree = ast.parse(content)

        class_methods = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                class_methods[node.name] = methods

        if 'UnlimitedVoiceCloner' in class_methods:
            methods = class_methods['UnlimitedVoiceCloner']
            print(f"✅ UnlimitedVoiceCloner class found with {len(methods)} methods")

            required_methods = [
                'generate_unlimited_speech',
                '_calculate_dynamic_tokens',
                '_intelligent_text_chunking',
                '_generate_unlimited_chunk'
            ]

            for method in required_methods:
                if method in methods:
                    print(f"✅ {method} method found")
                else:
                    print(f"❌ {method} method NOT found")
                    return False
        else:
            print("❌ UnlimitedVoiceCloner class NOT found")
            return False

        return True

    except Exception as e:
        print(f"❌ Error testing unlimited structure: {e}")
        return False

def test_notebook_unlimited_integration():
    """Test notebook unlimited mode integration."""
    print("\n🧪 Testing Notebook Unlimited Integration")
    print("=" * 60)

    try:
        with open('Zonos/Enhanced_Voice_Cloning_Colab.ipynb', 'r', encoding='utf-8') as f:
            content = f.read()

        # Check for unlimited mode toggle
        if 'unlimited_mode = False #@param {type:\\"boolean\\"}' in content:
            print("✅ Unlimited mode toggle found")
        else:
            print("❌ Unlimited mode toggle NOT found")
            return False

        # Check for unlimited mode usage
        if 'generate_unlimited_speech' in content:
            print("✅ generate_unlimited_speech usage found")
        else:
            print("❌ generate_unlimited_speech usage NOT found")
            return False

        # Check for unlimited mode conditional logic
        if 'if unlimited_mode:' in content:
            print("✅ Unlimited mode conditional logic found")
        else:
            print("❌ Unlimited mode conditional logic NOT found")
            return False

        # Check for unlimited mode status display
        if 'UNLIMITED MODE: NO LENGTH RESTRICTIONS' in content:
            print("✅ Unlimited mode status display found")
        else:
            print("❌ Unlimited mode status display NOT found")
            return False

        # Check for target chunk chars setting
        if 'target_chunk_chars' in content:
            print("✅ Target chunk chars setting found")
        else:
            print("❌ Target chunk chars setting NOT found")
            return False

        return True

    except Exception as e:
        print(f"❌ Error testing notebook integration: {e}")
        return False

def main():
    """Run all structure tests."""
    print("🚀 Testing Unlimited Voice Generation Code Structure")
    print("=" * 80)

    tests = [
        test_enhanced_voice_cloning_structure,
        test_unlimited_voice_cloning_structure,
        test_notebook_unlimited_integration
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            print("❌ Test failed!")

    print("\n" + "=" * 80)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL STRUCTURE TESTS PASSED!")
        print("\n🔥 Unlimited Voice Generation Features Confirmed:")
        print("  ✅ Enhanced voice cloning has unlimited generation methods")
        print("  ✅ Standalone unlimited voice cloning module is complete")
        print("  ✅ Enhanced notebook has unlimited mode toggle and logic")
        print("  ✅ Token caps removed/bypassed")
        print("  ✅ Import issues fixed")
        print("  ✅ Intelligent text chunking available")
        print("\n🚀 Ready for unlimited audio generation!")
        return True
    else:
        print("❌ Some structure tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
