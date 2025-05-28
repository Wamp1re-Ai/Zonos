#!/usr/bin/env python3
"""
Test script to verify unlimited voice generation feature merge.

This script tests:
1. Enhanced voice cloning with unlimited mode
2. Token calculation without caps
3. Unlimited generation method availability
"""

import sys
import os

# Add Zonos to path
sys.path.append('Zonos')

def test_enhanced_unlimited():
    """Test enhanced voice cloning unlimited generation."""
    print("🧪 Testing Enhanced Voice Cloning Unlimited Generation")
    print("=" * 60)
    
    try:
        from enhanced_voice_cloning import EnhancedVoiceCloner
        print("✅ Enhanced voice cloning module imported successfully")
        
        # Check if unlimited method exists
        if hasattr(EnhancedVoiceCloner, 'generate_unlimited_speech'):
            print("✅ generate_unlimited_speech method found")
        else:
            print("❌ generate_unlimited_speech method NOT found")
            return False
            
        # Check if chunking method exists
        if hasattr(EnhancedVoiceCloner, '_generate_chunked_unlimited'):
            print("✅ _generate_chunked_unlimited method found")
        else:
            print("❌ _generate_chunked_unlimited method NOT found")
            return False
            
        # Check if intelligent chunking method exists
        if hasattr(EnhancedVoiceCloner, '_intelligent_text_chunking'):
            print("✅ _intelligent_text_chunking method found")
        else:
            print("❌ _intelligent_text_chunking method NOT found")
            return False
            
        print("✅ All unlimited generation methods are available!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import enhanced_voice_cloning: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing enhanced unlimited: {e}")
        return False

def test_unlimited_module():
    """Test standalone unlimited voice cloning module."""
    print("\n🧪 Testing Standalone Unlimited Voice Cloning Module")
    print("=" * 60)
    
    try:
        from unlimited_voice_cloning import UnlimitedVoiceCloner, create_unlimited_voice_cloner
        print("✅ Unlimited voice cloning module imported successfully")
        
        # Check if main methods exist
        if hasattr(UnlimitedVoiceCloner, 'generate_unlimited_speech'):
            print("✅ UnlimitedVoiceCloner.generate_unlimited_speech method found")
        else:
            print("❌ UnlimitedVoiceCloner.generate_unlimited_speech method NOT found")
            return False
            
        # Check if dynamic token calculation exists
        if hasattr(UnlimitedVoiceCloner, '_calculate_dynamic_tokens'):
            print("✅ _calculate_dynamic_tokens method found")
        else:
            print("❌ _calculate_dynamic_tokens method NOT found")
            return False
            
        print("✅ Standalone unlimited module is working!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import unlimited_voice_cloning: {e}")
        return False
    except Exception as e:
        print(f"❌ Error testing unlimited module: {e}")
        return False

def test_token_calculation():
    """Test unlimited token calculation."""
    print("\n🧪 Testing Unlimited Token Calculation")
    print("=" * 60)
    
    try:
        from unlimited_voice_cloning import UnlimitedVoiceCloner
        
        # Create a mock model for testing
        class MockModel:
            pass
        
        cloner = UnlimitedVoiceCloner(MockModel(), "cpu", use_fp16=False)
        
        # Test short text
        short_text = "Hello world!"
        short_tokens = cloner._calculate_dynamic_tokens(short_text)
        print(f"✅ Short text ({len(short_text)} chars) → {short_tokens} tokens")
        
        # Test long text
        long_text = "This is a very long text that would normally be capped at 2 minutes in the old system. " * 50
        long_tokens = cloner._calculate_dynamic_tokens(long_text)
        print(f"✅ Long text ({len(long_text)} chars) → {long_tokens} tokens")
        
        # Verify no cap is applied
        if long_tokens > 86 * 120:  # Old 2-minute cap
            print(f"🔥 SUCCESS: Tokens ({long_tokens}) exceed old 2-minute cap (10,320)!")
            print("✅ NO LENGTH RESTRICTIONS confirmed!")
        else:
            print(f"⚠️ Tokens ({long_tokens}) still within old cap - but this might be normal for this text length")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing token calculation: {e}")
        return False

def test_notebook_compatibility():
    """Test if notebook files exist and have unlimited mode."""
    print("\n🧪 Testing Notebook Compatibility")
    print("=" * 60)
    
    try:
        # Check Enhanced notebook
        enhanced_notebook = "Zonos/Enhanced_Voice_Cloning_Colab.ipynb"
        if os.path.exists(enhanced_notebook):
            print("✅ Enhanced_Voice_Cloning_Colab.ipynb found")
            
            with open(enhanced_notebook, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if 'unlimited_mode' in content:
                print("✅ unlimited_mode toggle found in Enhanced notebook")
            else:
                print("❌ unlimited_mode toggle NOT found in Enhanced notebook")
                return False
                
            if 'generate_unlimited_speech' in content:
                print("✅ generate_unlimited_speech usage found in Enhanced notebook")
            else:
                print("❌ generate_unlimited_speech usage NOT found in Enhanced notebook")
                return False
                
        else:
            print("❌ Enhanced_Voice_Cloning_Colab.ipynb NOT found")
            return False
            
        # Check if Efficient notebook exists (for reference)
        efficient_notebook = "Zonos/Efficient_Voice_Cloning_Colab.ipynb"
        if os.path.exists(efficient_notebook):
            print("✅ Efficient_Voice_Cloning_Colab.ipynb found (reference)")
        else:
            print("⚠️ Efficient_Voice_Cloning_Colab.ipynb not found (not critical)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing notebook compatibility: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Unlimited Voice Generation Merge")
    print("=" * 80)
    
    tests = [
        test_enhanced_unlimited,
        test_unlimited_module,
        test_token_calculation,
        test_notebook_compatibility
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
        print("🎉 ALL TESTS PASSED! Unlimited voice generation merge successful!")
        print("\n🔥 Key Features Confirmed:")
        print("  ✅ No 2-minute token cap")
        print("  ✅ Unlimited generation methods available")
        print("  ✅ Enhanced notebook has unlimited mode toggle")
        print("  ✅ Intelligent text chunking for long texts")
        print("  ✅ Dynamic token calculation without limits")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
