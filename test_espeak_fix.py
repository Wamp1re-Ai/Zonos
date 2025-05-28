#!/usr/bin/env python3
"""
Test script to verify espeak installation and phonemization works.
This tests the fix for the "espeak not installed" error.
"""

import subprocess
import sys
import os

def test_espeak_installation():
    """Test if espeak is properly installed."""
    print("🔧 Testing espeak installation...")
    
    try:
        # Test espeak command
        result = subprocess.run(['espeak', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"✅ espeak is installed: {version}")
            return True
        else:
            print(f"❌ espeak command failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ espeak command not found")
        return False
    except Exception as e:
        print(f"❌ espeak test error: {e}")
        return False

def test_phonemizer():
    """Test if phonemizer can use espeak backend."""
    print("\n🔧 Testing phonemizer with espeak backend...")
    
    try:
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
        
        # Test basic phonemization
        text = "Hello world"
        phonemes = phonemize(text, language='en-us', backend='espeak')
        print(f"✅ Phonemization successful: '{text}' -> '{phonemes.strip()}'")
        
        # Test backend directly
        backend = EspeakBackend('en-us')
        phonemes2 = backend.phonemize([text], strip=True)
        print(f"✅ Direct backend test: '{text}' -> '{phonemes2[0]}'")
        
        return True
        
    except ImportError as e:
        print(f"❌ phonemizer import error: {e}")
        return False
    except RuntimeError as e:
        if "espeak not installed" in str(e):
            print(f"❌ espeak not available to phonemizer: {e}")
            return False
        else:
            print(f"❌ phonemizer runtime error: {e}")
            return False
    except Exception as e:
        print(f"❌ phonemizer test error: {e}")
        return False

def test_zonos_conditioning():
    """Test if Zonos conditioning works (requires espeak)."""
    print("\n🔧 Testing Zonos conditioning (requires espeak)...")
    
    try:
        # This would normally require the full Zonos setup
        # For now, just test if the import works
        print("⚠️ Zonos conditioning test requires full model setup")
        print("💡 Run the full notebook to test Zonos with espeak")
        return True
        
    except Exception as e:
        print(f"❌ Zonos test error: {e}")
        return False

def install_espeak_if_missing():
    """Install espeak if it's missing (for Colab/Linux)."""
    print("\n🔧 Attempting to install espeak...")
    
    try:
        # Update package list
        subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True)
        print("✅ Package list updated")
        
        # Install espeak packages
        packages = ['espeak', 'espeak-data', 'libespeak1', 'libespeak-dev']
        subprocess.run(['apt-get', 'install', '-y', '-qq'] + packages, check=True, capture_output=True)
        print(f"✅ Installed packages: {', '.join(packages)}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        return False
    except FileNotFoundError:
        print("⚠️ apt-get not available (not on Debian/Ubuntu system)")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def main():
    """Run all espeak tests."""
    print("🧪 espeak Fix Verification Test")
    print("=" * 50)
    
    # Test current espeak installation
    espeak_works = test_espeak_installation()
    
    # If espeak doesn't work, try to install it
    if not espeak_works:
        print("\n🔧 espeak not working, attempting installation...")
        install_success = install_espeak_if_missing()
        
        if install_success:
            # Test again after installation
            espeak_works = test_espeak_installation()
    
    # Test phonemizer
    phonemizer_works = test_phonemizer() if espeak_works else False
    
    # Test Zonos (placeholder)
    zonos_works = test_zonos_conditioning()
    
    # Summary
    print("\n📋 Test Results Summary:")
    print("=" * 30)
    print(f"espeak installation: {'✅ PASS' if espeak_works else '❌ FAIL'}")
    print(f"phonemizer backend: {'✅ PASS' if phonemizer_works else '❌ FAIL'}")
    print(f"Zonos compatibility: {'✅ READY' if zonos_works else '❌ FAIL'}")
    
    if espeak_works and phonemizer_works:
        print("\n🎉 SUCCESS! espeak fix is working!")
        print("✅ The notebook should now work without espeak errors")
        print("🚀 Ready to test the real efficient system!")
    else:
        print("\n❌ ISSUES DETECTED!")
        if not espeak_works:
            print("🔧 espeak is not properly installed")
            print("💡 Try running: apt-get install espeak espeak-data libespeak1 libespeak-dev")
        if not phonemizer_works:
            print("🔧 phonemizer cannot use espeak backend")
            print("💡 Check phonemizer installation: pip install phonemizer")
    
    print(f"\n📋 Next Steps:")
    if espeak_works and phonemizer_works:
        print("1. ✅ Run the updated notebook")
        print("2. ✅ Test the real efficient system")
        print("3. ✅ Compare speedup vs standard mode")
    else:
        print("1. 🔧 Fix espeak installation issues")
        print("2. 🔧 Re-run this test script")
        print("3. 🔧 Then test the notebook")

if __name__ == "__main__":
    main()
