#!/usr/bin/env python3
"""
Quick test script for Zonos in Google Colab
This script verifies that all dependencies are working correctly.
"""

import sys
import importlib
import subprocess

def check_system_deps():
    """Check if system dependencies are installed"""
    print("🔍 Checking system dependencies...")
    
    # Check eSpeak
    try:
        result = subprocess.run(['espeak', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ eSpeak-ng is installed")
        else:
            print("❌ eSpeak-ng not found")
            return False
    except FileNotFoundError:
        print("❌ eSpeak-ng not found")
        return False
    
    # Check git-lfs
    try:
        result = subprocess.run(['git', 'lfs', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git LFS is installed")
        else:
            print("❌ Git LFS not found")
            return False
    except FileNotFoundError:
        print("❌ Git LFS not found")
        return False
    
    return True

def check_python_deps():
    """Check if Python dependencies can be imported"""
    print("\n🐍 Checking Python dependencies...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('torchaudio', 'TorchAudio'),
        ('transformers', 'Transformers'),
        ('gradio', 'Gradio'),
        ('huggingface_hub', 'HuggingFace Hub'),
        ('soundfile', 'SoundFile'),
        ('phonemizer', 'Phonemizer'),
        ('numpy', 'NumPy'),
        ('scipy', 'SciPy')
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} not found")
            all_good = False
    
    return all_good

def check_zonos_import():
    """Check if Zonos modules can be imported"""
    print("\n🎵 Checking Zonos modules...")
    
    # Add current directory to path
    sys.path.insert(0, '/content/Zonos')
    
    zonos_modules = [
        ('zonos.model', 'Zonos Model'),
        ('zonos.conditioning', 'Conditioning'),
        ('zonos.utils', 'Utils'),
        ('zonos.autoencoder', 'Autoencoder'),
        ('zonos.sampling', 'Sampling')
    ]
    
    all_good = True
    for module, name in zonos_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            all_good = False
    
    return all_good

def check_gpu():
    """Check GPU availability"""
    print("\n🖥️ Checking GPU...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU Available: {gpu_name}")
            print(f"   Memory: {gpu_memory:.1f} GB")
            return True
        else:
            print("⚠️ No GPU available (will use CPU - slower)")
            return False
    except ImportError:
        print("❌ PyTorch not available")
        return False

def run_quick_test():
    """Run a quick functionality test"""
    print("\n🧪 Running quick functionality test...")
    
    try:
        from zonos.model import Zonos
        from zonos.conditioning import make_cond_dict
        import torch
        
        print("   Loading model (this may take a moment)...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Try to load model
        model = Zonos.from_pretrained("Wamp1re-Ai/Zonos-v0.1-transformer", device=device)
        print("   ✅ Model loaded successfully")
        
        # Test conditioning
        cond_dict = make_cond_dict(
            text="Test",
            language="en-us",
            device=device,
            unconditional_keys=["emotion", "speaker"]
        )
        conditioning = model.prepare_conditioning(cond_dict)
        print("   ✅ Conditioning created successfully")
        
        print("🎉 All tests passed! Zonos is ready to use.")
        return True
        
    except Exception as e:
        print(f"   ❌ Test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Zonos Colab Quick Test")
    print("=" * 40)
    
    success = True
    
    success &= check_system_deps()
    success &= check_python_deps()
    success &= check_zonos_import()
    check_gpu()  # GPU is optional
    
    if success:
        success &= run_quick_test()
    
    print("\n" + "=" * 40)
    if success:
        print("🎉 SUCCESS: Zonos is ready to use!")
        print("You can now run the main notebook cells for audio generation.")
    else:
        print("❌ FAILED: Some dependencies are missing or broken.")
        print("Please check the installation steps and try again.")
    
    return success

if __name__ == "__main__":
    main()
