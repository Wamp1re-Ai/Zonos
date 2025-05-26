"""
Test script for enhanced voice cloning functionality.
This script tests the improvements and compares with original implementation.
"""

import torch
import torchaudio
import os
import time
from typing import Dict, Any

# Import both original and enhanced implementations
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from enhanced_voice_cloning import EnhancedVoiceCloner, create_enhanced_voice_cloner
from zonos.utils import DEFAULT_DEVICE as device


def test_original_implementation(model: Zonos, audio_path: str, text: str) -> Dict[str, Any]:
    """Test the original voice cloning implementation."""
    print("🔄 Testing original implementation...")
    
    start_time = time.time()
    
    # Load and process audio (original way)
    wav, sampling_rate = torchaudio.load(audio_path)
    speaker = model.make_speaker_embedding(wav, sampling_rate)
    
    # Use default parameters
    torch.manual_seed(42)
    cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond_dict)
    
    # Generate with default sampling
    codes = model.generate(conditioning)
    wavs = model.autoencoder.decode(codes).cpu()
    
    generation_time = time.time() - start_time
    
    # Save output
    output_path = "test_original.wav"
    torchaudio.save(output_path, wavs[0], model.autoencoder.sampling_rate)
    
    duration = wavs[0].shape[-1] / model.autoencoder.sampling_rate
    
    return {
        'method': 'Original',
        'output_path': output_path,
        'duration': duration,
        'generation_time': generation_time,
        'audio_tensor': wavs[0]
    }


def test_enhanced_implementation(audio_path: str, text: str) -> Dict[str, Any]:
    """Test the enhanced voice cloning implementation."""
    print("🚀 Testing enhanced implementation...")
    
    start_time = time.time()
    
    # Create enhanced cloner
    cloner = create_enhanced_voice_cloner(device=device)
    
    # Clone voice with enhanced preprocessing
    speaker_embedding, quality_metrics = cloner.clone_voice_from_audio(
        audio_path,
        target_length_seconds=15.0,
        normalize=True,
        remove_silence=True,
        analyze_quality=True
    )
    
    # Generate with optimized parameters
    audio = cloner.generate_speech(
        text=text,
        speaker_embedding=speaker_embedding,
        language="en-us",
        voice_quality=quality_metrics,
        seed=42  # Same seed for fair comparison
    )
    
    generation_time = time.time() - start_time
    
    # Save output
    output_path = "test_enhanced.wav"
    sample_rate = cloner.model.autoencoder.sampling_rate
    torchaudio.save(output_path, audio, sample_rate)
    
    duration = audio.shape[-1] / sample_rate
    
    return {
        'method': 'Enhanced',
        'output_path': output_path,
        'duration': duration,
        'generation_time': generation_time,
        'quality_metrics': quality_metrics,
        'audio_tensor': audio
    }


def analyze_audio_consistency(audio_tensor: torch.Tensor, sample_rate: int) -> Dict[str, float]:
    """Analyze audio for consistency metrics."""
    if audio_tensor.dim() > 1:
        audio_tensor = audio_tensor.squeeze()
    
    # Calculate RMS energy over time windows
    window_size = sample_rate // 10  # 100ms windows
    num_windows = len(audio_tensor) // window_size
    
    energies = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        window_energy = torch.sqrt(torch.mean(audio_tensor[start:end] ** 2))
        energies.append(window_energy.item())
    
    if len(energies) < 2:
        return {'energy_variance': 0.0, 'consistency_score': 1.0}
    
    # Calculate variance in energy (lower is more consistent)
    energy_variance = torch.var(torch.tensor(energies)).item()
    
    # Calculate consistency score (0-1, higher is better)
    max_energy = max(energies)
    min_energy = min(energies)
    if max_energy > 0:
        consistency_score = 1.0 - (max_energy - min_energy) / max_energy
    else:
        consistency_score = 1.0
    
    return {
        'energy_variance': energy_variance,
        'consistency_score': consistency_score,
        'avg_energy': sum(energies) / len(energies),
        'energy_range': max_energy - min_energy
    }


def main():
    print("🧪 Enhanced Voice Cloning Test Suite")
    print("=" * 50)
    
    # Check for example audio
    audio_path = "assets/exampleaudio.mp3"
    if not os.path.exists(audio_path):
        print(f"❌ Test audio not found at {audio_path}")
        print("Please ensure the example audio file exists.")
        return
    
    # Test text
    test_text = "This is a comprehensive test of voice cloning quality and consistency. We are evaluating timing, naturalness, and overall speech quality."
    
    print(f"📝 Test text: {test_text}")
    print(f"🎵 Voice sample: {audio_path}")
    print()
    
    try:
        # Load model once for both tests
        print("📥 Loading Zonos model...")
        model = Zonos.from_pretrained("Wamp1re-Ai/Zonos-v0.1-transformer", device=device)
        print("✅ Model loaded successfully")
        print()
        
        # Test original implementation
        original_result = test_original_implementation(model, audio_path, test_text)
        
        # Test enhanced implementation  
        enhanced_result = test_enhanced_implementation(audio_path, test_text)
        
        print("\n📊 Results Comparison")
        print("-" * 30)
        
        # Compare basic metrics
        print(f"{'Metric':<20} {'Original':<15} {'Enhanced':<15} {'Improvement'}")
        print("-" * 65)
        
        # Duration comparison
        duration_diff = enhanced_result['duration'] - original_result['duration']
        duration_pct = (duration_diff / original_result['duration']) * 100 if original_result['duration'] > 0 else 0
        print(f"{'Duration (s)':<20} {original_result['duration']:<15.2f} {enhanced_result['duration']:<15.2f} {duration_pct:+.1f}%")
        
        # Generation time comparison
        time_diff = enhanced_result['generation_time'] - original_result['generation_time']
        time_pct = (time_diff / original_result['generation_time']) * 100 if original_result['generation_time'] > 0 else 0
        print(f"{'Gen Time (s)':<20} {original_result['generation_time']:<15.2f} {enhanced_result['generation_time']:<15.2f} {time_pct:+.1f}%")
        
        # Analyze audio consistency
        print("\n🔍 Audio Consistency Analysis")
        print("-" * 35)
        
        original_consistency = analyze_audio_consistency(
            original_result['audio_tensor'], 
            model.autoencoder.sampling_rate
        )
        
        enhanced_consistency = analyze_audio_consistency(
            enhanced_result['audio_tensor'], 
            model.autoencoder.sampling_rate
        )
        
        print(f"{'Metric':<20} {'Original':<15} {'Enhanced':<15} {'Improvement'}")
        print("-" * 65)
        
        # Consistency score (higher is better)
        consistency_improvement = ((enhanced_consistency['consistency_score'] - original_consistency['consistency_score']) / original_consistency['consistency_score']) * 100
        print(f"{'Consistency':<20} {original_consistency['consistency_score']:<15.3f} {enhanced_consistency['consistency_score']:<15.3f} {consistency_improvement:+.1f}%")
        
        # Energy variance (lower is better)
        variance_improvement = ((original_consistency['energy_variance'] - enhanced_consistency['energy_variance']) / original_consistency['energy_variance']) * 100 if original_consistency['energy_variance'] > 0 else 0
        print(f"{'Energy Variance':<20} {original_consistency['energy_variance']:<15.6f} {enhanced_consistency['energy_variance']:<15.6f} {variance_improvement:+.1f}%")
        
        # Voice quality metrics (enhanced only)
        if 'quality_metrics' in enhanced_result:
            quality = enhanced_result['quality_metrics']
            print(f"\n📈 Enhanced Voice Quality Metrics")
            print("-" * 35)
            print(f"Quality Score: {quality['quality_score']:.3f}")
            print(f"SNR Estimate: {quality['snr_estimate']:.1f} dB")
            print(f"Dynamic Range: {quality['dynamic_range']:.1f} dB")
            print(f"Duration: {quality['duration']:.2f} seconds")
        
        print(f"\n📁 Generated Files:")
        print(f"   🔊 Original: {original_result['output_path']}")
        print(f"   🚀 Enhanced: {enhanced_result['output_path']}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"💡 Listen to both files to compare quality and consistency.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
