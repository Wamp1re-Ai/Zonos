"""
Enhanced Voice Cloning Sample Script

This script demonstrates the improved voice cloning functionality with better
audio preprocessing, quality analysis, and optimized parameters.
"""

import torch
import torchaudio
import os
from enhanced_voice_cloning import EnhancedVoiceCloner, create_enhanced_voice_cloner, quick_voice_clone
from zonos.utils import DEFAULT_DEVICE as device

def main():
    print("🎤 Enhanced Zonos Voice Cloning Demo")
    print("=" * 50)
    
    # Check if example audio exists
    example_audio_path = "assets/exampleaudio.mp3"
    if not os.path.exists(example_audio_path):
        print(f"❌ Example audio not found at {example_audio_path}")
        print("Please ensure the example audio file exists or provide your own audio file.")
        return
    
    # Method 1: Quick voice cloning (simplest approach)
    print("\n🚀 Method 1: Quick Voice Cloning")
    print("-" * 30)
    
    try:
        result = quick_voice_clone(
            text="Hello! This is an enhanced voice cloning demonstration using Zonos TTS. The new system provides much better consistency and naturalness.",
            voice_audio_path=example_audio_path,
            output_path="enhanced_sample_quick.wav",
            language="en-us",
            seed=42  # For reproducible results
        )
        
        print(f"✅ Quick cloning completed!")
        print(f"   Output: {result['output_path']}")
        print(f"   Duration: {result['duration']:.2f} seconds")
        print(f"   Quality Score: {result['quality_metrics'].get('quality_score', 'N/A'):.3f}")
        print(f"   SNR Estimate: {result['quality_metrics'].get('snr_estimate', 'N/A'):.1f} dB")
        
    except Exception as e:
        print(f"❌ Quick cloning failed: {e}")
    
    # Method 2: Advanced voice cloning with full control
    print("\n🔧 Method 2: Advanced Voice Cloning")
    print("-" * 35)
    
    try:
        # Create enhanced voice cloner
        print("Loading model...")
        cloner = create_enhanced_voice_cloner(device=device)
        
        # Clone voice with analysis
        print("Analyzing and cloning voice...")
        speaker_embedding, quality_metrics = cloner.clone_voice_from_audio(
            example_audio_path,
            target_length_seconds=12.0,  # Use 12 seconds for optimal quality
            normalize=True,
            remove_silence=True,
            analyze_quality=True
        )
        
        print(f"📊 Voice Quality Analysis:")
        print(f"   Duration: {quality_metrics['duration']:.2f} seconds")
        print(f"   Quality Score: {quality_metrics['quality_score']:.3f}")
        print(f"   SNR Estimate: {quality_metrics['snr_estimate']:.1f} dB")
        print(f"   Dynamic Range: {quality_metrics['dynamic_range']:.1f} dB")
        print(f"   RMS Energy: {quality_metrics['rms_energy']:.4f}")
        
        # Generate multiple samples with different settings
        test_texts = [
            "This is a test of the enhanced voice cloning system with improved consistency.",
            "The new preprocessing pipeline removes silence and normalizes audio for better results.",
            "Advanced parameter optimization ensures natural speech patterns and timing."
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n🎵 Generating sample {i+1}/3...")
            
            # Generate with optimized parameters
            audio = cloner.generate_speech(
                text=text,
                speaker_embedding=speaker_embedding,
                language="en-us",
                voice_quality=quality_metrics,
                seed=42 + i  # Different seed for each sample
            )
            
            # Save audio
            output_path = f"enhanced_sample_advanced_{i+1}.wav"
            sample_rate = cloner.model.autoencoder.sampling_rate
            torchaudio.save(output_path, audio, sample_rate)
            
            duration = audio.shape[-1] / sample_rate
            print(f"   ✅ Saved: {output_path} ({duration:.2f}s)")
        
        print(f"\n🎉 Advanced cloning completed successfully!")
        
    except Exception as e:
        print(f"❌ Advanced cloning failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Method 3: Custom parameter demonstration
    print("\n⚙️  Method 3: Custom Parameter Tuning")
    print("-" * 35)
    
    try:
        # Load model if not already loaded
        if 'cloner' not in locals():
            cloner = create_enhanced_voice_cloner(device=device)
            speaker_embedding, quality_metrics = cloner.clone_voice_from_audio(example_audio_path)
        
        # Custom conditioning parameters for different styles
        custom_params = {
            'conservative': {
                'conditioning': {'pitch_std': 10.0, 'speaking_rate': 10.0},
                'sampling': {'min_p': 0.03, 'temperature': 0.7}
            },
            'expressive': {
                'conditioning': {'pitch_std': 25.0, 'speaking_rate': 16.0},
                'sampling': {'min_p': 0.08, 'temperature': 0.9}
            },
            'neutral': {
                'conditioning': {'pitch_std': 15.0, 'speaking_rate': 12.0},
                'sampling': {'min_p': 0.05, 'temperature': 0.8}
            }
        }
        
        test_text = "This demonstrates different parameter settings for voice cloning customization."
        
        for style_name, params in custom_params.items():
            print(f"\n🎭 Generating {style_name} style...")
            
            audio = cloner.generate_speech(
                text=test_text,
                speaker_embedding=speaker_embedding,
                language="en-us",
                voice_quality=quality_metrics,
                custom_conditioning_params=params['conditioning'],
                custom_sampling_params=params['sampling'],
                seed=123
            )
            
            output_path = f"enhanced_sample_{style_name}.wav"
            sample_rate = cloner.model.autoencoder.sampling_rate
            torchaudio.save(output_path, audio, sample_rate)
            
            duration = audio.shape[-1] / sample_rate
            print(f"   ✅ Saved: {output_path} ({duration:.2f}s)")
        
        print(f"\n🎨 Custom parameter demonstration completed!")
        
    except Exception as e:
        print(f"❌ Custom parameter demo failed: {e}")
    
    print("\n" + "=" * 50)
    print("🏁 Enhanced Voice Cloning Demo Complete!")
    print("\nGenerated files:")
    for filename in os.listdir("."):
        if filename.startswith("enhanced_sample") and filename.endswith(".wav"):
            print(f"   📁 {filename}")
    
    print("\n💡 Tips for better voice cloning:")
    print("   • Use clean, high-quality audio (16kHz+ sample rate)")
    print("   • Provide 10-20 seconds of speech for optimal results")
    print("   • Avoid background noise and music")
    print("   • Use consistent speaking style in the reference audio")
    print("   • Experiment with different parameter settings for your use case")


if __name__ == "__main__":
    main()
