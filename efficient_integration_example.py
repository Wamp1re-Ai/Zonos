#!/usr/bin/env python3
"""
Integration example showing how to use the efficient voice cloning system
with the existing Zonos TTS infrastructure.

This demonstrates the practical usage of the efficiency improvements
while maintaining compatibility with existing code.
"""

import torch
import torchaudio
import time
from pathlib import Path
from typing import Optional

# Import existing Zonos components
from zonos import Zonos
from enhanced_voice_cloning import EnhancedVoiceCloner
from efficient_voice_cloning import EfficientVoiceCloner


class ZonosEfficientTTS:
    """
    Efficient TTS system that combines Zonos TTS with speed optimizations.
    
    This class provides a drop-in replacement for the standard Zonos TTS
    with significant performance improvements for long texts.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        use_fp16: bool = True,
        cache_size: int = 10,
        enable_efficient_mode: bool = True
    ):
        """
        Initialize the efficient TTS system.
        
        Args:
            model_path: Path to Zonos model (None for default)
            device: Device to use for inference
            use_fp16: Whether to use FP16 precision
            cache_size: Size of voice cache
            enable_efficient_mode: Whether to use efficiency optimizations
        """
        self.device = device
        self.enable_efficient_mode = enable_efficient_mode
        
        print(f"🚀 Initializing ZonosEfficientTTS")
        print(f"   Device: {device}")
        print(f"   FP16: {use_fp16}")
        print(f"   Efficient mode: {enable_efficient_mode}")
        
        # Load Zonos model
        print("📥 Loading Zonos TTS model...")
        self.zonos_model = Zonos(device=device)
        
        # Initialize enhanced voice cloner (original)
        self.enhanced_cloner = EnhancedVoiceCloner(self.zonos_model, device)
        
        # Initialize efficient voice cloner (optimized)
        if enable_efficient_mode:
            self.efficient_cloner = EfficientVoiceCloner(
                self.zonos_model, 
                device, 
                use_fp16=use_fp16, 
                cache_size=cache_size
            )
        else:
            self.efficient_cloner = None
        
        print("✅ ZonosEfficientTTS initialized successfully!")
    
    def clone_voice(
        self,
        audio_path: str,
        target_length_seconds: float = 20.0,
        use_cache: bool = True
    ) -> tuple:
        """
        Clone a voice from audio file with caching support.
        
        Args:
            audio_path: Path to reference audio file
            target_length_seconds: Target length for processing
            use_cache: Whether to use caching (efficient mode only)
            
        Returns:
            Tuple of (speaker_embedding, quality_metrics)
        """
        print(f"🎤 Cloning voice from: {audio_path}")
        
        # Load audio
        wav, sr = torchaudio.load(audio_path)
        wav = wav.to(self.device)
        
        if self.enable_efficient_mode and self.efficient_cloner:
            # Use efficient cloner with caching
            return self.efficient_cloner.clone_voice_from_audio(
                wav, sr, target_length_seconds, use_cache=use_cache
            )
        else:
            # Use standard enhanced cloner
            return self.enhanced_cloner.clone_voice_from_audio(
                wav, sr, target_length_seconds
            )
    
    def generate_speech(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        voice_quality: Optional[dict] = None,
        language: str = "en-us",
        cfg_scale: float = 2.0,
        seed: Optional[int] = None,
        use_efficient_mode: Optional[bool] = None,
        progress_callback: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Generate speech with automatic efficiency optimization.
        
        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding from clone_voice
            voice_quality: Voice quality metrics
            language: Language code
            cfg_scale: Classifier-free guidance scale
            seed: Random seed
            use_efficient_mode: Override efficient mode setting
            progress_callback: Progress callback function
            
        Returns:
            Generated audio tensor
        """
        # Determine which mode to use
        efficient_mode = (use_efficient_mode if use_efficient_mode is not None 
                         else self.enable_efficient_mode)
        
        # Automatically choose efficient mode for long texts
        if len(text) > 500 and self.efficient_cloner:
            efficient_mode = True
            print(f"📝 Long text detected ({len(text)} chars), using efficient mode")
        
        if efficient_mode and self.efficient_cloner:
            # Use efficient generation
            return self.efficient_cloner.generate_speech_fast(
                text=text,
                speaker_embedding=speaker_embedding,
                language=language,
                voice_quality=voice_quality,
                cfg_scale=cfg_scale,
                seed=seed,
                progress_callback=progress_callback
            )
        else:
            # Use standard generation
            return self.enhanced_cloner.generate_speech(
                text=text,
                speaker_embedding=speaker_embedding,
                language=language,
                voice_quality=voice_quality,
                cfg_scale=cfg_scale,
                seed=seed
            )
    
    def generate_speech_from_file(
        self,
        text_file: str,
        audio_file: str,
        output_file: str,
        **kwargs
    ) -> str:
        """
        Generate speech from text file using reference audio.
        
        Args:
            text_file: Path to text file
            audio_file: Path to reference audio file
            output_file: Path for output audio file
            **kwargs: Additional arguments for generation
            
        Returns:
            Path to generated audio file
        """
        print(f"📄 Processing text file: {text_file}")
        print(f"🎤 Using reference audio: {audio_file}")
        
        # Read text
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        print(f"📝 Text length: {len(text)} characters")
        
        # Clone voice
        speaker_embedding, voice_quality = self.clone_voice(audio_file)
        
        # Generate speech
        start_time = time.time()
        audio = self.generate_speech(
            text=text,
            speaker_embedding=speaker_embedding,
            voice_quality=voice_quality,
            **kwargs
        )
        generation_time = time.time() - start_time
        
        # Save audio
        torchaudio.save(output_file, audio.cpu(), self.zonos_model.autoencoder.sampling_rate)
        
        # Calculate metrics
        audio_duration = audio.shape[-1] / self.zonos_model.autoencoder.sampling_rate
        rtf = generation_time / audio_duration
        
        print(f"✅ Generation completed!")
        print(f"   Output: {output_file}")
        print(f"   Generation time: {generation_time:.2f}s")
        print(f"   Audio duration: {audio_duration:.2f}s")
        print(f"   RTF: {rtf:.4f}")
        print(f"   Speed: {1/rtf:.1f}x faster than real-time")
        
        return output_file
    
    def benchmark_performance(self, test_texts: list, reference_audio: str):
        """
        Benchmark performance comparison between modes.
        
        Args:
            test_texts: List of test texts
            reference_audio: Path to reference audio
            
        Returns:
            Dictionary with benchmark results
        """
        print("🏁 Starting performance benchmark...")
        
        # Clone voice once
        speaker_embedding, voice_quality = self.clone_voice(reference_audio)
        
        results = {}
        
        for i, text in enumerate(test_texts):
            print(f"\n🧪 Test {i+1}/{len(test_texts)} - {len(text)} characters")
            
            # Test standard mode
            if not self.enable_efficient_mode:
                print("⚡ Testing standard mode...")
                start_time = time.time()
                audio_standard = self.generate_speech(
                    text, speaker_embedding, voice_quality, use_efficient_mode=False
                )
                standard_time = time.time() - start_time
            else:
                standard_time = None
                audio_standard = None
            
            # Test efficient mode
            if self.efficient_cloner:
                print("🚀 Testing efficient mode...")
                start_time = time.time()
                audio_efficient = self.generate_speech(
                    text, speaker_embedding, voice_quality, use_efficient_mode=True
                )
                efficient_time = time.time() - start_time
            else:
                efficient_time = None
                audio_efficient = None
            
            # Calculate speedup
            speedup = (standard_time / efficient_time 
                      if standard_time and efficient_time else None)
            
            results[f"test_{i+1}"] = {
                'text_length': len(text),
                'standard_time': standard_time,
                'efficient_time': efficient_time,
                'speedup': speedup
            }
            
            if speedup:
                print(f"   📊 Speedup: {speedup:.1f}x")
        
        return results
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics."""
        stats = {'efficient_mode': self.enable_efficient_mode}
        
        if self.efficient_cloner:
            stats.update(self.efficient_cloner.get_stats())
        
        return stats
    
    def clear_cache(self):
        """Clear all caches."""
        if self.efficient_cloner:
            self.efficient_cloner.clear_cache()


def demo_usage():
    """Demonstrate usage of the efficient TTS system."""
    print("🎬 ZonosEfficientTTS Demo")
    print("=" * 50)
    
    # Initialize system
    tts = ZonosEfficientTTS(
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_fp16=True,
        enable_efficient_mode=True
    )
    
    # Example texts
    short_text = "Hello, this is a short test."
    long_text = """
    This is a longer text that will demonstrate the efficiency improvements.
    The system should automatically detect that this is a long text and use
    the efficient processing mode with batching and optimization. This should
    result in significantly faster generation times compared to the standard
    sequential processing approach.
    """
    
    # Note: You would need actual audio files for this to work
    # reference_audio = "path/to/reference/audio.wav"
    # 
    # # Clone voice
    # speaker_embedding, voice_quality = tts.clone_voice(reference_audio)
    # 
    # # Generate speech
    # audio = tts.generate_speech(long_text, speaker_embedding, voice_quality)
    # 
    # # Save result
    # torchaudio.save("output.wav", audio.cpu(), 24000)
    
    print("✅ Demo completed! (Note: Requires actual audio files to run)")
    print("\n📊 Performance stats:")
    stats = tts.get_performance_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    demo_usage()
