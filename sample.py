"""
Enhanced Zonos Voice Cloning Sample

This sample demonstrates the improved voice cloning with better consistency,
timing, and naturalness. The enhanced system addresses common issues like
long pauses, speed variations, and gibberish generation.
"""

import torch
import torchaudio
import os
from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.utils import DEFAULT_DEVICE as device

# Try to use enhanced voice cloning if available
try:
    from enhanced_voice_cloning import create_enhanced_voice_cloner
    USE_ENHANCED = True
    print("🚀 Using Enhanced Voice Cloning")
except ImportError:
    USE_ENHANCED = False
    print("📢 Using Original Voice Cloning (Enhanced version not available)")

def main():
    audio_path = "assets/exampleaudio.mp3"

    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"❌ Audio file not found: {audio_path}")
        print("Please ensure the example audio file exists.")
        return

    if USE_ENHANCED:
        # Enhanced voice cloning approach
        print("🔧 Creating enhanced voice cloner...")
        cloner = create_enhanced_voice_cloner(device=device)

        print("🎤 Analyzing and cloning voice...")
        speaker_embedding, quality_metrics = cloner.clone_voice_from_audio(
            audio_path,
            target_length_seconds=15.0,
            normalize=True,
            remove_silence=True,
            analyze_quality=True
        )

        print(f"📊 Voice Quality Score: {quality_metrics['quality_score']:.3f}")
        print(f"📊 SNR Estimate: {quality_metrics['snr_estimate']:.1f} dB")

        print("🎵 Generating enhanced speech...")
        audio = cloner.generate_speech(
            text="Hello, world! This is an enhanced voice cloning demonstration with improved consistency and naturalness.",
            speaker_embedding=speaker_embedding,
            language="en-us",
            voice_quality=quality_metrics,
            seed=421  # For reproducible results
        )

        # Save enhanced output
        sample_rate = cloner.model.autoencoder.sampling_rate
        torchaudio.save("sample_enhanced.wav", audio, sample_rate)

        duration = audio.shape[-1] / sample_rate
        print(f"✅ Enhanced sample saved: sample_enhanced.wav ({duration:.2f}s)")

    else:
        # Original voice cloning approach
        print("📥 Loading model...")
        model = Zonos.from_pretrained("Wamp1re-Ai/Zonos-v0.1-transformer", device=device)

        print("🎤 Loading and processing audio...")
        wav, sampling_rate = torchaudio.load(audio_path)
        speaker = model.make_speaker_embedding(wav, sampling_rate)

        torch.manual_seed(421)

        print("🎵 Generating speech...")
        cond_dict = make_cond_dict(text="Hello, world!", speaker=speaker, language="en-us")
        conditioning = model.prepare_conditioning(cond_dict)

        codes = model.generate(conditioning)

        wavs = model.autoencoder.decode(codes).cpu()
        torchaudio.save("sample.wav", wavs[0], model.autoencoder.sampling_rate)

        duration = wavs[0].shape[-1] / model.autoencoder.sampling_rate
        print(f"✅ Original sample saved: sample.wav ({duration:.2f}s)")

    print("\n💡 For better results, try the enhanced voice cloning:")
    print("   python enhanced_sample.py")

if __name__ == "__main__":
    main()
