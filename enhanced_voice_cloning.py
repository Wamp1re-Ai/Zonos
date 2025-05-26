"""
Enhanced Voice Cloning Module for Zonos TTS

This module provides improved voice cloning functionality with better audio preprocessing,
quality analysis, and optimized parameters for more consistent and natural speech generation.
"""

import torch
import torchaudio
from typing import Optional, Dict, Any, Tuple
import warnings

from zonos.model import Zonos
from zonos.conditioning import make_cond_dict
from zonos.speaker_cloning import (
    preprocess_audio_for_cloning, 
    analyze_voice_quality,
    get_voice_cloning_conditioning_params,
    get_voice_cloning_sampling_params
)
from zonos.utils import DEFAULT_DEVICE


class EnhancedVoiceCloner:
    """
    Enhanced voice cloning class with improved audio processing and parameter optimization.
    """
    
    def __init__(self, model: Zonos, device: str = DEFAULT_DEVICE):
        """
        Initialize the enhanced voice cloner.
        
        Args:
            model: Zonos TTS model instance
            device: Device to run on
        """
        self.model = model
        self.device = device
        
    def clone_voice_from_audio(
        self,
        audio_path_or_tensor: str | torch.Tensor,
        sample_rate: Optional[int] = None,
        target_length_seconds: float = 15.0,
        normalize: bool = True,
        remove_silence: bool = True,
        analyze_quality: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Create a speaker embedding from audio with enhanced preprocessing.
        
        Args:
            audio_path_or_tensor: Path to audio file or audio tensor
            sample_rate: Sample rate if providing tensor
            target_length_seconds: Target length for voice sample (optimal: 10-20 seconds)
            normalize: Whether to normalize audio amplitude
            remove_silence: Whether to remove leading/trailing silence
            analyze_quality: Whether to analyze voice quality
            
        Returns:
            Tuple of (speaker_embedding, quality_metrics)
        """
        # Load audio if path provided
        if isinstance(audio_path_or_tensor, str):
            wav, sr = torchaudio.load(audio_path_or_tensor)
        else:
            wav = audio_path_or_tensor
            sr = sample_rate
            if sr is None:
                raise ValueError("sample_rate must be provided when using tensor input")
        
        # Preprocess audio for better cloning
        processed_wav = preprocess_audio_for_cloning(
            wav, sr, 
            target_length_seconds=target_length_seconds,
            normalize=normalize,
            remove_silence=remove_silence
        )
        
        # Analyze voice quality
        quality_metrics = {}
        if analyze_quality:
            quality_metrics = analyze_voice_quality(processed_wav, sr)
            
            # Warn if quality is poor
            quality_score = quality_metrics.get('quality_score', 0.5)
            if quality_score < 0.3:
                warnings.warn(
                    f"Voice quality score is low ({quality_score:.2f}). "
                    "Consider using a cleaner audio sample for better results.",
                    UserWarning
                )
            elif quality_score < 0.5:
                warnings.warn(
                    f"Voice quality score is moderate ({quality_score:.2f}). "
                    "Results may vary. Consider using a higher quality audio sample.",
                    UserWarning
                )
        
        # Create speaker embedding
        speaker_embedding = self.model.make_speaker_embedding(processed_wav, sr)
        
        return speaker_embedding, quality_metrics
    
    def generate_speech(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        language: str = "en-us",
        voice_quality: Optional[Dict[str, Any]] = None,
        custom_conditioning_params: Optional[Dict[str, Any]] = None,
        custom_sampling_params: Optional[Dict[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
        cfg_scale: float = 2.0,
        progress_bar: bool = True,
        seed: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate speech with optimized parameters for voice cloning.
        
        Args:
            text: Text to synthesize
            speaker_embedding: Speaker embedding from clone_voice_from_audio
            language: Language code
            voice_quality: Voice quality metrics from clone_voice_from_audio
            custom_conditioning_params: Custom conditioning parameters (overrides defaults)
            custom_sampling_params: Custom sampling parameters (overrides defaults)
            max_new_tokens: Maximum tokens to generate
            cfg_scale: Classifier-free guidance scale
            progress_bar: Whether to show progress bar
            seed: Random seed for reproducibility
            
        Returns:
            Generated audio tensor
        """
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
        
        # Get optimized parameters based on voice quality
        conditioning_params = get_voice_cloning_conditioning_params(voice_quality)
        sampling_params = get_voice_cloning_sampling_params(voice_quality)
        
        # Override with custom parameters if provided
        if custom_conditioning_params:
            conditioning_params.update(custom_conditioning_params)
        if custom_sampling_params:
            sampling_params.update(custom_sampling_params)
        
        # Create conditioning dictionary
        cond_dict = make_cond_dict(
            text=text,
            language=language,
            speaker=speaker_embedding,
            **conditioning_params
        )
        
        # Prepare conditioning
        conditioning = self.model.prepare_conditioning(cond_dict)
        
        # Calculate max_new_tokens if not provided
        if max_new_tokens is None:
            # Estimate based on text length (roughly 86 tokens per second, 30 seconds max)
            estimated_duration = min(30, len(text) * 0.1)  # Rough estimate
            max_new_tokens = int(86 * estimated_duration)
        
        # Generate audio codes
        codes = self.model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=max_new_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params=sampling_params,
            progress_bar=progress_bar
        )
        
        # Decode to audio
        audio = self.model.autoencoder.decode(codes).cpu().detach()
        
        # Ensure mono output
        if audio.dim() == 2 and audio.size(0) > 1:
            audio = audio[0:1, :]
        
        return audio
    
    def clone_and_speak(
        self,
        text: str,
        audio_path_or_tensor: str | torch.Tensor,
        sample_rate: Optional[int] = None,
        language: str = "en-us",
        target_length_seconds: float = 15.0,
        seed: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete voice cloning pipeline: clone voice and generate speech in one call.
        
        Args:
            text: Text to synthesize
            audio_path_or_tensor: Path to audio file or audio tensor for voice cloning
            sample_rate: Sample rate if providing tensor
            language: Language code
            target_length_seconds: Target length for voice sample
            seed: Random seed for reproducibility
            **kwargs: Additional arguments for generate_speech
            
        Returns:
            Tuple of (generated_audio, voice_quality_metrics)
        """
        # Clone voice
        speaker_embedding, quality_metrics = self.clone_voice_from_audio(
            audio_path_or_tensor,
            sample_rate=sample_rate,
            target_length_seconds=target_length_seconds
        )
        
        # Generate speech
        audio = self.generate_speech(
            text=text,
            speaker_embedding=speaker_embedding,
            language=language,
            voice_quality=quality_metrics,
            seed=seed,
            **kwargs
        )
        
        return audio, quality_metrics


def create_enhanced_voice_cloner(model_name: str = "Wamp1re-Ai/Zonos-v0.1-transformer", 
                                device: str = DEFAULT_DEVICE) -> EnhancedVoiceCloner:
    """
    Create an enhanced voice cloner with a pre-loaded model.
    
    Args:
        model_name: Model name to load
        device: Device to run on
        
    Returns:
        EnhancedVoiceCloner instance
    """
    model = Zonos.from_pretrained(model_name, device=device)
    return EnhancedVoiceCloner(model, device)


# Convenience function for quick voice cloning
def quick_voice_clone(
    text: str,
    voice_audio_path: str,
    output_path: str = "cloned_speech.wav",
    language: str = "en-us",
    model_name: str = "Wamp1re-Ai/Zonos-v0.1-transformer",
    device: str = DEFAULT_DEVICE,
    seed: Optional[int] = None
) -> Dict[str, Any]:
    """
    Quick voice cloning function for simple use cases.
    
    Args:
        text: Text to synthesize
        voice_audio_path: Path to audio file for voice cloning
        output_path: Path to save generated audio
        language: Language code
        model_name: Model name to use
        device: Device to run on
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with generation info and quality metrics
    """
    # Create cloner
    cloner = create_enhanced_voice_cloner(model_name, device)
    
    # Clone and speak
    audio, quality_metrics = cloner.clone_and_speak(
        text=text,
        audio_path_or_tensor=voice_audio_path,
        language=language,
        seed=seed
    )
    
    # Save audio
    sample_rate = cloner.model.autoencoder.sampling_rate
    torchaudio.save(output_path, audio, sample_rate)
    
    return {
        'output_path': output_path,
        'sample_rate': sample_rate,
        'duration': audio.shape[-1] / sample_rate,
        'quality_metrics': quality_metrics
    }
