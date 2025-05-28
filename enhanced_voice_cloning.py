"""
Enhanced Voice Cloning Module for Zonos TTS

This module provides improved voice cloning functionality with better audio preprocessing,
quality analysis, and optimized parameters for more consistent and natural speech generation.
"""

import torch
import torchaudio
from typing import Optional, Dict, Any, Tuple, List
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

    def _split_text_into_chunks(self, text: str, max_chunk_length: int = 200) -> List[str]:
        """
        Split text into chunks for better long-text generation.

        Args:
            text: Text to split
            max_chunk_length: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        import re

        # First try to split by sentences
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence would exceed the limit, start a new chunk
            if len(current_chunk) + len(sentence) + 1 > max_chunk_length and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If we still have chunks that are too long, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) <= max_chunk_length:
                final_chunks.append(chunk)
            else:
                # Split by words if sentence splitting wasn't enough
                words = chunk.split()
                current_word_chunk = ""
                for word in words:
                    if len(current_word_chunk) + len(word) + 1 > max_chunk_length and current_word_chunk:
                        final_chunks.append(current_word_chunk.strip())
                        current_word_chunk = word
                    else:
                        if current_word_chunk:
                            current_word_chunk += " " + word
                        else:
                            current_word_chunk = word
                if current_word_chunk.strip():
                    final_chunks.append(current_word_chunk.strip())

        return final_chunks

    def _generate_chunked_speech(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        language: str = "en-us",
        voice_quality: Optional[Dict[str, Any]] = None,
        max_new_tokens: Optional[int] = None,
        cfg_scale: float = 2.0,
        sampling_params: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        progress_bar: bool = True,
        max_chunk_length: int = 200
    ) -> torch.Tensor:
        """
        Generate speech for long text by splitting into chunks and concatenating.

        Args:
            text: Long text to synthesize
            speaker_embedding: Speaker embedding from voice cloning
            language: Language code
            voice_quality: Voice quality metrics for conditioning
            max_new_tokens: Maximum tokens to generate per chunk
            cfg_scale: Classifier-free guidance scale
            sampling_params: Sampling parameters for generation
            seed: Random seed for reproducibility
            progress_bar: Whether to show progress bar
            max_chunk_length: Maximum characters per chunk

        Returns:
            Concatenated audio tensor
        """
        print(f"🔄 Processing long text ({len(text)} chars) in chunks...")

        # Split text into manageable chunks
        chunks = self._split_text_into_chunks(text, max_chunk_length)
        print(f"📝 Split into {len(chunks)} chunks")

        audio_chunks = []
        sample_rate = None

        for i, chunk in enumerate(chunks):
            if progress_bar:
                print(f"🎵 Generating chunk {i+1}/{len(chunks)}: '{chunk[:50]}{'...' if len(chunk) > 50 else ''}'")

            # Generate audio for this chunk (disable chunking to avoid recursion)
            chunk_audio = self.generate_speech(
                text=chunk,
                speaker_embedding=speaker_embedding,
                language=language,
                voice_quality=voice_quality,
                max_new_tokens=max_new_tokens,
                cfg_scale=cfg_scale,
                custom_sampling_params=sampling_params,
                seed=seed,
                progress_bar=False,  # Disable progress bar for individual chunks
                chunk_long_text=False  # Disable chunking to avoid recursion
            )

            audio_chunks.append(chunk_audio)

            if sample_rate is None:
                sample_rate = self.model.autoencoder.sampling_rate

        # Concatenate all audio chunks
        if len(audio_chunks) == 1:
            final_audio = audio_chunks[0]
        else:
            print("🔗 Concatenating audio chunks...")
            # Add small silence between chunks (100ms)
            silence_samples = int(0.1 * sample_rate)  # 100ms of silence
            silence = torch.zeros(1, silence_samples)

            concatenated_chunks = []
            for i, chunk in enumerate(audio_chunks):
                concatenated_chunks.append(chunk)
                # Add silence between chunks (but not after the last one)
                if i < len(audio_chunks) - 1:
                    concatenated_chunks.append(silence)

            final_audio = torch.cat(concatenated_chunks, dim=1)

        print(f"✅ Long text generation completed! Total duration: {final_audio.shape[1] / sample_rate:.2f} seconds")
        return final_audio

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
        seed: Optional[int] = None,
        chunk_long_text: bool = True,
        max_chunk_length: int = 200
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
            chunk_long_text: Whether to chunk long text for better generation
            max_chunk_length: Maximum characters per chunk

        Returns:
            Generated audio tensor
        """
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)

        # Handle long text by chunking if enabled
        if chunk_long_text and len(text) > max_chunk_length:
            return self._generate_chunked_speech(
                text, speaker_embedding, language, voice_quality,
                max_new_tokens, cfg_scale, custom_sampling_params, seed,
                progress_bar, max_chunk_length
            )

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
            # UNLIMITED token estimation - NO CAPS for unlimited generation!
            # Use approximately 20-25 tokens per character for better estimation
            tokens_per_char = 25  # Higher for better quality
            estimated_tokens = len(text) * tokens_per_char

            # Set reasonable minimum only - NO MAXIMUM CAP!
            min_tokens = 1000
            max_tokens = max(min_tokens, estimated_tokens)  # NO CAP - unlimited length!
            max_new_tokens = max_tokens

            print(f"🚀 UNLIMITED generation: {len(text)} chars → {max_new_tokens} tokens (NO CAP!)")

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

    def generate_unlimited_speech(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        language: str = "en-us",
        voice_quality: Optional[Dict[str, Any]] = None,
        target_chunk_chars: int = 800,
        cfg_scale: float = 2.0,
        seed: Optional[int] = None,
        progress_callback: Optional[callable] = None
    ) -> torch.Tensor:
        """
        Generate speech of unlimited length with no restrictions.

        This method can handle texts of ANY length by using intelligent chunking
        and unlimited token calculation.

        Args:
            text: Text to synthesize (ANY LENGTH!)
            speaker_embedding: Speaker embedding
            language: Language code
            voice_quality: Voice quality metrics
            target_chunk_chars: Target characters per chunk (flexible)
            cfg_scale: Classifier-free guidance scale
            seed: Random seed
            progress_callback: Progress callback function

        Returns:
            Generated audio tensor (unlimited length)
        """
        if seed is not None:
            torch.manual_seed(seed)

        print(f"🚀 UNLIMITED Speech Generation Started!")
        print(f"📝 Text length: {len(text)} characters")
        print(f"🎯 Target chunk size: {target_chunk_chars} characters")
        print(f"⚡ NO LENGTH LIMITS APPLIED!")

        # For very long texts, use intelligent chunking
        if len(text) > target_chunk_chars:
            print(f"📊 Long text detected, using intelligent chunking...")
            return self._generate_chunked_unlimited(
                text, speaker_embedding, language, voice_quality,
                target_chunk_chars, cfg_scale, seed, progress_callback
            )
        else:
            # For shorter texts, generate directly with unlimited tokens
            print(f"📊 Generating directly with unlimited tokens...")
            return self.generate_speech(
                text=text,
                speaker_embedding=speaker_embedding,
                language=language,
                voice_quality=voice_quality,
                cfg_scale=cfg_scale,
                seed=seed,
                chunk_long_text=False  # Disable chunking to use unlimited tokens
            )

    def _generate_chunked_unlimited(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        language: str,
        voice_quality: Optional[Dict[str, Any]],
        target_chunk_chars: int,
        cfg_scale: float,
        seed: Optional[int],
        progress_callback: Optional[callable]
    ) -> torch.Tensor:
        """Generate unlimited audio using intelligent chunking."""

        # Intelligent text chunking at natural boundaries
        chunks = self._intelligent_text_chunking(text, target_chunk_chars)
        total_chunks = len(chunks)

        print(f"🗂️ Split into {total_chunks} intelligent chunks")
        chunk_sizes = [len(chunk) for chunk in chunks]
        print(f"📊 Chunk sizes: {chunk_sizes}")

        # Generate audio for each chunk
        all_audio_chunks = []

        for i, chunk in enumerate(chunks):
            if progress_callback:
                progress_callback(i, total_chunks, f"Generating chunk {i+1}/{total_chunks}")

            print(f"🎵 Generating unlimited chunk {i+1}/{total_chunks}: {len(chunk)} chars")

            # Generate with unlimited tokens (no chunking)
            chunk_audio = self.generate_speech(
                text=chunk,
                speaker_embedding=speaker_embedding,
                language=language,
                voice_quality=voice_quality,
                cfg_scale=cfg_scale,
                seed=seed,
                chunk_long_text=False,  # Disable chunking for unlimited generation
                progress_bar=False
            )

            all_audio_chunks.append(chunk_audio)

        print(f"\n🔗 Concatenating {len(all_audio_chunks)} unlimited chunks...")

        if len(all_audio_chunks) > 1:
            # Add natural pauses between chunks (300ms)
            sample_rate = self.model.autoencoder.sampling_rate
            pause_samples = int(0.3 * sample_rate)
            pause = torch.zeros(1, pause_samples)

            final_chunks = []
            for i, chunk in enumerate(all_audio_chunks):
                final_chunks.append(chunk)
                if i < len(all_audio_chunks) - 1:
                    final_chunks.append(pause)

            final_audio = torch.cat(final_chunks, dim=1)
        else:
            final_audio = all_audio_chunks[0]

        return final_audio

    def _intelligent_text_chunking(self, text: str, target_chunk_chars: int) -> List[str]:
        """Intelligently chunk text at natural boundaries."""
        if len(text) <= target_chunk_chars:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Calculate end position for this chunk
            end_pos = min(current_pos + target_chunk_chars, len(text))

            # If we're not at the end, try to find a natural break point
            if end_pos < len(text):
                # Look for sentence endings first
                for i in range(end_pos, max(current_pos + target_chunk_chars // 2, current_pos + 1), -1):
                    if text[i-1] in '.!?':
                        end_pos = i
                        break
                else:
                    # Look for other punctuation
                    for i in range(end_pos, max(current_pos + target_chunk_chars // 2, current_pos + 1), -1):
                        if text[i-1] in ',;:':
                            end_pos = i
                            break
                    else:
                        # Look for word boundaries
                        for i in range(end_pos, max(current_pos + target_chunk_chars // 2, current_pos + 1), -1):
                            if text[i-1] == ' ':
                                end_pos = i
                                break

            # Extract chunk and clean it
            chunk = text[current_pos:end_pos].strip()
            if chunk:
                chunks.append(chunk)

            current_pos = end_pos

        return chunks

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


def create_enhanced_voice_cloner(model_name: str = "Zyphra/Zonos-v0.1-transformer",
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
    model_name: str = "Zyphra/Zonos-v0.1-transformer",
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
