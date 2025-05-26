#!/usr/bin/env python3
"""
Unlimited Voice Cloning System - No Length Restrictions

This module implements a completely unlimited audio generation system that can
handle texts of any length without caps or restrictions. It uses dynamic
chunking, progressive generation, and intelligent memory management.

Key Features:
- NO LENGTH LIMITS - Generate audio of any duration
- Dynamic token calculation based on content complexity
- Progressive chunking for very long texts
- Intelligent sentence boundary detection
- Memory-efficient streaming generation
- Automatic quality preservation across chunks
"""

import time
import re
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List, Generator
from tqdm import tqdm
import warnings

from zonos.tts.utils import make_cond_dict


class UnlimitedVoiceCloner:
    """
    Unlimited voice cloning system with no length restrictions.
    
    This system can generate audio of any length by using:
    1. Dynamic token calculation without caps
    2. Progressive chunking for very long texts
    3. Intelligent sentence boundary detection
    4. Memory-efficient streaming generation
    5. Seamless audio concatenation
    """
    
    def __init__(self, model, device="cuda", use_fp16=True, cache_size=10):
        """
        Initialize the unlimited voice cloner.
        
        Args:
            model: Zonos TTS model
            device: Device to use for inference
            use_fp16: Whether to use FP16 precision for speed
            cache_size: Maximum number of cached reference audios
        """
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        
        # Caching system
        self.cache_size = cache_size
        self.audio_cache = {}
        self.cache_order = []
        
        # Performance tracking
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'total_generations': 0,
            'total_time': 0.0,
            'total_audio_duration': 0.0,
            'longest_generation': 0.0,
            'chunks_processed': 0
        }
        
        print(f"🚀 UnlimitedVoiceCloner initialized")
        print(f"   Device: {device}")
        print(f"   FP16: {use_fp16}")
        print(f"   NO LENGTH LIMITS!")
    
    def _get_audio_hash(self, wav: torch.Tensor, sr: int) -> str:
        """Generate a hash for audio caching."""
        mean_val = wav.mean().item()
        std_val = wav.std().item()
        length = wav.shape[-1]
        return f"{mean_val:.6f}_{std_val:.6f}_{length}_{sr}"
    
    def _manage_cache(self, audio_hash: str, data: Tuple):
        """Manage LRU cache for reference audios."""
        if audio_hash in self.audio_cache:
            self.cache_order.remove(audio_hash)
            self.cache_order.append(audio_hash)
            return
        
        if len(self.audio_cache) >= self.cache_size:
            oldest = self.cache_order.pop(0)
            del self.audio_cache[oldest]
        
        self.audio_cache[audio_hash] = data
        self.cache_order.append(audio_hash)
    
    def _intelligent_text_chunking(self, text: str, target_chunk_chars: int = 800) -> List[str]:
        """
        Intelligently chunk text at natural boundaries for unlimited generation.
        
        Args:
            text: Input text to chunk
            target_chunk_chars: Target characters per chunk (flexible)
            
        Returns:
            List of text chunks with natural boundaries
        """
        if len(text) <= target_chunk_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If paragraph is small enough, add to current chunk
            if len(current_chunk) + len(paragraph) + 2 <= target_chunk_chars:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                # Save current chunk if it exists
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # If paragraph is too long, split by sentences
                if len(paragraph) > target_chunk_chars:
                    sentences = re.split(r'[.!?]+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        
                        if len(temp_chunk) + len(sentence) + 1 <= target_chunk_chars:
                            if temp_chunk:
                                temp_chunk += ". " + sentence
                            else:
                                temp_chunk = sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk + ".")
                            temp_chunk = sentence
                    
                    if temp_chunk:
                        current_chunk = temp_chunk + "."
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _calculate_dynamic_tokens(self, text: str, base_tokens_per_char: int = 25) -> int:
        """
        Calculate tokens dynamically based on text complexity - NO CAPS!
        
        Args:
            text: Text to analyze
            base_tokens_per_char: Base tokens per character
            
        Returns:
            Number of tokens needed (unlimited)
        """
        # Base calculation
        base_tokens = len(text) * base_tokens_per_char
        
        # Complexity factors (no caps applied)
        complexity_multiplier = 1.0
        
        # More tokens for complex punctuation
        punctuation_count = len(re.findall(r'[.!?,:;]', text))
        if punctuation_count > 0:
            complexity_multiplier += punctuation_count / len(text) * 2
        
        # More tokens for numbers and special characters
        special_chars = len(re.findall(r'[0-9$%&@#]', text))
        if special_chars > 0:
            complexity_multiplier += special_chars / len(text) * 1.5
        
        # More tokens for mixed case (proper nouns, etc.)
        mixed_case = len(re.findall(r'[A-Z][a-z]', text))
        if mixed_case > 0:
            complexity_multiplier += mixed_case / len(text) * 1.2
        
        # Calculate final tokens (NO MAXIMUM CAP!)
        final_tokens = int(base_tokens * complexity_multiplier)
        
        # Only apply a reasonable minimum
        min_tokens = 500
        final_tokens = max(min_tokens, final_tokens)
        
        print(f"📊 Dynamic token calculation:")
        print(f"   Text length: {len(text)} chars")
        print(f"   Base tokens: {base_tokens}")
        print(f"   Complexity multiplier: {complexity_multiplier:.2f}")
        print(f"   Final tokens: {final_tokens} (NO CAP!)")
        
        return final_tokens
    
    def _generate_unlimited_chunk(
        self,
        text: str,
        speaker_embedding: torch.Tensor,
        language: str,
        voice_quality: Optional[Dict[str, Any]],
        cfg_scale: float
    ) -> torch.Tensor:
        """
        Generate audio for a text chunk with unlimited token calculation.
        """
        # Create conditioning dictionary
        cond_dict_params = {
            "text": text,
            "language": language,
            "speaker": speaker_embedding,
            "device": self.device
        }
        
        # Add voice quality parameters if available
        if voice_quality:
            if 'emotion_scores' in voice_quality:
                emotion_tensor = torch.tensor(voice_quality['emotion_scores'], device=self.device)
                cond_dict_params['emotion'] = emotion_tensor
            
            for param in ['speaking_rate', 'pitch_std', 'fmax', 'dnsmos_ovrl']:
                if param in voice_quality:
                    cond_dict_params[param] = voice_quality[param]
            
            if 'vq_score' in voice_quality:
                vq_tensor = torch.tensor([voice_quality['vq_score']] * 8, device=self.device).unsqueeze(0)
                cond_dict_params['vqscore_8'] = vq_tensor
        
        # Create conditioning dictionary
        cond_dict = make_cond_dict(**cond_dict_params)
        conditioning = self.model.prepare_conditioning(cond_dict)
        
        # Calculate tokens dynamically - NO CAPS!
        max_tokens = self._calculate_dynamic_tokens(text)
        
        print(f"🎵 Generating chunk: {len(text)} chars → {max_tokens} tokens")
        
        # Generate audio codes
        codes = self.model.generate(
            prefix_conditioning=conditioning,
            max_new_tokens=max_tokens,
            cfg_scale=cfg_scale,
            batch_size=1,
            sampling_params={"min_p": 0.1, "top_k": 0, "top_p": 0.0},
            progress_bar=True
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
        start_time = time.time()
        self.stats['total_generations'] += 1
        
        if seed is not None:
            torch.manual_seed(seed)
        
        print(f"🚀 UNLIMITED Speech Generation Started!")
        print(f"📝 Text length: {len(text)} characters")
        print(f"🎯 Target chunk size: {target_chunk_chars} characters")
        print(f"⚡ NO LENGTH LIMITS APPLIED!")
        
        # Intelligent text chunking
        chunks = self._intelligent_text_chunking(text, target_chunk_chars)
        total_chunks = len(chunks)
        
        print(f"🗂️ Split into {total_chunks} intelligent chunks")
        
        # Show chunk distribution
        chunk_sizes = [len(chunk) for chunk in chunks]
        print(f"📊 Chunk sizes: {chunk_sizes}")
        
        # Generate audio for each chunk
        all_audio_chunks = []
        
        for chunk_idx, chunk_text in enumerate(chunks):
            if progress_callback:
                progress = chunk_idx / total_chunks
                progress_callback(progress, f"Processing chunk {chunk_idx + 1}/{total_chunks}")
            
            print(f"\n🔄 Processing chunk {chunk_idx + 1}/{total_chunks}")
            print(f"📝 Chunk text: {chunk_text[:100]}{'...' if len(chunk_text) > 100 else ''}")
            
            # Generate audio for this chunk
            with torch.amp.autocast(self.device, enabled=self.use_fp16, dtype=self.dtype):
                chunk_audio = self._generate_unlimited_chunk(
                    chunk_text, speaker_embedding, language, voice_quality, cfg_scale
                )
            
            # Calculate chunk duration
            chunk_duration = chunk_audio.shape[-1] / self.model.autoencoder.sampling_rate
            print(f"🎵 Generated {chunk_duration:.1f}s of audio")
            
            all_audio_chunks.append(chunk_audio.cpu())
            self.stats['chunks_processed'] += 1
            
            # Clean up GPU memory
            self._torch_empty_cache()
        
        # Final concatenation with natural pauses
        if progress_callback:
            progress_callback(0.95, "Concatenating unlimited audio...")
        
        print(f"\n🔗 Concatenating {len(all_audio_chunks)} chunks...")
        
        if len(all_audio_chunks) > 1:
            # Add natural pauses between chunks
            pause_samples = int(0.3 * self.model.autoencoder.sampling_rate)  # 300ms pause
            pause = torch.zeros(1, pause_samples, dtype=self.dtype)
            
            final_chunks = []
            for i, chunk in enumerate(all_audio_chunks):
                final_chunks.append(chunk)
                if i < len(all_audio_chunks) - 1:
                    final_chunks.append(pause)
            
            final_audio = torch.cat(final_chunks, dim=1)
        else:
            final_audio = all_audio_chunks[0]
        
        # Final cleanup
        self._torch_empty_cache()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        self.stats['total_time'] += total_time
        
        audio_duration = final_audio.shape[-1] / self.model.autoencoder.sampling_rate
        self.stats['total_audio_duration'] += audio_duration
        self.stats['longest_generation'] = max(self.stats['longest_generation'], audio_duration)
        
        rtf = total_time / audio_duration
        
        print(f"\n✅ UNLIMITED Generation Completed!")
        print(f"   📝 Text: {len(text)} characters")
        print(f"   🗂️ Chunks: {total_chunks}")
        print(f"   ⏱️ Total time: {total_time:.2f}s")
        print(f"   🎵 Audio duration: {audio_duration:.2f}s ({audio_duration/60:.1f} minutes)")
        print(f"   📊 RTF: {rtf:.4f}")
        print(f"   🚀 Speed: {1/rtf:.1f}x faster than real-time")
        print(f"   🎯 Average chunk duration: {audio_duration/total_chunks:.1f}s")
        
        return final_audio
    
    def _torch_empty_cache(self):
        """Clear GPU memory cache."""
        try:
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            elif "mps" in str(self.device):
                torch.mps.empty_cache()
        except Exception:
            pass
    
    def get_unlimited_stats(self) -> Dict[str, Any]:
        """Get unlimited generation statistics."""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / max(total_requests, 1) * 100
        avg_audio_duration = self.stats['total_audio_duration'] / max(self.stats['total_generations'], 1)
        
        return {
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'total_generations': self.stats['total_generations'],
            'total_audio_duration': f"{self.stats['total_audio_duration']:.1f}s",
            'longest_generation': f"{self.stats['longest_generation']:.1f}s",
            'average_audio_duration': f"{avg_audio_duration:.1f}s",
            'chunks_processed': self.stats['chunks_processed'],
            'average_time': self.stats['total_time'] / max(self.stats['total_generations'], 1),
            'cache_size': len(self.audio_cache),
            **self.stats
        }


def create_unlimited_voice_cloner(model, device="cuda", **kwargs):
    """
    Factory function to create an unlimited voice cloner.
    
    Args:
        model: Zonos TTS model
        device: Device to use
        **kwargs: Additional arguments
        
    Returns:
        UnlimitedVoiceCloner instance
    """
    return UnlimitedVoiceCloner(model, device, **kwargs)
