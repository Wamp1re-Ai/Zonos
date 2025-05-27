import nltk
import torch
import torchaudio 
import time

# --- Text Segmentation Function ---
def setup_nltk_punkt():
    """
    Downloads the NLTK 'punkt' tokenizer models if not already present.
    """
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)

def segment_text_into_chunks(text: str, max_chars_per_chunk: int = 600, max_words_per_chunk: int = 100) -> list[str]:
    """
    Segments a given text into chunks, prioritizing sentence boundaries.
    """
    setup_nltk_punkt() 
    sentences = nltk.sent_tokenize(text)
    if not sentences: return []
    
    chunks = []
    current_chunk_text = ""
    current_chunk_char_count = 0
    current_chunk_word_count = 0

    for sentence in sentences:
        sentence_char_count = len(sentence)
        sentence_word_count = len(sentence.split()) 

        if not current_chunk_text and \
           (sentence_char_count > max_chars_per_chunk or sentence_word_count > max_words_per_chunk):
            chunks.append(sentence)
            continue

        space_char_needed = 1 if current_chunk_text else 0
        if current_chunk_text and \
           ((current_chunk_char_count + sentence_char_count + space_char_needed > max_chars_per_chunk) or \
            (current_chunk_word_count + sentence_word_count > max_words_per_chunk)):
            chunks.append(current_chunk_text)
            current_chunk_text = sentence
            current_chunk_char_count = sentence_char_count
            current_chunk_word_count = sentence_word_count
        else:
            if not current_chunk_text:
                current_chunk_text = sentence
                current_chunk_char_count = sentence_char_count
                current_chunk_word_count = sentence_word_count
            else:
                current_chunk_text += " " + sentence
                current_chunk_char_count += (sentence_char_count + space_char_needed) 
                current_chunk_word_count += sentence_word_count
    
    if current_chunk_text: chunks.append(current_chunk_text)
    return chunks

# --- Long Audio Generation Function ---
def generate_long_audio_from_text(
    full_text: str, 
    model,  # zonos.model.Zonos instance
    device: torch.device, 
    speaker_embedding: torch.Tensor, 
    language: str, 
    cfg_scale: float,
    custom_conditioning_params: dict, 
    custom_sampling_params: dict,   
    seed: int,
    make_cond_dict_func, 
    max_chars_per_chunk: int = 500, 
    max_words_per_chunk: int = 80,  
    prefix_duration_seconds: float = 1.5, 
    max_new_tokens_per_chunk: int = 86 * 30,
    crossfade_duration_ms: int = 0 # New parameter for cross-fading
) -> tuple[torch.Tensor | None, int]:
    """
    Generates long audio by segmenting text, generating chunks with audio prefixing,
    and concatenating the results, with an optional cross-fade.
    """
    print(f"🎙️ Starting long audio generation for text ({len(full_text)} chars)...")
    if crossfade_duration_ms > 0:
        print(f"  Applying cross-fade of {crossfade_duration_ms}ms between chunks.")
    
    text_chunks = segment_text_into_chunks(full_text, max_chars_per_chunk, max_words_per_chunk)
    if not text_chunks:
        print("No text chunks to process.")
        return None, model.autoencoder.sampling_rate

    torch.manual_seed(seed) 
    
    # Store all generated audio parts (full, original waveforms) on CPU
    generated_audio_chunks_cpu = [] 
    previous_chunk_audio_for_prefix_cpu = None # For prefixing, original CPU tensor
    sample_rate = model.autoencoder.sampling_rate

    for i, chunk_text in enumerate(text_chunks):
        print(f"\nGenerating chunk {i+1}/{len(text_chunks)}: \"{chunk_text[:80]}...\"")
        
        audio_prefix_codes = None
        if previous_chunk_audio_for_prefix_cpu is not None and previous_chunk_audio_for_prefix_cpu.numel() > 0:
            prefix_num_samples = int(prefix_duration_seconds * sample_rate)
            
            current_prefix_audio = previous_chunk_audio_for_prefix_cpu
            if current_prefix_audio.dim() == 2 and current_prefix_audio.shape[0] == 1: 
                current_prefix_audio = current_prefix_audio.squeeze(0)

            if current_prefix_audio.shape[-1] > prefix_num_samples:
                audio_prefix_snippet = current_prefix_audio[-prefix_num_samples:]
            else:
                audio_prefix_snippet = current_prefix_audio
            
            print(f"  Using audio prefix of {audio_prefix_snippet.shape[-1]/sample_rate:.2f}s from previous chunk.")
            audio_prefix_snippet_for_encode = audio_prefix_snippet.to(device).unsqueeze(0).unsqueeze(0)

            if audio_prefix_snippet_for_encode.numel() > 0 and audio_prefix_snippet_for_encode.shape[-1] > 0 :
                try:
                    with torch.no_grad():
                         audio_prefix_codes_encoded = model.autoencoder.encode(audio_prefix_snippet_for_encode)
                    if audio_prefix_codes_encoded.numel() > 0:
                        audio_prefix_codes = audio_prefix_codes_encoded 
                        print(f"    Encoded audio prefix: {audio_prefix_codes.shape}") 
                    else:
                        print("    Warning: Encoding audio prefix resulted in empty tensor. Skipping prefix.")
                except Exception as e:
                    print(f"    Error encoding audio prefix: {e}. Skipping prefix for this chunk.")
            else:
                print("    Warning: Audio prefix snippet is empty or invalid. Skipping prefix.")
        
        cond_dict = make_cond_dict_func(
            text=chunk_text, language=language, speaker=speaker_embedding,
            device=device, **custom_conditioning_params 
        )
        prepared_conditioning = model.prepare_conditioning(cond_dict, cfg_scale=cfg_scale)

        print(f"  Generating with CFG: {cfg_scale}, Max Tokens: {max_new_tokens_per_chunk}")
        start_gen_time = time.time()
        
        with torch.no_grad(): 
            generated_codes = model.generate(
                prefix_conditioning=prepared_conditioning,
                audio_prefix_codes=audio_prefix_codes,
                max_new_tokens=max_new_tokens_per_chunk,
                cfg_scale=cfg_scale, batch_size=1, 
                sampling_params=custom_sampling_params, progress_bar=True 
            )
            # Store the full, original audio for this chunk on CPU
            current_chunk_audio_wav_cpu = model.autoencoder.decode(generated_codes).squeeze(0).squeeze(0).cpu()
        
        end_gen_time = time.time()
        if current_chunk_audio_wav_cpu.dim() == 0: 
            current_chunk_audio_wav_cpu = current_chunk_audio_wav_cpu.unsqueeze(0)
            
        print(f"  Chunk generation time: {end_gen_time - start_gen_time:.2f}s, Audio duration: {current_chunk_audio_wav_cpu.shape[-1]/sample_rate:.2f}s")

        if current_chunk_audio_wav_cpu.numel() > 0:
            generated_audio_chunks_cpu.append(current_chunk_audio_wav_cpu)
            previous_chunk_audio_for_prefix_cpu = current_chunk_audio_wav_cpu 
        else:
            print(f"  Warning: Chunk {i+1} resulted in empty audio.")
    
    if not generated_audio_chunks_cpu:
        print("No audio parts were generated.")
        return None, sample_rate
        
    # Concatenate with optional cross-fading
    final_audio_waveform_cpu = torch.tensor([], dtype=torch.float32, device='cpu')
    crossfade_samples = int((crossfade_duration_ms / 1000.0) * sample_rate) if crossfade_duration_ms > 0 else 0

    for i, chunk_wav in enumerate(generated_audio_chunks_cpu):
        if i == 0 or crossfade_samples == 0:
            final_audio_waveform_cpu = torch.cat((final_audio_waveform_cpu, chunk_wav), dim=-1)
        else:
            # Ensure previous part and current chunk are long enough for crossfade
            len_prev = final_audio_waveform_cpu.shape[-1]
            len_curr = chunk_wav.shape[-1]
            
            actual_xfade_samples = min(crossfade_samples, len_prev, len_curr)
            
            if actual_xfade_samples > 0:
                print(f"  Applying {actual_xfade_samples / sample_rate * 1000:.0f}ms crossfade between chunk {i} and {i+1}.")
                
                # Get overlapping parts
                prev_tail = final_audio_waveform_cpu[..., -actual_xfade_samples:]
                curr_head = chunk_wav[..., :actual_xfade_samples]
                
                # Create fade ramps
                fade_out_ramp = torch.linspace(1.0, 0.0, actual_xfade_samples, device='cpu')
                fade_in_ramp = torch.linspace(0.0, 1.0, actual_xfade_samples, device='cpu')
                
                # Apply fades
                crossfaded_region = prev_tail * fade_out_ramp + curr_head * fade_in_ramp
                
                # Concatenate
                non_overlapping_prev = final_audio_waveform_cpu[..., :-actual_xfade_samples]
                non_overlapping_curr = chunk_wav[..., actual_xfade_samples:]
                
                final_audio_waveform_cpu = torch.cat((non_overlapping_prev, crossfaded_region, non_overlapping_curr), dim=-1)
            else: # Not enough samples for crossfade, do direct concatenation
                 final_audio_waveform_cpu = torch.cat((final_audio_waveform_cpu, chunk_wav), dim=-1)

    print(f"\n🎉 Long audio generation complete. Total duration: {final_audio_waveform_cpu.shape[-1]/sample_rate:.2f}s")
    return final_audio_waveform_cpu.unsqueeze(0), sample_rate


if __name__ == '__main__':
    print("Setting up conceptual test for long audio generation with cross-fading...")
    setup_nltk_punkt() 
    
    class MockAutoencoder:
        def __init__(self): self.sampling_rate = 24000 
        def encode(self, wav_data_in): 
            print(f"Mock encode: {wav_data_in.shape}")
            if wav_data_in.shape[-1] == 0: return torch.empty(wav_data_in.shape[0], 9, 0)
            return torch.randn(wav_data_in.shape[0], 9, max(1, wav_data_in.shape[-1] // 240))
        def decode(self, codes_in): 
            print(f"Mock decode: {codes_in.shape}")
            if codes_in.shape[-1] == 0: return torch.empty(codes_in.shape[0], 1, 0)
            return torch.randn(codes_in.shape[0], 1, codes_in.shape[-1] * 240)

    class MockZonosModel:
        def __init__(self, device_in):
            self.autoencoder = MockAutoencoder(); self.device = device_in
        def prepare_conditioning(self, cond_dict_in, cfg_scale_in):
            return torch.randn(1 if cfg_scale_in == 1.0 else 2, 10, 1024).to(self.device) 
        def generate(self, **kwargs):
            num_frames = max(1, kwargs.get('max_new_tokens', 86*10) // 2 )
            print(f"Mock generate. CFG: {kwargs.get('cfg_scale')}, Prefix: {kwargs.get('audio_prefix_codes') is not None}, Frames: {num_frames}")
            return torch.randn(kwargs.get('batch_size',1), 9, num_frames).to(self.device)

    mock_device_main = torch.device('cpu')
    mock_model_main = MockZonosModel(mock_device_main)
    mock_speaker_embedding_main = torch.randn(1, 256).to(mock_device_main) 
    
    def mock_make_cond_dict_main(text, **kwargs): return {"text":text, **kwargs}

    long_text_example_main = "This is sentence one. This is sentence two, a bit longer. Sentence three is the final one for this test."
    
    print("\n--- Running Conceptual Long Audio Generation Test with Cross-fading ---")
    generated_audio_main, sr_main = generate_long_audio_from_text(
        full_text=long_text_example_main, model=mock_model_main, device=mock_device_main,
        speaker_embedding=mock_speaker_embedding_main, language="en-us",
        cfg_scale=2.0, custom_conditioning_params={'pitch_std':10.0}, custom_sampling_params={'min_p':0.1},
        seed=42, make_cond_dict_func=mock_make_cond_dict_main,
        max_chars_per_chunk=60, max_words_per_chunk=10, prefix_duration_seconds=0.5, 
        max_new_tokens_per_chunk= 86 * 5, crossfade_duration_ms=50 # Test with 50ms crossfade
    )

    if generated_audio_main is not None:
        print(f"\nConceptual test finished. Final audio shape: {generated_audio_main.shape}, Sample rate: {sr_main}")
    else:
        print("\nConceptual test finished. No audio generated.")

```
