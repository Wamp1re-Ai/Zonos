import nltk
import torch
import torchaudio # For saving/loading audio, and potentially for operations
import time

# --- Text Segmentation Function (from previous task) ---
def setup_nltk_punkt():
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)

def segment_text_into_chunks(text: str, max_chars_per_chunk: int = 600, max_words_per_chunk: int = 100) -> list[str]:
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
    quality_preset: str, # e.g., "Balanced", "Expressive"
    seed: int,
    # make_cond_dict and other helpers should be available from the notebook's global scope (Cell 3)
    # Or passed explicitly if this were a standalone module
    make_cond_dict_func, # Function reference
    prepare_conditioning_func, # Function reference (model.prepare_conditioning)
    voice_quality_metrics=None, # Optional: dict from voice quality analysis
    max_chars_per_chunk: int = 500, # Slightly reduced for safety with prefixes
    max_words_per_chunk: int = 80,  # Slightly reduced
    prefix_duration_seconds: float = 1.5,
    max_new_tokens_per_chunk: int = 86 * 25 # Approx 25 seconds to allow for prefix and prevent overly long individual chunks
) -> tuple[torch.Tensor | None, int]:
    """
    Generates long audio by segmenting text, generating chunks with audio prefixing,
    and concatenating the results.
    """
    print(f"🎙️ Starting long audio generation for text ({len(full_text)} chars)...")
    
    text_chunks = segment_text_into_chunks(full_text, max_chars_per_chunk, max_words_per_chunk)
    if not text_chunks:
        print("No text chunks to process.")
        return None, model.autoencoder.sampling_rate

    torch.manual_seed(seed) # Set seed for the first chunk and overall determinism if model internals are fixed
    
    full_audio_parts = []
    previous_chunk_audio_wav = None
    sample_rate = model.autoencoder.sampling_rate

    for i, chunk_text in enumerate(text_chunks):
        print(f"\nGenerating chunk {i+1}/{len(text_chunks)}: \"{chunk_text[:80]}...\"")
        
        audio_prefix_codes = None
        if previous_chunk_audio_wav is not None and previous_chunk_audio_wav.numel() > 0:
            prefix_num_samples = int(prefix_duration_seconds * sample_rate)
            if previous_chunk_audio_wav.shape[-1] > prefix_num_samples:
                audio_prefix_snippet = previous_chunk_audio_wav[..., -prefix_num_samples:]
            else:
                audio_prefix_snippet = previous_chunk_audio_wav # Use full previous chunk if shorter
            
            print(f"  Using audio prefix of {audio_prefix_snippet.shape[-1]/sample_rate:.2f}s from previous chunk.")
            # Ensure correct device and unsqueeze for batch dimension
            audio_prefix_snippet_device = audio_prefix_snippet.to(device).unsqueeze(0)
            
            # Handle potential empty tensor after operations
            if audio_prefix_snippet_device.numel() > 0:
                try:
                    # Encode requires [Batch, Channels (1 for mono), Samples]
                    # Model expects [Batch, NumCodebooks, Frames]
                    # Autoencoder handles [B, 1, T] -> [B, D, T'] where D is num_quantizers/codebooks
                    if audio_prefix_snippet_device.dim() == 2: # Should be [1, Samples]
                        audio_prefix_snippet_device = audio_prefix_snippet_device.unsqueeze(1) # Add channel dim -> [1, 1, Samples]

                    with torch.no_grad(): # Ensure no gradients are computed for encoding
                         audio_prefix_codes_encoded = model.autoencoder.encode(audio_prefix_snippet_device)
                    
                    if audio_prefix_codes_encoded.numel() > 0:
                        audio_prefix_codes = audio_prefix_codes_encoded
                        print(f"    Encoded audio prefix: {audio_prefix_codes.shape}") # Expected: [1, 9, Frames]
                    else:
                        print("    Warning: Encoding audio prefix resulted in empty tensor. Skipping prefix.")
                except Exception as e:
                    print(f"    Error encoding audio prefix: {e}. Skipping prefix for this chunk.")
            else:
                print("    Warning: Audio prefix snippet is empty. Skipping prefix.")


        # Get generation parameters based on quality_preset (logic adapted from Cell 5)
        # This assumes 'model' and 'device' are available from the parent scope if not passed to a sub-function
        quality_score = voice_quality_metrics.get('quality_score', 0.7) if voice_quality_metrics else 0.7
        snr_estimate = voice_quality_metrics.get('snr_estimate', 20.0) if voice_quality_metrics else 20.0
        
        emotion_vector_override = None
        if quality_preset == "Conservative":
            base_pitch, base_rate, base_min_p, base_temp = 8.0, 10.0, 0.02, 0.6; cfg_scale = 2.5
        elif quality_preset == "Fast (Less Expressive)":
            base_pitch, base_rate, base_min_p, base_temp = 8.0, 10.0, 0.03, 0.7; cfg_scale = 1.5
        elif quality_preset == "Expressive":
            base_pitch, base_rate, base_min_p, base_temp = 18.0, 14.0, 0.06, 0.85; cfg_scale = 2.0
            emotion_vector_override = [0.6, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05, 0.05]
        elif quality_preset == "Creative":
            base_pitch, base_rate, base_min_p, base_temp = 22.0, 16.0, 0.08, 0.95; cfg_scale = 1.8
            emotion_vector_override = [0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]
        else:  # Balanced (default)
            base_pitch, base_rate, base_min_p, base_temp = 12.0, 12.0, 0.04, 0.75; cfg_scale = 2.2

        quality_factor = min(1.2, max(0.8, quality_score * 1.2))
        snr_factor = min(1.1, max(0.9, (snr_estimate - 15.0) / 20.0 + 1.0))
        pitch_std = max(5.0, min(25.0, base_pitch * quality_factor))
        speaking_rate = max(8.0, min(18.0, base_rate * snr_factor))
        min_p_val = max(0.01, min(0.15, base_min_p * quality_factor))
        temperature_val = max(0.5, min(1.0, base_temp * quality_factor))
        cfg_scale = max(1.0, min(3.0, cfg_scale))

        current_custom_conditioning = {'pitch_std': pitch_std, 'speaking_rate': speaking_rate}
        current_custom_sampling = {'min_p': min_p_val, 'temperature': temperature_val}
        if emotion_vector_override: current_custom_conditioning['emotion'] = emotion_vector_override
        
        cond_dict_extra_args = {}
        if emotion_vector_override is not None:
            cond_dict_extra_args['emotion'] = emotion_vector_override

        cond_dict = make_cond_dict_func(
            text=chunk_text, language=language, speaker=speaker_embedding,
            device=device, **current_custom_conditioning # Pass full conditioning dict
        )
        
        # Use the model's own prepare_conditioning method
        prepared_conditioning = model.prepare_conditioning(cond_dict, cfg_scale=cfg_scale)

        print(f"  Generating with CFG: {cfg_scale}, Max Tokens: {max_new_tokens_per_chunk}")
        start_gen_time = time.time()
        generated_codes = model.generate(
            prefix_conditioning=prepared_conditioning,
            audio_prefix_codes=audio_prefix_codes, # Pass the encoded audio prefix
            max_new_tokens=max_new_tokens_per_chunk,
            cfg_scale=cfg_scale,
            batch_size=1, # Assuming batch_size 1 for sequential generation
            sampling_params=current_custom_sampling,
            progress_bar=True 
        )
        current_chunk_audio_wav = model.autoencoder.decode(generated_codes).cpu().detach().squeeze(0) # Squeeze batch
        end_gen_time = time.time()
        print(f"  Chunk generation time: {end_gen_time - start_gen_time:.2f}s, Audio duration: {current_chunk_audio_wav.shape[-1]/sample_rate:.2f}s")

        if current_chunk_audio_wav.numel() > 0:
            full_audio_parts.append(current_chunk_audio_wav)
            previous_chunk_audio_wav = current_chunk_audio_wav
        else:
            print(f"  Warning: Chunk {i+1} resulted in empty audio. Skipping.")
            # previous_chunk_audio_wav remains the one from the last successful chunk

    if not full_audio_parts:
        print("No audio parts were generated.")
        return None, sample_rate
        
    full_audio_output = torch.cat(full_audio_parts, dim=-1)
    print(f"\n🎉 Long audio generation complete. Total duration: {full_audio_output.shape[-1]/sample_rate:.2f}s")
    
    return full_audio_output, sample_rate

# Example of how this might be integrated into the Colab notebook:
# This __main__ block is for conceptual testing and won't run directly when pasted as a Colab cell.
if __name__ == '__main__':
    print("Setting up conceptual test for long audio generation...")
    # This requires a mock model, device, speaker_embedding, etc.
    # For now, this block just confirms the script can be parsed.
    
    # Mock objects and functions that would normally come from Colab's global scope
    class MockAutoencoder:
        def __init__(self):
            self.sampling_rate = 24000 # Example SR
        def encode(self, wav_data):
            print(f"Mock encode called with shape: {wav_data.shape}")
            # Simulate encoding: [B, 1, T_samples] -> [B, NumCodebooks, T_frames]
            # T_frames is roughly T_samples / hop_length. Let hop_length be ~240 for 100fps at 24kHz.
            num_frames = wav_data.shape[-1] // 240
            return torch.randn(wav_data.shape[0], 9, num_frames) # 9 codebooks
        def decode(self, codes):
            print(f"Mock decode called with shape: {codes.shape}")
            # Simulate decoding: [B, NumCodebooks, T_frames] -> [B, 1, T_samples]
            num_samples = codes.shape[-1] * 240
            return torch.randn(codes.shape[0], 1, num_samples)

    class MockZonosModel:
        def __init__(self, device):
            self.autoencoder = MockAutoencoder()
            self.device = device
        def prepare_conditioning(self, cond_dict, cfg_scale):
            print(f"Mock prepare_conditioning called with cfg_scale: {cfg_scale}")
            # Simulate conditioning: returns a tensor [batch_cfg, seq_len, dim]
            # If cfg_scale=1, batch_cfg=1, else batch_cfg=2
            batch_cfg = 1 if cfg_scale == 1.0 else 2
            return torch.randn(batch_cfg, 10, 1024).to(self.device) # Dummy conditioning
        def generate(self, prefix_conditioning, audio_prefix_codes, max_new_tokens, cfg_scale, batch_size, sampling_params, progress_bar):
            print(f"Mock generate called. CFG: {cfg_scale}, Max Tokens: {max_new_tokens}")
            if audio_prefix_codes is not None:
                print(f"  Audio prefix codes shape: {audio_prefix_codes.shape}")
            # Simulate code generation, outputting a certain number of frames
            # Let's assume max_new_tokens translates to roughly max_new_tokens audio frames for DAC
            num_frames = max_new_tokens // 2 # Rough approximation
            return torch.randn(batch_size, 9, num_frames).to(self.device) # Dummy codes

    mock_device = torch.device('cpu')
    mock_model = MockZonosModel(mock_device)
    mock_speaker_embedding = torch.randn(1, 256).to(mock_device) # Dummy speaker embedding
    
    # Mock make_cond_dict (simplified)
    def mock_make_cond_dict(text, language, speaker, device, **kwargs):
        print(f"Mock make_cond_dict called for text: '{text[:20]}...'")
        return {"text": text, "language": language, "speaker": speaker, "device": device, **kwargs}

    long_text_example = (
        "This is the first paragraph. It has several sentences. Let's see how the chunking works. "
        "Now for a second paragraph. This one might also be split. Audio prefixing should help maintain continuity. "
        "A third and final paragraph to ensure the process completes gracefully. This is quite a lot of text."
    )

    print("\n--- Running Conceptual Long Audio Generation Test ---")
    generated_audio, sr = generate_long_audio_from_text(
        full_text=long_text_example,
        model=mock_model,
        device=mock_device,
        speaker_embedding=mock_speaker_embedding,
        language="en-us",
        quality_preset="Balanced", # This will determine internal cfg, emotion etc.
        seed=42,
        make_cond_dict_func=mock_make_cond_dict,
        prepare_conditioning_func=mock_model.prepare_conditioning, # Pass the method directly
        voice_quality_metrics={}, # Empty for mock
        max_chars_per_chunk=150, # Shorter chunks for more iterations in testing
        max_words_per_chunk=30,
        prefix_duration_seconds=1.0
    )

    if generated_audio is not None:
        print(f"\nConceptual test finished. Final audio shape: {generated_audio.shape}, Sample rate: {sr}")
        # In a real scenario, you'd save or play this audio:
        # torchaudio.save("conceptual_long_audio_output.wav", generated_audio.cpu(), sr)
    else:
        print("\nConceptual test finished. No audio generated.")

```

**Explanation of Key Aspects and Assumptions:**

1.  **`segment_text_into_chunks`:** This function is included at the top of the script for completeness.
2.  **`generate_long_audio_from_text` Function Signature:**
    *   It takes essential parameters like `full_text`, `model`, `device`, `speaker_embedding`, `language`, `quality_preset`, and `seed`.
    *   Crucially, it now expects `make_cond_dict_func` and `prepare_conditioning_func` to be passed in. In the Colab environment, these would be `make_cond_dict` (from `zonos.conditioning`, imported in Cell 3) and `model.prepare_conditioning` respectively. This makes the function more testable and clearly defines its dependencies.
    *   `voice_quality_metrics` is included as it's used by the preset logic.
    *   Chunking parameters (`max_chars_per_chunk`, `max_words_per_chunk`) and `prefix_duration_seconds` are configurable.
    *   `max_new_tokens_per_chunk` is set to a reasonable value (e.g., `86 * 25` for ~25 seconds) to prevent individual chunks from becoming excessively long, which could lead to OOM or very slow generation for that chunk.
3.  **Parameter Retrieval for Chunks:**
    *   The logic for determining `cfg_scale`, `custom_conditioning` (including emotion vectors), and `custom_sampling` based on the `quality_preset` is replicated from Cell 5 of the notebook. This ensures that each chunk is generated with parameters consistent with the user's choice.
4.  **Audio Prefixing:**
    *   The last `prefix_duration_seconds` of the previously generated chunk (`previous_chunk_audio_wav`) is taken.
    *   It's converted to the correct device and shape (`[1, 1, num_samples]`) before being passed to `model.autoencoder.encode()`.
    *   The resulting `audio_prefix_codes` are then passed to `model.generate()`.
    *   Error handling is added for encoding and to check for empty tensors.
5.  **`model.generate` Call:**
    *   It now receives both `prefix_conditioning` (from text, speaker, etc.) and `audio_prefix_codes`.
6.  **Audio Concatenation:**
    *   `current_chunk_audio_wav.squeeze(0)` is used because the autoencoder decodes to `[B, 1, T]`, and we want `[1, T]` or just `[T]` for concatenation. Since `batch_size` for generation is 1, squeezing the batch dimension is appropriate.
    *   The generated audio chunks are appended to `full_audio_parts` and then concatenated using `torch.cat(full_audio_parts, dim=-1)`.
    *   **Assumption:** As stated before, it's assumed `model.generate` outputs only the *newly generated* audio when a prefix is supplied. If it re-outputs the prefix, that prefix would need to be trimmed from `current_chunk_audio_wav` before concatenation.
7.  **Seed Handling:**
    *   `torch.manual_seed(seed)` is set once before the loop. This ensures that if all other inputs (including prefixes) are deterministic, the overall generation process is reproducible. The prefix itself carries the history, making subsequent chunks deterministic continuations.
8.  **Return Value:** The function returns the complete concatenated audio waveform as a 2D tensor `[1, num_samples]` (or could be squeezed to 1D if preferred by downstream use) and the sample rate. The current code returns `[num_samples]` after `squeeze(0)`.
9.  **Mock Objects for Testing:** The `if __name__ == '__main__':` block includes mock objects for `ZonosModel` and its components, and a mock `make_cond_dict` to allow conceptual testing of the `generate_long_audio_from_text` flow without needing the actual model or a GPU. This helps verify the logic flow and parameter passing.

This code should be placed in a new cell in the Colab notebook, ideally after Cell 3 (where the model and `make_cond_dict` are loaded/defined) and before Cell 5 (the interactive generation cell). This allows users to call `generate_long_audio_from_text` directly if they have very long text. An alternative would be to modify Cell 5 to use this function if the input text exceeds a certain length. For now, I'm providing it as a separate utility function.The Python code for the `generate_long_audio_from_text` function, along with the necessary `segment_text_into_chunks` helper, has been created. This code is intended to be placed in a new cell within the `Enhanced_Voice_Cloning_Colab.ipynb`.

**Key features of the implementation:**

1.  **`segment_text_into_chunks` Function:**
    *   Re-included from the previous task for completeness. It uses NLTK to split text into sentences and then groups them into manageable chunks based on character and word limits, prioritizing sentence boundaries.
    *   Includes `setup_nltk_punkt()` for on-demand download of NLTK resources.

2.  **`generate_long_audio_from_text` Function:**
    *   **Signature:** Accepts `full_text`, `model`, `device`, `speaker_embedding`, `language`, `quality_preset`, `seed`, references to `make_cond_dict_func` and `model.prepare_conditioning`, and optional `voice_quality_metrics`. Also includes parameters for chunking behavior (`max_chars_per_chunk`, `max_words_per_chunk`), `prefix_duration_seconds`, and `max_new_tokens_per_chunk`.
    *   **Text Segmentation:** Uses the included `segment_text_into_chunks` to divide the `full_text`.
    *   **Sequential Loop:** Iterates through text chunks.
        *   **Audio Prefixing:**
            *   For chunks after the first, it extracts the last `prefix_duration_seconds` from the previously generated audio chunk (`previous_chunk_audio_wav`).
            *   This audio snippet is encoded into `audio_prefix_codes` using `model.autoencoder.encode()`. The shape is adjusted to `[1, 1, num_samples]` before encoding.
            *   Error handling for empty prefix snippets or encoding errors is included.
        *   **Parameter Retrieval:** Logic from Cell 5 of the notebook is adapted to determine `cfg_scale`, conditioning parameters (including `emotion_vector` for "Expressive" and "Creative" presets), and sampling parameters based on the chosen `quality_preset` and `voice_quality_metrics`.
        *   **Chunk Generation:**
            *   `make_cond_dict_func` (expected to be the `make_cond_dict` from `zonos.conditioning`) is called to prepare the dictionary for conditioning.
            *   `model.prepare_conditioning` is called to get the `prepared_conditioning` tensor.
            *   `model.generate` is invoked with both `prepared_conditioning` and the `audio_prefix_codes`.
            *   The output `codes` are decoded using `model.autoencoder.decode()`.
        *   **Audio Concatenation:** The generated audio for the current chunk is appended to a list (`full_audio_parts`).
            *   **Assumption:** `model.generate`, when given `audio_prefix_codes`, outputs only the *newly generated* audio content (not the prefix itself). This means direct concatenation is appropriate.
        *   The `current_chunk_audio_wav` is stored as `previous_chunk_audio_wav` for the next iteration.
    *   **Seed Handling:** `torch.manual_seed(seed)` is set once before the loop to ensure reproducibility for the first chunk and, by extension, the sequence if the model and prefixing are deterministic.
    *   **Output:** The function returns a tuple containing the full concatenated audio waveform (as a `torch.Tensor`) and the `sample_rate`. If no audio parts are generated, it returns `(None, sample_rate)`.

3.  **`if __name__ == '__main__':` Block:**
    *   A conceptual test suite is included with mock objects for `ZonosModel`, `MockAutoencoder`, and `make_cond_dict`. This allows for testing the flow and logic of `generate_long_audio_from_text` without requiring the actual model or a GPU environment.

**Placement in Colab:**
This entire code block (including both functions and the `if __name__ == '__main__'` example) should be placed in a new code cell in the `Enhanced_Voice_Cloning_Colab.ipynb`. A suitable location would be after Cell 3 (Model Loading) and before Cell 5 (Interactive Generation), making it available as a utility function.

The Python code is as follows:
