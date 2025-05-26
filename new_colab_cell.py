import time
import torchaudio
import IPython.display as ipd
import numpy as np
import os

# Ensure the /content/Zonos directory exists for saving audio files
if not os.path.exists("/content/Zonos/benchmark_audio"):
    os.makedirs("/content/Zonos/benchmark_audio")

def run_benchmark_trial(text_input, language_code, seed_value, cfg_scale_value, quality_preset_value, 
                        speaker_embedding_tensor, voice_quality_data, 
                        zonos_model, torch_device, is_enhanced_available,
                        run_warmup=False):
    """
    Runs a benchmark trial for Zonos TTS generation.

    Args:
        text_input (str): Text to synthesize.
        language_code (str): Language code (e.g., 'en-us').
        seed_value (int): Random seed for reproducibility.
        cfg_scale_value (float): The CFG scale to test.
        quality_preset_value (str): Quality preset ('Conservative', 'Balanced', 'Expressive', 'Creative').
        speaker_embedding_tensor (torch.Tensor): Speaker embedding.
        voice_quality_data (dict): Voice quality metrics.
        zonos_model (zonos.model.Zonos): The loaded Zonos model.
        torch_device (torch.device): The device to run on.
        is_enhanced_available (bool): Flag if enhanced functions are available.
        run_warmup (bool): If True, runs a short generation first.

    Returns:
        tuple: (rtf, audio_duration, generation_time, audio_filepath)
    """
    print(f"\\n--- Benchmarking Trial ---")
    print(f"Text: '{text_input[:50]}...'")
    print(f"CFG Scale: {cfg_scale_value}, Preset: {quality_preset_value}")

    torch.manual_seed(seed_value)

    # Parameter setup logic adapted from Cell 5
    quality_score = voice_quality_data.get('quality_score', 0.7) if voice_quality_data else 0.7
    snr_estimate = voice_quality_data.get('snr_estimate', 20.0) if voice_quality_data else 20.0

    if quality_preset_value == "Conservative":
        base_pitch, base_rate, base_min_p, base_temp = 8.0, 10.0, 0.02, 0.6
        # Original CFG for this preset was 2.5, we'll use the passed cfg_scale_value
    elif quality_preset_value == "Expressive":
        base_pitch, base_rate, base_min_p, base_temp = 18.0, 14.0, 0.06, 0.85
    elif quality_preset_value == "Creative":
        base_pitch, base_rate, base_min_p, base_temp = 22.0, 16.0, 0.08, 0.95
    else:  # Balanced (default)
        base_pitch, base_rate, base_min_p, base_temp = 12.0, 12.0, 0.04, 0.75
        # Original CFG for balanced was 2.2

    quality_factor = min(1.2, max(0.8, quality_score * 1.2))
    snr_factor = min(1.1, max(0.9, (snr_estimate - 15.0) / 20.0 + 1.0))
    
    pitch_std = max(5.0, min(25.0, base_pitch * quality_factor))
    speaking_rate = max(8.0, min(18.0, base_rate * snr_factor))
    min_p_val = max(0.01, min(0.15, base_min_p * quality_factor))
    temperature_val = max(0.5, min(1.0, base_temp * quality_factor))

    current_custom_conditioning = {'pitch_std': pitch_std, 'speaking_rate': speaking_rate}
    current_custom_sampling = {'min_p': min_p_val, 'temperature': temperature_val}

    # Use the global 'enhanced_generate_speech' or 'simple_enhanced_generate_speech' if available
    # These functions internally call model.generate after preparing conditioning with model.prepare_conditioning
    # We need to ensure model.prepare_conditioning is called with the correct cfg_scale_value by these helper functions,
    # or call model.generate directly.
    # The fallback simple_enhanced_generate_speech in Cell 3 uses make_cond_dict and then model.prepare_conditioning(cond_dict)
    # without passing cfg_scale to prepare_conditioning. This will need adjustment if we rely on it.
    # For now, let's assume we will call model.generate directly for more control in benchmark.

    # Warm-up run (optional, typically for the very first call to a torch.compile'd function)
    if run_warmup:
        print("Running warmup...")
        warmup_text = "Warmup."
        warmup_cond_dict = make_cond_dict(
            text=warmup_text, language=language_code, speaker=speaker_embedding_tensor,
            device=torch_device, **current_custom_conditioning
        )
        # Pass cfg_scale_value to prepare_conditioning
        warmup_conditioning = zonos_model.prepare_conditioning(warmup_cond_dict, cfg_scale=cfg_scale_value)
        _ = zonos_model.generate(
            prefix_conditioning=warmup_conditioning,
            max_new_tokens=30, # Short generation
            cfg_scale=cfg_scale_value,
            batch_size=1,
            sampling_params=current_custom_sampling,
            progress_bar=False # Disable progress bar for warmup
        )
        print("Warmup complete.")

    # Actual generation
    generation_start_time = time.time()

    cond_dict = make_cond_dict(
        text=text_input, language=language_code, speaker=speaker_embedding_tensor,
        device=torch_device, **current_custom_conditioning
    )
    # Critical: Pass cfg_scale_value to prepare_conditioning
    prepared_conditioning = zonos_model.prepare_conditioning(cond_dict, cfg_scale=cfg_scale_value)
    
    # Determine max_new_tokens based on text length (simplified from notebook)
    tokens_per_char = 15 # Adjusted from 20 for safety with various languages
    estimated_tokens = len(text_input) * tokens_per_char
    min_gen_tokens = 500 
    # Max tokens for ~2 mins of audio. 86 tokens/sec * 120 sec = 10320
    # Let's use a slightly lower cap for safety in benchmarks if text is extremely long
    max_gen_tokens = max(min_gen_tokens, min(estimated_tokens, 86 * 100))


    codes = zonos_model.generate(
        prefix_conditioning=prepared_conditioning,
        max_new_tokens=max_gen_tokens,
        cfg_scale=cfg_scale_value,
        batch_size=1,
        sampling_params=current_custom_sampling,
        progress_bar=True
    )
    audio_output = zonos_model.autoencoder.decode(codes).cpu().detach()
    
    generation_time = time.time() - generation_start_time
    sample_rate = zonos_model.autoencoder.sampling_rate
    
    if audio_output.dim() == 2 and audio_output.size(0) > 1:
        audio_output = audio_output[0:1, :] # Ensure mono
    audio_duration = audio_output.shape[-1] / sample_rate
    
    rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')
    
    print(f"  Generated {audio_duration:.2f}s audio in {generation_time:.2f}s. RTF: {rtf:.2f}")

    # Save audio
    audio_filename = f"benchmark_cfg_{cfg_scale_value}_seed_{seed_value}_text_{text_input[:15].replace(' ', '_')}.wav"
    audio_filepath = os.path.join("/content/Zonos/benchmark_audio", audio_filename)
    torchaudio.save(audio_filepath, audio_output.squeeze(0), sample_rate)
    print(f"  Saved audio to: {audio_filepath}")

    return rtf, audio_duration, generation_time, audio_filepath

# --- Define Benchmark Parameters ---
texts_to_benchmark = [
    "Hello world.",
    "This is a test of the emergency broadcast system.",
    "The quick brown fox jumps over the lazy dog, and other fables are often used for typing practice."
]
cfg_scales_to_benchmark = [1.0, 1.5, 2.2]
benchmark_language = "en-us"
benchmark_seed = 42 
# Use the "Balanced" preset logic for conditioning/sampling params, but CFG will be overridden
benchmark_quality_preset = "Balanced" 

benchmark_results = []

# --- Ensure Prerequisite Globals are Available ---
# These should be set by running previous cells in the notebook
# model, device, cloned_voice (speaker_embedding), voice_quality_metrics, ENHANCED_AVAILABLE
# For robustness, check if they exist:
if 'model' not in globals() or 'device' not in globals():
    print("⚠️ Model or device not found. Please run previous cells (1-3) to load the model.")
else:
    current_speaker_embedding = globals().get('cloned_voice', None)
    if current_speaker_embedding is None:
        print("🎤 No cloned voice found (Cell 4 not run or failed). Benchmarking with default/no speaker embedding if model supports.")
        # Fallback: create a dummy speaker embedding if necessary, or rely on model's default
        # For Zonos, speaker embedding is usually required. The make_cond_dict might handle None for speaker.
        # Let's assume for now that if no 'cloned_voice', the user intends to test without specific cloning.
        # The `make_cond_dict` in zonos.conditioning seems to handle speaker=None by omitting speaker_embedding.
        
    current_voice_quality_metrics = globals().get('voice_quality_metrics', {}) # Default to empty dict if not found
    is_enhanced_globally = globals().get('ENHANCED_AVAILABLE', False)

    # --- Run Warmup (once) ---
    # A single warmup before the loop. Subsequent calls to model.generate with different
    # cfg_scale values might still trigger some recompilation if internal control flow changes significantly.
    # The `torch.compile` cache should ideally handle recompilations for different graph structures.
    print("\\n🔥 Running a single warm-up generation before benchmark loop (using CFG 2.2)...")
    run_benchmark_trial(
        "Warmup generation.", benchmark_language, benchmark_seed, 2.2, 
        benchmark_quality_preset, current_speaker_embedding, current_voice_quality_metrics,
        model, device, is_enhanced_globally, run_warmup=False # No nested warmup
    )
    print("🔥 Warm-up finished.\\n")

    # --- Run Benchmark Loop ---
    for cfg_val in cfg_scales_to_benchmark:
        for text_sample in texts_to_benchmark:
            rtf, audio_dur, gen_time, audio_file = run_benchmark_trial(
                text_sample, benchmark_language, benchmark_seed, cfg_val,
                benchmark_quality_preset, current_speaker_embedding, current_voice_quality_metrics,
                model, device, is_enhanced_globally
            )
            benchmark_results.append({
                "text": text_sample,
                "cfg_scale": cfg_val,
                "rtf": rtf,
                "audio_duration": audio_dur,
                "generation_time": gen_time,
                "audio_file": audio_file
            })

    # --- Display Results ---
    print("\\n\\n--- Benchmark Summary ---")
    print(f"{'CFG':<5} | {'Text Length':<12} | {'RTF':<5} | {'Audio (s)':<10} | {'Gen Time (s)':<12} | {'File':<60}")
    print("-" * 120)
    for res in benchmark_results:
        text_len_desc = "Short" if len(res['text']) < 20 else "Medium" if len(res['text']) < 70 else "Long"
        print(f"{res['cfg_scale']:<5.1f} | {text_len_desc:<12} | {res['rtf']:<5.2f} | {res['audio_duration']:<10.2f} | {res['generation_time']:<12.2f} | {res['audio_file']:<60}")
        ipd.display(ipd.HTML(f"<b>Text:</b> {res['text']}<br><b>CFG:</b> {res['cfg_scale']}, <b>File:</b> {res['audio_file']}"))
        ipd.display(ipd.Audio(res['audio_file']))
        print("-" * 60)

    # Store results in a global for easy access later if needed
    globals()['benchmark_run_results'] = benchmark_results

print("\\n✅ Benchmarking cell execution complete.")
print("Note: You might need to reload the model (Cell 3) for the zonos.model.py changes to take effect if you haven't already.")
print("After reloading the model, re-run Cell 4 (if using cloned voice) and Cell 5 (optional, to check normal generation), then this benchmark cell.")
