# 🔧 Troubleshooting Guide for Unlimited Voice Cloning

## 🚨 Common Errors and Solutions

### 1. ❌ "espeak not installed on your system"

**Error Message:**
```
RuntimeError: espeak not installed on your system
```

**🚀 QUICK FIX:**
1. **Go back to Cell 2** and re-run it (it will install espeak automatically)
2. **Re-run Cell 3** to reload the system with espeak available
3. **Re-run Cell 5** to generate your audio

**Why this happens:** Zonos TTS uses espeak for phonemization (converting text to phonemes). Google Colab doesn't have espeak pre-installed.

**Prevention:** Always run Cell 2 completely before proceeding to other cells.

---

### 2. ⚠️ "Invalid Notebook" or JSON Errors

**Error Message:**
```
The Notebook Does Not Appear to Be Valid JSON
```

**🚀 SOLUTION:**
- Use the **fixed notebook link**: 
  ```
  https://colab.research.google.com/github/Wamp1re-Ai/Zonos/blob/efficient/Efficient_Voice_Cloning_Colab.ipynb
  ```
- The notebook has been completely rebuilt with proper JSON structure

---

### 3. 🔥 NumPy 2.x Compatibility Issues

**Error Message:**
```
AttributeError: module 'numpy' has no attribute 'something'
```

**🚀 SOLUTION:**
1. **Restart Runtime**: Runtime → Restart runtime
2. **Re-run Cell 1**: Clone the repository
3. **Re-run Cell 2**: Install dependencies (forces NumPy 1.x)
4. **Re-run Cell 3**: Load the system
5. **Continue normally**

**Why this happens:** Google Colab sometimes has NumPy 2.x which is incompatible with some dependencies.

---

### 4. 📦 Dependency Installation Failures

**Error Message:**
```
Failed to install [package]
```

**🚀 SOLUTIONS:**

**Option A - UV Package Manager (Recommended):**
```python
# In Cell 2, UV should install automatically
# If it fails, the system falls back to pip
```

**Option B - Manual Installation:**
```python
!pip install torch torchaudio transformers accelerate
!pip install librosa soundfile scipy matplotlib
!pip install "numpy<2.0"
!apt-get update && apt-get install -y espeak espeak-data
```

---

### 5. 🎵 Model Loading Errors

**Error Message:**
```
Error loading model: [various errors]
```

**🚀 SOLUTIONS:**
1. **Check internet connection** - Model downloads from Hugging Face
2. **Restart runtime** if memory issues
3. **Try T4 GPU** in Colab (Runtime → Change runtime type → T4 GPU)
4. **Re-run cells in order**: 1 → 2 → 3

---

### 6. 🔥 Unlimited Mode Not Available

**Symptoms:**
- Shows "📢 Using STANDARD mode" instead of unlimited
- No unlimited options in Cell 5

**🚀 SOLUTION:**
1. **Verify efficient branch**: Cell 1 should show "Current branch: efficient"
2. **Check for files**: Cell 1 should show "✅ efficient_voice_cloning.py - Found"
3. **Re-clone if needed**: Re-run Cell 1 to clone the efficient branch

---

### 7. ⚡ Performance Issues / Slow Generation

**Symptoms:**
- Very slow audio generation
- High RTF (Real-Time Factor > 1.0)

**🚀 OPTIMIZATIONS:**
1. **Enable GPU**: Runtime → Change runtime type → T4 GPU
2. **Use FP16**: Enable "Use FP16" toggle in Cell 5
3. **Enable Unlimited Mode**: Turn on "Unlimited Mode" for long texts
4. **Reduce chunk size**: Lower "Target Chunk Chars" for memory issues

---

### 8. 🎤 Voice Upload Issues

**Error Message:**
```
Error processing audio: [various errors]
```

**🚀 SOLUTIONS:**
1. **Supported formats**: WAV, MP3, FLAC, M4A
2. **Audio length**: 10-30 seconds recommended
3. **File size**: Keep under 25MB for Colab
4. **Quality**: Clear speech, minimal background noise

---

### 9. 🔊 Audio Playback Issues

**Symptoms:**
- No audio plays in Colab
- Download fails

**🚀 SOLUTIONS:**
1. **Browser audio**: Enable audio in browser settings
2. **Download manually**: Files will be saved as `unlimited_audio_[timestamp].wav`
3. **Check file size**: Very long audio may take time to process

---

## 🚀 Performance Optimization Tips

### For Maximum Speed:
1. ✅ **Enable Unlimited Mode** (Cell 5)
2. ✅ **Use FP16 precision** (Cell 5)
3. ✅ **Enable GPU** (Runtime settings)
4. ✅ **Upload voice once** (gets cached automatically)
5. ✅ **Use efficient chunking** for long texts

### For Maximum Quality:
1. ✅ **High-quality voice sample** (clear, 15-20 seconds)
2. ✅ **Disable FP16** if quality issues
3. ✅ **Smaller chunk sizes** for very long texts
4. ✅ **Clean input text** (proper punctuation)

---

## 📋 Step-by-Step Recovery Process

If you're having multiple issues, follow this complete reset:

### 🔄 Complete Reset:
1. **Runtime → Restart runtime**
2. **Runtime → Change runtime type → T4 GPU**
3. **Run Cell 1**: Clone efficient branch
4. **Run Cell 2**: Install all dependencies (including espeak)
5. **Run Cell 3**: Load system (should show unlimited features)
6. **Run Cell 4**: Upload voice sample
7. **Run Cell 5**: Generate unlimited audio

### ✅ Success Indicators:
- Cell 1: "Current branch: efficient"
- Cell 2: "✅ espeak installed successfully!"
- Cell 3: "🔥 UNLIMITED Voice Cloning (NO LENGTH CAPS!)"
- Cell 4: "✅ Voice cloning successful!"
- Cell 5: "🔥 UNLIMITED MODE SUCCESS!"

---

## 🆘 Still Having Issues?

If none of these solutions work:

1. **Check the GitHub Issues**: [Zonos Issues](https://github.com/Wamp1re-Ai/Zonos/issues)
2. **Try the Enhanced Notebook**: If unlimited fails, try the enhanced version
3. **Use Standard Mode**: Disable unlimited mode as fallback
4. **Report the Issue**: Include error messages and steps to reproduce

---

## 🎉 Success Metrics

When everything is working correctly, you should see:

- ⚡ **RTF < 0.5** (faster than real-time)
- 🔥 **"NO LENGTH RESTRICTIONS APPLIED!"**
- 📊 **Cache hit rates** for repeated voices
- 🚀 **Speedup estimates** vs standard mode
- 🎵 **Audio duration** matching your text length

**Remember: The unlimited system can generate hours of audio - there are truly no caps!** 🔥
