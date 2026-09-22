import queue
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# ==========================================
# 1. CONFIGURATION
# ==========================================
# We use "base" because it balances translation accuracy and CPU/GPU speed.
# If you experience lag on CPU, change this to "tiny".
MODEL_SIZE = "small"
# Hardware acceleration settings:
# - Once your NVIDIA CUDA libraries are installed, change DEVICE to "cuda"
#   and COMPUTE_TYPE to "float16" for lightning-fast performance.
DEVICE = "cpu"       
COMPUTE_TYPE = "float32" 

# Audio Settings
SAMPLE_RATE = 16000  # Whisper strictly expects 16kHz audio
CHUNK_DURATION = 3   # Listen in 3-second blocks to capture full Japanese sentences
BLOCK_SIZE = SAMPLE_RATE * CHUNK_DURATION

# Thread-safe queue to pass audio from the microphone thread to the main thread
audio_queue = queue.Queue()

# ==========================================
# 2. INITIALIZE FASTER-WHISPER
# ==========================================
print(f"🔄 Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
try:
    model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load Whisper model: {e}")
    sys.exit(1)


def audio_callback(indata, frames, time, status):
    """This background function is called by sounddevice for every new audio block."""
    if status:
        print(f"⚠️ Audio status warning: {status}", file=sys.stderr)
    # Put a copy of the raw microphone audio array into our queue
    audio_queue.put(indata.copy())


# ==========================================
# 3. MAIN TRANSLATION LOOP
# ==========================================
try:
    # Start capturing raw audio from your default system microphone
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BLOCK_SIZE):
        print("\n=======================================================")
        print("🎤 Live JP ➔ EN Translator Active!")
        print("🗣️ Speak Japanese into your microphone.")
        print("🛑 Press Ctrl+C in your terminal to stop.")
        print("=======================================================\n")
        
        while True:
            # Block and wait for the next 3-second audio block to be ready
            audio_chunk = audio_queue.get()
            
            # Flatten the 2D array from sounddevice into a 1D array for Whisper
            audio_data = audio_chunk.flatten().astype(np.float32)
            
            # Run translation on the audio block
            segments, info = model.transcribe(
                audio_data, 
                beam_size=5,
                task="translate",
                language="ja", 
                # --- ADD THESE SILENCE-FILTERING SETTINGS ---
                vad_filter=True,             # Voice Activity Detection (Filters out pure silence)
                vad_parameters=dict(min_silence_duration_ms=500), # Ignore gaps shorter than 0.5s
                no_speech_threshold=0.6,     # If confidence of "no speech" is above 60%, ignore it
                log_prob_threshold=-1.0      # If translation quality is suspiciously low, ignore it
            )
            # Print translations as soon as they are processed
            for segment in segments:
                # We filter out whisper artifacts (like empty spaces or quiet rooms transcribing as [Music])
                clean_text = segment.text.strip()
                if clean_text:
                    print(f"🇺🇸 [English translation]: {clean_text}")
                    print(f"DEBUG: Detected language: {info.language} (Probability: {info.language_probability:.2f})")

except KeyboardInterrupt:
    print("\n👋 Stopping live translator. Ja ne!")
except Exception as e:
    print(f"\n❌ An unexpected error occurred: {e}")
