import queue
import sys
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

# 1. Configuration
# We use "base" because it is fast enough for real-time. 
# Use "cpu" if you don't have a CUDA GPU configured yet.
MODEL_SIZE = "base" 
DEVICE = "cpu"       # Change to "cuda" once GPU/CUDA drivers are ready
COMPUTE_TYPE = "float32" # Change to "float16" if running on CUDA GPU

SAMPLE_RATE = 16000  # Whisper expects 16kHz audio
BLOCK_SIZE = int(16000*4)   # Check mic every X seconds (16000 * X)

# Thread-safe queue to hold audio chunks
audio_queue = queue.Queue()

# 2. Initialize Faster-Whisper
print(f"Loading Whisper model '{MODEL_SIZE}' on {DEVICE}...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded successfully!")

def audio_callback(indata, frames, time, status):
    """This function is called by sounddevice for every new audio block."""
    if status:
        print(status, file=sys.stderr)
    # Put a copy of the raw microphone audio into our queue
    audio_queue.put(indata.copy())

# 3. Main Loop
try:
    # Start recording from default microphone
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BLOCK_SIZE):
        print("\n🎤 Listening... Speak into your microphone. Press Ctrl+C to stop.\n")
        
        while True:
            # Get the next chunk of audio from the queue
            audio_chunk = audio_queue.get()
            
            # Flatten the 2D array from sounddevice to a 1D array for Whisper
            audio_data = audio_chunk.flatten().astype(np.float32)
            
            # Transcribe the chunk
            # 'beam_size=5' balances speed and accuracy
            segments, info = model.transcribe(audio_data, beam_size=5)
            
            for segment in segments:
                # Print transcription on the fly
                print(f"[{segment.start:.1f}s -> {segment.end:.1f}s] {segment.text}")

except KeyboardInterrupt:
    print("\nStopping live transcription. Goodbye!")
except Exception as e:
    print(f"\nAn error occurred: {e}")
