from transformers import pipeline
import torch

print("Debugging Whisper Load...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-small",
        device=device,
        chunk_length_s=30,
    )
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
