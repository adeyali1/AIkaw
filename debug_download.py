from transformers import pipeline
import torch

print("Starting debug download...")
try:
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device="cpu"
    )
    print("Download/Load Success!")
except Exception as e:
    print(f"Error: {e}")
