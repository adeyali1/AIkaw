import torch
from transformers import Wav2Vec2Processor
import re

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

text = "مرحبا بكم"
# Test tokenizer output
inputs = processor.tokenizer(text, return_tensors="pt")
ids = inputs.input_ids[0].tolist()
tokens = processor.tokenizer.convert_ids_to_tokens(ids)

print("Text:", text)
print("IDs:", ids)
print("Tokens:", tokens)

# Check for BOS/EOS
vocab = processor.tokenizer.get_vocab()
print("BOS ID:", vocab.get("<s>"))
print("EOS ID:", vocab.get("</s>"))
print("PAD ID:", vocab.get("<pad>"))

# simulate strictness
cleaned_text = re.sub(r'[\u064b-\u0652\u0670]', '', text)
print("Cleaned text:", cleaned_text)
inputs_clean = processor.tokenizer(cleaned_text, return_tensors="pt")
print("Clean IDs:", inputs_clean.input_ids[0].tolist())
