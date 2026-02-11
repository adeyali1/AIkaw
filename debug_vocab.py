from transformers import Wav2Vec2Processor

MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
try:
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    vocab = processor.tokenizer.get_vocab()
    print(f"Vocab size: {len(vocab)}")
    
    # Check for diacritics
    diacritics = ["\u064b", "\u064c", "\u064d", "\u064e", "\u064f", "\u0650", "\u0651", "\u0652", "\u0670"]
    present = [d for d in diacritics if d in vocab]
    print(f"Diacritics present in vocab: {present}")
    
    # Check standard letters
    print(f"Sample vocab: {list(vocab.keys())[:20]}")
    
except Exception as e:
    print(e)
