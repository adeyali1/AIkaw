"""
Arabic Forced Alignment using Wav2Vec2
Uses jonatasgrosman/wav2vec2-large-xlsr-53-arabic for precise word alignment
"""

import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np

class ArabicForcedAligner:
    """
    Forced alignment for Arabic using Wav2Vec2.
    Provides precise word-level timestamps for hesitation detection.
    """
    
    def __init__(self, model_name="jonatasgrosman/wav2vec2-large-xlsr-53-arabic"):
        print(f"[CTC Aligner] Loading Arabic Wav2Vec2 model: {model_name}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and processor
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.sample_rate = 16000
        print(f"[CTC Aligner] Model loaded on {self.device}")
    
    def load_audio(self, audio_path):
        """Load and resample audio to 16kHz mono using librosa"""
        import librosa
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return waveform
    
    def transcribe_with_timestamps(self, audio_path):
        """
        Transcribe audio and get word-level timestamps.
        Returns list of words with start/end times and confidence.
        """
        # Load audio
        waveform = self.load_audio(audio_path)
        audio_length_seconds = len(waveform) / self.sample_rate
        
        # Process audio (waveform is already numpy array from librosa)
        inputs = self.processor(
            waveform,  # Already numpy, no need for .numpy()
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
        
        # Get probabilities and predicted IDs
        probs = torch.softmax(logits, dim=-1)
        predicted_ids = torch.argmax(logits, dim=-1)
        
        # Decode to text
        transcription = self.processor.decode(predicted_ids[0])
        
        # Get frame-level predictions for timestamp calculation
        # Each frame represents ~20ms of audio
        frame_duration = audio_length_seconds / logits.shape[1]
        
        # Extract word timestamps by finding character boundaries
        words_with_timestamps = self._extract_word_timestamps(
            predicted_ids[0].cpu().numpy(),
            probs[0].cpu().numpy(),
            frame_duration
        )
        
        return words_with_timestamps, transcription
    
    def _extract_word_timestamps(self, predicted_ids, probs, frame_duration):
        """
        Extract word-level timestamps from frame-level predictions.
        """
        words = []
        current_word = ""
        word_start_frame = None
        word_probs = []
        
        blank_id = self.processor.tokenizer.pad_token_id
        
        prev_id = None
        for frame_idx, token_id in enumerate(predicted_ids):
            # Skip blanks and repeated tokens (CTC decoding)
            if token_id == blank_id:
                if current_word:
                    # End of word
                    words.append({
                        'word': current_word,
                        'start': word_start_frame * frame_duration,
                        'end': frame_idx * frame_duration,
                        'confidence': float(np.mean(word_probs)) if word_probs else 0.0
                    })
                    current_word = ""
                    word_start_frame = None
                    word_probs = []
                prev_id = token_id
                continue
            
            if token_id == prev_id:
                # Repeated token - extend duration but don't add character
                word_probs.append(float(probs[frame_idx, token_id]))
                prev_id = token_id
                continue
            
            # New character
            char = self.processor.decode([token_id])
            
            # Check if this starts a new word (space or first char)
            if char == ' ' or char == '':
                if current_word:
                    words.append({
                        'word': current_word,
                        'start': word_start_frame * frame_duration,
                        'end': frame_idx * frame_duration,
                        'confidence': float(np.mean(word_probs)) if word_probs else 0.0
                    })
                    current_word = ""
                    word_start_frame = None
                    word_probs = []
            else:
                if word_start_frame is None:
                    word_start_frame = frame_idx
                current_word += char
                word_probs.append(float(probs[frame_idx, token_id]))
            
            prev_id = token_id
        
        # Don't forget last word
        if current_word:
            words.append({
                'word': current_word,
                'start': word_start_frame * frame_duration,
                'end': len(predicted_ids) * frame_duration,
                'confidence': float(np.mean(word_probs)) if word_probs else 0.0
            })
        
        return words


def detect_hesitations_from_alignment(words, gap_threshold=0.8, duration_threshold=1.5):
    """
    Detect hesitations based on word alignment data.
    
    Args:
        words: List of word dicts with 'start', 'end', 'word', 'confidence'
        gap_threshold: Gap before word in seconds to mark as hesitation
        duration_threshold: Word duration in seconds to mark as hesitation
    
    Returns:
        List of word dicts with 'hesitation' flag added
    """
    if not words:
        return words
    
    # Calculate average word duration
    durations = [w['end'] - w['start'] for w in words]
    avg_duration = sum(durations) / len(durations) if durations else 0.5
    
    print(f"[CTC Hesitation] Average word duration: {avg_duration:.2f}s")
    
    for i, word in enumerate(words):
        word['hesitation'] = False
        word_duration = word['end'] - word['start']
        
        # Check gap before this word
        if i > 0:
            gap = word['start'] - words[i-1]['end']
            if gap > gap_threshold:
                word['hesitation'] = True
                print(f"[CTC Hesitation] Gap: '{word['word']}' has {gap:.2f}s gap before it")
        
        # Check if word is unusually long (>2x average or > threshold)
        if word_duration > duration_threshold or word_duration > avg_duration * 2.5:
            word['hesitation'] = True
            print(f"[CTC Hesitation] Long: '{word['word']}' took {word_duration:.2f}s")
    
    return words


# Test function
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python arabic_aligner.py <audio_file.wav>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    print(f"Testing with: {audio_file}")
    
    # Initialize aligner
    aligner = ArabicForcedAligner()
    
    # Get word timestamps
    words, transcript = aligner.transcribe_with_timestamps(audio_file)
    
    print(f"\nTranscript: {transcript}")
    print(f"\nWords with timestamps:")
    for w in words:
        print(f"  {w['start']:.2f}s - {w['end']:.2f}s: '{w['word']}' (conf: {w['confidence']:.2f})")
    
    # Detect hesitations
    words = detect_hesitations_from_alignment(words)
    
    hesitations = [w for w in words if w['hesitation']]
    print(f"\nDetected {len(hesitations)} hesitations:")
    for w in hesitations:
        print(f"  '{w['word']}' at {w['start']:.2f}s")
