import gradio as gr
import torch
import torchaudio
import librosa
import numpy as np
import scipy
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import re
import difflib
import sys

# Windows requires explicit UTF-8 encoding for console output
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# --- Constants & Configuration ---
# Using whisper-small for better Arabic accuracy (2x better than tiny)
MODEL_ID = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEXT_1 = "رَغِبَ رائِدٌ امْتلاكَ قُوَّةٍ خارِقَةٍ كأبطالِ الأفلامِ. وَثَبَ عاليًا، وتَمْتَمَ بِألْفاظٍ غامِضَةٍ، لكْنَّ الواقِعَ لَمْ يُصْغِ لِرَغَباتِهِ، فَسَقَطَ وَتَحَطَّمَتْ آمالُهُ. وَبَيْنَما هو يفكِّرُ ويُصارِعُ شُعورَهُ أَنَّهُ ضَعيفٌ، أَقْبَلَتْ زينَةُ شاحِبَةً يَتْبَعُها ثامِرٌ مَذْعورًا، فَجَلَسَ رائِدٌ قُرْبَهُما يُصْغي إِلى ما حَدَثَ مَعَهُما. كان صَوْتُ ثامِرٍ يَرْتَجِفُ، لكِنْ شَيْئًا فَشَيْئًا، هَدَأَ واطمَأَنَّ. ثُمَّ شَكَتْ زينَةُ ضَياعَ مِحْفَظَتِها، في أثناء حَديثِها تَذَكَّرَتْ أَينَ وَضَعَتْها. في ذلك اليَوم، عَلِمَ رائِدٌ أنَّ قُوَّتَهُ الحَقيقيَّةَ؛ ليْسَتْ بالطَّيرانِ أو الكلِماتِ الغَريبَةِ، وإنَّما في قُدْرَتِهِ على الإِصْغاءِ وَمُسانَدةِ الأَصْدِقاءِ."
TEXT_2 = "بدأَتْ جَولتُنا مع المرشِدةِ السّياحيَّةِ وهي تقودُنا بحَماسٍ بينَ قاعاتِ المُتحَفِ العَريقِ. في القاعةِ الأُولى شاهدْنا مُجسَّماتٍ لحَيَواناتٍ مُنقرِضةٍ، وتوقَّفنا أمامَ مُجسَّمٍ لديناصورٍ ضخمٍ بدا كأنَّهُ يستيقظُ من نومٍ عميقٍ. ثمَّ دَخلنا قاعةَ الآثارِ، وتأمَّلنا أَوانيَ خزفيَّةً مُلوَّنةً بزَخارِفَ جميلةٍ، نُحِتَتْ منذُ قرونٍ. بعدَ ذلكَ توجَّهْنا إلى قِسمِ الفنونِ، فشدَّتني لوحةٌ لبحرٍ أَحسَسْتُ كأنَّ موجَهُ يقتربُ منّي. وفي نهايةِ الجولةِ تابعْنا فيلمًا على شاشةٍ كبيرةٍ يشرحُ كيفيَّةَ حِفظِ الآثارِ وحمايتِها منَ التَّلفِ والضَّياعِ. كانَتْ زيارةُ المُتحَفِ مُمتِعةً ومُفيدةً، فقَدْ شعَرْنا أنَّ كلَّ خُطوةٍ داخلَ المُتحَفِ تكشِفُ لنا معلومةً جديدةً، وتدفعُنا لرؤيةِ العالمِ وتاريخِهِ بمنظورٍ أوسعَ."

# --- Arabic Normalization & Utilities ---
class ArabicUtils:
    @staticmethod
    def normalize(text):
        # Remove tatweel (kashida)
        text = re.sub(r'[\u0640]', '', text)
        # Normalize Alefs
        text = re.sub(r'[إأآ]', 'ا', text)
        # Normalize Ta Marbuta to Ha (optional, but standardizes ending) - actually better to keep distinct?
        # Typically in reading, Ta Marbuta is pronounced 'h' in pause or 't' in connection.
        # Let's normalize ending ة to ه for robust comparison if desired, OR keep strict.
        # For general alignment, unifying is safer.
        text = re.sub(r'ة', 'ه', text)
        # Normalize Ya/Alef Maqsura (often swapped)
        text = re.sub(r'ى', 'ي', text)
        return text
    
    @staticmethod
    def remove_punctuation(text):
        # Remove common Arabic/English punctuation
        return re.sub(r'[،؛؟.,!:]', '', text)

    @staticmethod
    def strip_diacritics(text):
        # Arabic diacritics range roughly 064B-0652 + 0670
        return re.sub(r'[\u064b-\u0652\u0670]', '', text)

# --- ASR Pipeline ---
from transformers import pipeline

# --- Silero VAD for Accurate Pause Detection ---
from silero_vad import load_silero_vad, get_speech_timestamps

class VADPipeline:
    """Voice Activity Detection using Silero-VAD for accurate pause/hesitation detection."""
    def __init__(self):
        print("Loading Silero-VAD Model...")
        self.model = load_silero_vad(onnx=True)  # ONNX is faster on CPU
        print("Silero-VAD Model Loaded.")
    
    def detect_pauses(self, audio_path, threshold=1.0):
        """
        Detect significant pauses (gaps between speech segments).
        Returns list of pause intervals: [{'start': float, 'end': float, 'duration': float}, ...]
        """
        # Load audio at 16kHz (required by Silero-VAD)
        audio_array, sr = librosa.load(audio_path, sr=16000)
        audio_tensor = torch.from_numpy(audio_array).float()
        
        # Get speech timestamps
        speech_timestamps = get_speech_timestamps(
            audio_tensor, 
            self.model, 
            sampling_rate=16000,
            threshold=0.5,  # Speech probability threshold
            min_speech_duration_ms=100,
            min_silence_duration_ms=300  # Minimum pause to detect
        )
        
        pauses = []
        audio_duration = len(audio_array) / 16000
        
        # Find gaps between speech segments
        prev_end = 0.0
        for segment in speech_timestamps:
            seg_start = segment['start'] / 16000  # Convert samples to seconds
            seg_end = segment['end'] / 16000
            
            gap = seg_start - prev_end
            if gap >= threshold:
                pauses.append({
                    'start': prev_end,
                    'end': seg_start,
                    'duration': gap
                })
            prev_end = seg_end
        
        # Check for trailing silence
        if audio_duration - prev_end >= threshold:
            pauses.append({
                'start': prev_end,
                'end': audio_duration,
                'duration': audio_duration - prev_end
            })
        
        return pauses, speech_timestamps

class ASRPipeline:
    def __init__(self):
        print("Loading Whisper Model...")
        # efficient_v2 behavior: reliable word timestamps
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            device=DEVICE,
            return_timestamps="word",
            chunk_length_s=30,
        )
        print("Whisper Model Loaded.")

    def transcribe(self, audio_path):
        # Whisper pipeline handles loading/resampling internally usually, 
        # but passing the path directly is safest.
        
        # Load audio using librosa (uses soundfile/audioread) to avoid internal ffmpeg call by transformers
        # sr=16000 is required by Whisper/Wav2Vec2
        audio_array, _ = librosa.load(audio_path, sr=16000)
        
        # generate_kwargs={"language": "arabic"} forces Arabic
        result = self.pipe(audio_array, generate_kwargs={"language": "arabic"})
        
        # Whisper output with return_timestamps="word" usually looks like:
        # {'text': '...', 'chunks': [{'text': ' word', 'timestamp': (0.0, 0.5)}, ...]}
        
        words = []
        raw_transcript = result.get("text", "")
        chunks = result.get("chunks", [])
        
        for chunk in chunks:
            # Whisper chunks often include leading spaces
            w_text = chunk['text'].strip()
            if not w_text: 
                continue
                
            start, end = chunk['timestamp']
            
            # Handle None timestamps (rare but possible in some Whisper versions/edge cases)
            if start is None: start = 0.0
            if end is None: end = 0.0
            
            words.append({
                "word": w_text,
                "start": start,
                "end": end
            })
            
        return words, raw_transcript

# --- Grading Logic ---
class FluencyGrader:
    def __init__(self):
        pass

    def calculate_similarity(self, s1, s2):
        return difflib.SequenceMatcher(None, s1, s2).ratio()

    def needleman_wunsch(self, ref_words, hyp_words):
        n = len(ref_words)
        m = len(hyp_words)
        
        # Matrix initialization
        score = np.zeros((n + 1, m + 1))
        # Pointers: 1=Diag (Match/Sub), 2=Up (Delete/Omission), 3=Left (Insert)
        ptr = np.zeros((n + 1, m + 1), dtype=int)
        
        # Penalties (Tuned for Arabic Rubric)
        GAP_PENALTY = -2.0       # Cost of missing a word / inserting extra
        MISMATCH_PENALTY = -1.0  # Cost of swapping a word
        MATCH_BONUS = 3.0        # Reward for hitting the word
        
        # Initialize boundaries
        for i in range(n + 1):
            score[i][0] = i * GAP_PENALTY
            ptr[i][0] = 2 # Up
        for j in range(m + 1):
            score[0][j] = j * GAP_PENALTY
            ptr[0][j] = 3 # Left
        ptr[0][0] = 0
            
        # Fill
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                ref_w = ref_words[i-1]['base']
                hyp_w = hyp_words[j-1]['base']
                
                sim = self.calculate_similarity(ref_w, hyp_w)
                
                # Dynamic scoring
                if sim > 0.85: # Almost exact match
                    match_score = MATCH_BONUS
                elif sim > 0.6: # Fuzzy match
                    match_score = 1.0
                else:
                    match_score = MISMATCH_PENALTY
                
                diag = score[i-1][j-1] + match_score
                up = score[i-1][j] + GAP_PENALTY
                left = score[i][j-1] + GAP_PENALTY
                
                best = max(diag, up, left)
                score[i][j] = best
                
                if best == diag:
                    ptr[i][j] = 1
                elif best == up:
                    ptr[i][j] = 2
                else:
                    ptr[i][j] = 3
                    
        # Backward Trace
        alignment = []
        i, j = n, m
        
        while i > 0 or j > 0:
            direction = ptr[i][j]
            
            if direction == 1: # Match/Subst
                ref_full = ref_words[i-1]['text']
                hyp_full = hyp_words[j-1]['word']
                
                # Normalize for strict check
                ref_norm_full = ArabicUtils.normalize(ref_full)
                hyp_norm_full = ArabicUtils.normalize(hyp_full)
                
                base_sim = self.calculate_similarity(ref_words[i-1]['base'], hyp_words[j-1]['base'])
                
                status = "SUBSTITUTION"
                confidence = 0.0
                info = f"Read: {hyp_full}"
                
                if ref_norm_full == hyp_norm_full:
                    status = "CORRECT"
                    confidence = 1.0
                    info = ""
                # With Whisper (Unvocalized), Base Comparison is the primary metric for "Correctness".
                # If the base letters match perfectly, we count it as Correct to achieve high accuracy 
                # (ignoring the missing diacritics in the ASR output).
                elif base_sim > 0.95: 
                    status = "CORRECT" # Treated as correct even if vowels missing in ASR
                    confidence = 1.0
                    info = f"Read: {hyp_full} (Base Match)"
                elif base_sim > 0.60: 
                    status = "CORRECT" # Relaxed threshold for noise (e.g., Raqaid vs Raid)
                    confidence = 0.9 # Slightly lower confidence but still "Green/Correct" for scoring
                    info = f"Read: {hyp_full} (Approx)"
                elif base_sim > 0.5: 
                    status = "SUBSTITUTION" # Was MISPRONOUNCED
                    confidence = 0.6
                    info = f"Read: {hyp_full} (Sim: {int(base_sim*100)}%)"
                
                alignment.append({
                    "word": ref_full, # Display Reference Word
                    "status": status,
                    "confidence": confidence,
                    "start": hyp_words[j-1]['start'],
                    "end": hyp_words[j-1]['end'],
                    "info": info
                })
                i -= 1
                j -= 1
                
            elif direction == 2: # Up (Ommission)
                alignment.append({
                    "word": ref_words[i-1]['text'],
                    "status": "OMISSION",
                    "confidence": 0.0,
                    "info": "Not heard"
                })
                i -= 1
                
            elif direction == 3: # Left (Insertion)
                hyp_full = hyp_words[j-1]['word']
                alignment.append({
                    "word": hyp_full, # Display Spoken Word
                    "status": "INSERTION",
                    "confidence": 0.0,
                    "start": hyp_words[j-1]['start'],
                    "end": hyp_words[j-1]['end'],
                    "info": "Extra word"
                })
                j -= 1
                
        alignment.reverse()
        return alignment

    def split_merged_hyps(self, ref_tokens, hyp_words):
        """
        Splits hypothesis words that are likely merged reference bigrams.
        Example: "وثبعاليا" matches "وثَبَ" + "عاليًا"
        """
        # Create cleaned joined bigrams from ref (normalized & stripped)
        ref_bigrams = {}
        for i in range(len(ref_tokens) - 1):
            w1 = ref_tokens[i]
            w2 = ref_tokens[i+1]
            
            # Clean for matching
            c1 = ArabicUtils.remove_punctuation(ArabicUtils.strip_diacritics(ArabicUtils.normalize(w1)))
            c2 = ArabicUtils.remove_punctuation(ArabicUtils.strip_diacritics(ArabicUtils.normalize(w2)))
            
            combined = c1 + c2
            ref_bigrams[combined] = (w1, w2) 
            
        new_hyps = []
        for w in hyp_words:
            # Clean hyp for matching
            h_clean = ArabicUtils.remove_punctuation(ArabicUtils.strip_diacritics(ArabicUtils.normalize(w['word'])))
            
            if h_clean in ref_bigrams:
                # Found a merge!
                full_text = w['word']
                # Split roughly in half for display/timing purposes
                # We simply duplicate the entry but split the time, 
                # expecting Needleman-Wunsch to align them to the 2 reference words.
                # Actually, if we just split the time, the *text* remains merged "Wathabaliya".
                # NW will try to align "Wathabaliya" to "Wathaba" -> Match (prefix)? 
                # Then "Wathabaliya" to "Aliya" -> Mismatch.
                # We need to change the TEXT too.
                # Since we don't know exactly where to split index-wise in the raw string easily,
                # and we want "Correct" credits, let's use the Reference words tokens provided we found a match.
                # This ensures they get the Green 100%. 
                # It is pedagogically acceptable because they *did* say the words, just connectedly.
                
                parts = ref_bigrams[h_clean] # (w1, w2) from ref
                
                mid = w['start'] + (w['end'] - w['start']) / 2
                
                new_hyps.append({
                    'word': parts[0],  # Use clean ref word 1
                    'start': w['start'],
                    'end': mid
                })
                new_hyps.append({
                    'word': parts[1],  # Use clean ref word 2
                    'start': mid,
                    'end': w['end']
                })
            else:
                new_hyps.append(w)
                
        return new_hyps

    def grade(self, reference_text, hypothesis_words):
        # 1. Prepare Data
        ref_norm_text = ArabicUtils.normalize(reference_text)
        ref_tokens = ref_norm_text.split()
        
        # 1a. Filter Hallucinations (e.g. "Net" / "النت")
        # Whisper often outputs "النت" or "نت" as noise.
        # We remove them ONLY if they are NOT in the reference text.
        hallucinations = ["النت", "نت", "ال نت", "النت.", "ت.", "النت،"]
        filtered_hyps = []
        for w in hypothesis_words:
            # Check if this word is a known hallucination AND not in the reference
            w_norm = ArabicUtils.normalize(w['word'])
            # Also check if it contains the substring "النت" which is very specific noise
            is_noise = (w_norm in hallucinations) or ("النت" in w_norm and len(w_norm) < 7)
            
            if is_noise and w_norm not in ref_norm_text:
                continue # Skip this word (it's noise)
            filtered_hyps.append(w)
        hypothesis_words = filtered_hyps
        
        # 1b. Pre-process Hypothesis for Merged Words
        hypothesis_words = self.split_merged_hyps(ref_tokens, hypothesis_words)
        
        processed_refs = []
        for w in ref_tokens:
            # Base = Normalized (Alefs) + Stripped Diacritics + Removed Punctuation
            base_clean = ArabicUtils.remove_punctuation(ArabicUtils.strip_diacritics(w))
            processed_refs.append({
                'text': w,
                'base': base_clean
            })
            
        processed_hyps = []
        for w in hypothesis_words:
            norm = ArabicUtils.normalize(w['word'])
            # Base = Normalized (Alefs) + Stripped Diacritics + Removed Punctuation
            base_clean = ArabicUtils.remove_punctuation(ArabicUtils.strip_diacritics(norm))
            
            processed_hyps.append({
                'word': w['word'],
                'base': base_clean,
                'start': w['start'],
                'end': w['end']
            })
            
        # 2. Run DP Alignment
        raw_alignment = self.needleman_wunsch(processed_refs, processed_hyps)
        
        # 3. Post-Process for Repetition (Takrar)
        final_analysis = []
        for k, item in enumerate(raw_alignment):
            # Check Repetition
            if item['status'] == "INSERTION":
                # Check previous word
                if k > 0:
                    prev_word = raw_alignment[k-1]['word']
                    # Compare bases to be safe
                    if ArabicUtils.strip_diacritics(item['word']) == ArabicUtils.strip_diacritics(prev_word):
                        item['status'] = "REPETITION"
                        item['info'] = "Repeated"
            
            final_analysis.append(item)

        # 4. Self-Correction Detection (Improved with Phonetic Similarity)
        # NOTE: Hesitation is now handled by VAD in analyze_reading(), not here.
        processed = []
        for k, item in enumerate(final_analysis):
            item['hesitation'] = False  # Will be set by VAD in analyze_reading()
            
            # SELF-CORRECTION Logic (Improved)
            # Case 1: INSERTION followed by CORRECT (failed attempt then correction)
            # Case 2: INSERTION similar to PREVIOUS CORRECT (hesitant repeat/stumble)
            if item['status'] == "INSERTION":
                inserted_base = ArabicUtils.strip_diacritics(ArabicUtils.normalize(item['word']))
                
                # Check NEXT word (original logic)
                if k + 1 < len(final_analysis):
                    next_item = final_analysis[k+1]
                    if next_item['status'] == "CORRECT":
                        correct_base = ArabicUtils.strip_diacritics(ArabicUtils.normalize(next_item['word']))
                        sim = difflib.SequenceMatcher(None, inserted_base, correct_base).ratio()
                        
                        if sim > 0.3 or len(inserted_base) <= 3:
                            item['status'] = "SELF_CORRECTION"
                            item['info'] = f"Self-Corrected (Next: {int(sim*100)}%)"
                
                # Check PREVIOUS word (catches cases like "وطعيف" after "ضعيف")
                if item['status'] == "INSERTION" and k > 0:  # Still insertion, check prev
                    prev_item = final_analysis[k-1]
                    if prev_item['status'] == "CORRECT":
                        prev_base = ArabicUtils.strip_diacritics(ArabicUtils.normalize(prev_item['word']))
                        sim = difflib.SequenceMatcher(None, inserted_base, prev_base).ratio()
                        
                        # Higher threshold for prev match (0.5) since this is a "fumbled repeat"
                        if sim > 0.4:
                            item['status'] = "SELF_CORRECTION"
                            item['info'] = f"Self-Corrected (Prev: {int(sim*100)}%)"

            processed.append(item)
            
        return processed

# --- App Integration ---
aligner_pipeline = ASRPipeline()
vad_pipeline = VADPipeline()  # NEW: VAD for accurate pause detection
grader = FluencyGrader()

def analyze_reading(text_option, audio_file, mic_file):
    input_audio = mic_file if mic_file else audio_file
    if not input_audio: return "No audio", {}
    
    try:
        # Step 1: ASR Transcription with word-level timestamps
        hyp_words, raw_transcript = aligner_pipeline.transcribe(input_audio)
        
        # Step 2: Grading (alignment + self-correction detection)
        analysis = grader.grade(text_option, hyp_words)
        
        # Step 3: IMPROVED Hesitation Detection using GAP analysis
        # Hesitation = LONG GAP before a word (indicates pause/struggling)
        # Also mark words before self-corrections
        
        # Calculate average gap between words for baseline
        gaps = []
        for i in range(1, len(analysis)):
            if 'start' in analysis[i] and 'end' in analysis[i-1]:
                gap = analysis[i]['start'] - analysis[i-1]['end']
                if gap > 0:
                    gaps.append(gap)
        
        avg_gap = sum(gaps) / len(gaps) if gaps else 0.2
        # Hesitation threshold: gap > 0.6s OR gap > 3x average (whichever is higher)
        gap_threshold = max(0.6, avg_gap * 3)
        
        print(f"[HESITATION DEBUG] Avg gap: {avg_gap:.2f}s, Threshold: {gap_threshold:.2f}s")
        
        hesitation_count = 0
        for idx, word in enumerate(analysis):
            word['hesitation'] = False
            
            # Method 1: GAP-based hesitation
            # If there's a significant gap BEFORE this word, mark as hesitation
            if idx > 0 and 'start' in word and 'end' in analysis[idx-1]:
                gap_before = word['start'] - analysis[idx-1]['end']
                if gap_before > gap_threshold:
                    word['hesitation'] = True
                    print(f"[HESITATION] Gap: '{word['word']}' has {gap_before:.2f}s gap before it")
            
            # Method 2: Pre-self-correction hesitation
            # The word BEFORE a self-correction was the difficult one
            if idx + 1 < len(analysis):
                next_word = analysis[idx + 1]
                if next_word.get('status') == 'SELF_CORRECTION':
                    word['hesitation'] = True
                    print(f"[HESITATION] Pre-correction: '{word['word']}' before '{next_word['word']}'")
            
            if word['hesitation']:
                hesitation_count += 1
        
        print(f"[HESITATION DEBUG] Total hesitations: {hesitation_count}")
        
        # Calc Accuracy (Partial Credit)
        score_sum = 0
        total_words = 0
        
        for w in analysis:
            if w['status'] == "INSERTION" or w['status'] == "REPETITION":
                continue # Valid words didn't change count, but errors might penalize?
                # Usually Accuracy = Correct / Total Reference Words
            
            total_words += 1
            if w['status'] == "CORRECT": score_sum += 1.0
            elif w['status'] == "TASHKEEL": score_sum += 0.8
            elif w['status'] == "SUBSTITUTION": score_sum += 0.0 # Strict substitution = 0? Or partial? Rubric usually 0.
            # Keeping mispronounced logic if sim is high? 
            # In needleman_wunsch: > 0.6 is SUBSTITUTION. 
            # Let's give partial credit (0.5) if it was a "Close Substitution"
            if w['status'] == "SUBSTITUTION":
                 if "Sim:" in w['info']: score_sum += 0.5 # Give credit for close try
            
        accuracy = (score_sum / total_words * 100) if total_words > 0 else 0
        
        html = generate_html(analysis, accuracy)
        return html, {"accuracy": f"{accuracy:.1f}%", "raw_transcript": raw_transcript, "word_analysis": analysis}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}", {"error": str(e)}

def generate_html(analysis, accuracy):
    html = f"""
    <div style="font-family: 'Arial'; direction: rtl; text-align: right; font-size: 20px; line-height: 2.5; background: #f9f9f9; padding: 20px; border-radius: 10px;">
        <h3>نتائج التحليل - تركيز على التردد والتصحيح الذاتي</h3>
        <p><strong>Focus:</strong> Hesitation (تردد) = Pink | Self-Correction (التصحيح الذاتي) = Orange</p>
        <hr>
        <p>
    """
    
    for word in analysis:
        text = word['word']
        status = word['status']
        hesitation = word.get('hesitation', False)
        
        # SIMPLIFIED: Only color Hesitation and Self-Correction
        # All other words are plain black
        color = "black"
        bg = "transparent"
        tooltip = ""
        
        # SELF-CORRECTION: Orange
        if status == "SELF_CORRECTION":
            color = "#E64A19"  # Dark Orange
            bg = "#FFCCBC"     # Peach
            tooltip = "(التصحيح الذاتي - Self Correction)"
        
        # HESITATION: Pink (overrides everything)
        if hesitation:
            color = "#880E4F"  # Deep Pink/Purple
            bg = "#F8BBD0"     # Light Pink
            tooltip = "(تردد - Hesitation)"            
        html += f'<span title="{tooltip}" style="color: {color}; background-color: {bg}; padding: 2px 5px; border-radius: 4px;">{text}</span> '
        
    html += "</p></div>"
    html += """
    <div style="font-size: 14px; margin_top: 15px; direction: rtl; display: flex; gap: 10px; flex-wrap: wrap;">
        <span style="background: #C8E6C9; padding: 4px 8px; border-radius: 4px;">✅ صحيح</span>
        <span style="background: #FFF9C4; padding: 4px 8px; border-radius: 4px;">➕ إدخال</span>
        <span style="background: #FFCCBC; padding: 4px 8px; border-radius: 4px;">❌ حذف</span>
        <span style="background: #BBDEFB; padding: 4px 8px; border-radius: 4px;">🔄 استبدال</span>
        <span style="background: #FFCDD2; padding: 4px 8px; border-radius: 4px;">⚠️ تشكيل</span>
        <span style="background: #B3E5FC; padding: 4px 8px; border-radius: 4px;">🔁 تكرار</span>
        <span style="background: #FFCCBC; color: #E64A19; padding: 4px 8px; border-radius: 4px;">وا تصحيح ذاتي</span>
        <span style="background: #E91E63; color: white; padding: 4px 8px; border-radius: 4px;">⏸️ تردد</span>
    </div>
    """
    return html

# --- Interface ---
custom_css = """
.gradio-container {font-family: 'IBM Plex Sans Arabic', sans-serif;}
"""

with gr.Blocks(css=custom_css, title="Arabic Reading Fluency POC") as demo:
    gr.Markdown("# 🎙️ Arabic Reading Fluency Assessment")
    gr.Markdown("Select a text, read it aloud, and get instant pedagogical feedback.")
    
    with gr.Row():
        with gr.Column():
            text_dropdown = gr.Dropdown(
                choices=[TEXT_1, TEXT_2],
                value=TEXT_1,
                label="Select Text (نص القراءة)",
                interactive=True
            )
            text_display = gr.Textbox(label="Text to Read", value=TEXT_1, lines=5, text_align="right", interactive=False)
            
            def update_text(val):
                return val
            
            text_dropdown.change(update_text, inputs=text_dropdown, outputs=text_display)
            
            with gr.Tab("Upload"):
                audio_file = gr.Audio(sources=["upload"], type="filepath", label="Upload Audio")
            with gr.Tab("Microphone"):
                mic_file = gr.Audio(sources=["microphone"], type="filepath", label="Record Audio")
            
            analyze_btn = gr.Button("🔍 Analyze Pronunciation", variant="primary")
            
        with gr.Column():
            html_output = gr.HTML(label="Visual Report")
            json_output = gr.JSON(label="Detailed Debug Data")
            
    analyze_btn.click(
        analyze_reading,
        inputs=[text_dropdown, audio_file, mic_file],
        outputs=[html_output, json_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
