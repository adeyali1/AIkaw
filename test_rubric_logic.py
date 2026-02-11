import re
import difflib
import numpy as np

# --- Arabic Normalization & Utilities (Copied from app.py) ---
class ArabicUtils:
    @staticmethod
    def normalize(text):
        # Remove tatweel (kashida)
        text = re.sub(r'[\u0640]', '', text)
        # Normalize Alefs
        text = re.sub(r'[إأآ]', 'ا', text)
        text = re.sub(r'ة', 'ه', text)
        text = re.sub(r'ى', 'ي', text)
        return text

    @staticmethod
    def strip_diacritics(text):
        # Arabic diacritics range roughly 064B-0652 + 0670
        return re.sub(r'[\u064b-\u0652\u0670]', '', text)

# --- Grading Logic (Copied from app.py) ---
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
                elif base_sim > 0.85: 
                    status = "TASHKEEL"
                    confidence = 0.8
                    info = f"Read: {hyp_full}"
                elif base_sim > 0.6: 
                    status = "SUBSTITUTION" 
                    confidence = 0.6
                    info = f"Read: {hyp_full} (Sim: {int(base_sim*100)}%)"
                
                alignment.append({
                    "word": ref_full,
                    "status": status,
                    "info": info
                })
                i -= 1
                j -= 1
                
            elif direction == 2: # Up (Omission)
                alignment.append({
                    "word": ref_words[i-1]['text'],
                    "status": "OMISSION",
                    "info": "Not heard"
                })
                i -= 1
                
            elif direction == 3: # Left (Insertion)
                hyp_full = hyp_words[j-1]['word']
                alignment.append({
                    "word": hyp_full, 
                    "status": "INSERTION",
                    "info": "Extra word"
                })
                j -= 1
                
        alignment.reverse()
        return alignment

    def grade(self, reference_text, hypothesis_words):
        # 1. Prepare Data
        ref_norm_text = ArabicUtils.normalize(reference_text)
        ref_tokens = ref_norm_text.split()
        
        processed_refs = []
        for w in ref_tokens:
            processed_refs.append({
                'text': w,
                'base': ArabicUtils.strip_diacritics(w)
            })
            
        processed_hyps = []
        for w in hypothesis_words:
            norm = ArabicUtils.normalize(w['word'])
            processed_hyps.append({
                'word': w['word'],
                'base': ArabicUtils.strip_diacritics(norm),
                'start': w.get('start', 0),
                'end': w.get('end', 0)
            })
            
        # 2. Run DP Alignment
        raw_alignment = self.needleman_wunsch(processed_refs, processed_hyps)
        
        # 3. Post-Process for Repetition (Takrar)
        final_analysis = []
        for k, item in enumerate(raw_alignment):
            if item['status'] == "INSERTION":
                # Check previous word
                if k > 0:
                    prev_word = raw_alignment[k-1]['word']
                    # Compare bases to be safe
                    if ArabicUtils.strip_diacritics(item['word']) == ArabicUtils.strip_diacritics(prev_word):
                        item['status'] = "REPETITION"
                        item['info'] = "Repeated"
            
            final_analysis.append(item)
            
        return final_analysis

# --- Test Case ---
if __name__ == "__main__":
    grader = FluencyGrader()
    
    # Reference: "Raghiba Ra'idom Imtilaka Quwwatin"
    reference_text = "رَغِبَ رائِدٌ امْتلاكَ قُوَّةٍ"
    
    hypothesis_words = [
        {"word": "رَئِبَ", "start": 1.0, "end": 1.5},      # Substitution for رَغِبَ
        {"word": "رائِدٌ", "start": 1.6, "end": 2.0},       # Correct
        {"word": "امْتلاكَ", "start": 2.1, "end": 2.5},     # Correct
        {"word": "امْتلاكَ", "start": 2.6, "end": 3.0},     # REPETITION
        {"word": "قُوَّةٍ", "start": 3.1, "end": 3.5},      # Correct
        {"word": "زِيَادَة", "start": 3.6, "end": 4.0}      # INSERTION
    ]
    
    analysis = grader.grade(reference_text, hypothesis_words)
    
    with open("test_result.txt", "w", encoding="utf-8") as f:
        f.write(f"Reference: {reference_text}\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Word':<15} | {'Status':<15} | {'Info'}\n")
        f.write("-" * 50 + "\n")
        for item in analysis:
            f.write(f"{item['word']:<15} | {item['status']:<15} | {item['info']}\n")
            
    print("Test complete. Results written to test_result.txt")
