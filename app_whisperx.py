"""
Arabic Reading Fluency Assessment - WhisperX Version
Uses WhisperX with Arabic Wav2Vec2 forced alignment for maximum accuracy
"""

import gradio as gr
import torch
import whisperx
import gc
import os
import sys

# Windows UTF-8 fix
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Ensure ffmpeg is in PATH
import imageio_ffmpeg
ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["PATH"] += os.pathsep + os.path.dirname(ffmpeg_path)
print(f"[Setup] Added ffmpeg to PATH: {os.path.dirname(ffmpeg_path)}")

# --- Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
BATCH_SIZE = 16 if DEVICE == "cuda" else 4

# Use large-v3 for best Arabic accuracy (now downloaded!)
# Options: tiny, base, small, medium, large-v2, large-v3
WHISPER_MODEL = "large-v3"

# Arabic reference text
REFERENCE_TEXT = """رَغِبَ رائِدٌ امْتلاكَ قُوَّةٍ خارِقَةٍ كأبطالِ الأفلامِ. 
وَثَبَ عاليًا، وتَمْتَمَ بِألْفاظٍ غامِضَةٍ، 
لكْنَّ الواقِعَ لَمْ يُصْغِ لِرَغَباتِهِ، فَسَقَطَ وَتَحَطَّمَتْ آمالُهُ. 
وَبَيْنَما هو يفكِّرُ ويُصارِعُ شُعورَهُ أَنَّهُ ضَعيفٌ، 
أَقْبَلَتْ زينَةُ شاحِبَةً يَتْبَعُها ثامِرٌ مَذْعورًا، 
فَجَلَسَ رائِدٌ قُرْبَهُما يُصْغي إِلى ما حَدَثَ مَعَهُما. 
كان صَوْتُ ثامِرٍ يَرْتَجِفُ، لكِنْ شَيْئًا فَشَيْئًا، هَدَأَ واطمَأَنَّ. 
ثُمَّ شَكَتْ زينَةُ ضَياعَ مِحْفَظَتِها، في أثناء حَديثِها تَذَكَّرَتْ أَينَ وَضَعَتْها. 
في ذلك اليَوم, عَلِمَ رائِدٌ أنَّ قُوَّتَهُ الحَقيقيَّةَ؛ 
ليْسَتْ بالطَّيرانِ أو الكلِماتِ الغَريبَةِ، 
وإنَّما في قُدْرَتِهِ على الإِصْغاءِ وَمُسانَدةِ الأَصْدِقاءِ."""


print(f"[WhisperX] Loading model: {WHISPER_MODEL} on {DEVICE}")
print(f"[WhisperX] This may take a few minutes for the first run...")

# Load WhisperX model (using silero VAD to avoid PyTorch 2.6 pyannote issue)
model = whisperx.load_model(
    WHISPER_MODEL, 
    DEVICE, 
    compute_type=COMPUTE_TYPE,
    language="ar",
    vad_method="silero"  # Use silero instead of pyannote (PyTorch 2.6 compatible)
)
print("[WhisperX] Model loaded successfully!")


def normalize_arabic(text):
    """Normalize Arabic text for comparison"""
    import re
    # Remove tashkeel (diacritics)
    tashkeel = re.compile(r'[\u064B-\u065F\u0670]')
    text = tashkeel.sub('', text)
    # Normalize alef variants
    text = re.sub('[إأآا]', 'ا', text)
    # Normalize taa marbuta
    text = re.sub('ة', 'ه', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def get_similarity(word1, word2):
    """Calculate similarity between two Arabic words"""
    from difflib import SequenceMatcher
    w1 = normalize_arabic(word1)
    w2 = normalize_arabic(word2)
    if w1 == w2:
        return 1.0
    return SequenceMatcher(None, w1, w2).ratio()


def analyze_reading_whisperx(audio_path, reference_text):
    """
    Analyze reading using WhisperX with forced alignment
    """
    print(f"\n[WhisperX] Processing audio: {audio_path}")
    
    # Load audio
    audio = whisperx.load_audio(audio_path)
    
    # Step 1: Transcribe with WhisperX
    print("[WhisperX] Step 1: Transcribing...")
    result = model.transcribe(audio, batch_size=BATCH_SIZE, language="ar")
    
    # Step 2: Align with Arabic Wav2Vec2
    print("[WhisperX] Step 2: Forced alignment with Arabic Wav2Vec2...")
    model_a, metadata = whisperx.load_align_model(
        language_code="ar", 
        device=DEVICE
    )
    result = whisperx.align(
        result["segments"], 
        model_a, 
        metadata, 
        audio, 
        DEVICE,
        return_char_alignments=False
    )
    
    # Clean up alignment model to save memory
    del model_a
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    # Extract words with timestamps
    hyp_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                hyp_words.append({
                    "word": word_info["word"],
                    "start": word_info.get("start", 0),
                    "end": word_info.get("end", 0),
                    "confidence": word_info.get("score", 1.0)
                })
    
    raw_transcript = " ".join([w["word"] for w in hyp_words])
    print(f"[WhisperX] Transcript: {raw_transcript[:100]}...")
    print(f"[WhisperX] Found {len(hyp_words)} words with timestamps")
    
    # Step 3: Align with reference text
    print("[WhisperX] Step 3: Aligning with reference text...")
    analysis = align_with_reference(hyp_words, reference_text)
    
    # Step 4: Detect hesitations
    print("[WhisperX] Step 4: Detecting hesitations...")
    analysis = detect_hesitations(analysis)
    
    return analysis, raw_transcript


def align_with_reference(hyp_words, reference_text):
    """
    Align hypothesis words with reference text using Needleman-Wunsch
    """
    import re
    
    # Tokenize reference (split by space to preserve punctuation for hesitation detection)
    ref_words = reference_text.split()
    
    # Build alignment using dynamic programming
    n, m = len(ref_words), len(hyp_words)
    
    # Scoring
    MATCH = 2
    MISMATCH = -1
    GAP = -2
    
    # DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * GAP
    for j in range(m + 1):
        dp[0][j] = j * GAP
    
    # Fill matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = get_similarity(ref_words[i-1], hyp_words[j-1]["word"])
            match_score = MATCH if sim > 0.6 else (MISMATCH if sim < 0.3 else 0)
            
            dp[i][j] = max(
                dp[i-1][j-1] + match_score,  # Match/Mismatch
                dp[i-1][j] + GAP,             # Deletion (omission)
                dp[i][j-1] + GAP              # Insertion
            )
    
    # Traceback
    analysis = []
    i, j = n, m
    
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            sim = get_similarity(ref_words[i-1], hyp_words[j-1]["word"])
            match_score = MATCH if sim > 0.6 else (MISMATCH if sim < 0.3 else 0)
            
            if dp[i][j] == dp[i-1][j-1] + match_score:
                # Match or substitution
                if sim >= 0.85:
                    status = "CORRECT"
                else:
                    status = "SUBSTITUTION"
                
                analysis.append({
                    "word": ref_words[i-1],
                    "status": status,
                    "confidence": sim,
                    "start": hyp_words[j-1]["start"],
                    "end": hyp_words[j-1]["end"],
                    "read_as": hyp_words[j-1]["word"],
                    "info": f"Read: {hyp_words[j-1]['word']} (Sim: {sim*100:.0f}%)" if sim < 1.0 else ""
                })
                i -= 1
                j -= 1
                continue
        
        if i > 0 and (j == 0 or dp[i][j] == dp[i-1][j] + GAP):
            # Omission
            analysis.append({
                "word": ref_words[i-1],
                "status": "OMISSION",
                "confidence": 0,
                "info": "Not read"
            })
            i -= 1
        elif j > 0:
            # Insertion
            # Check if it's a self-correction
            is_self_correction = False
            if analysis and len(analysis) > 0:
                prev = analysis[-1]
                sim_prev = get_similarity(hyp_words[j-1]["word"], prev["word"])
                if sim_prev > 0.4:
                    is_self_correction = True
            
            if is_self_correction:
                analysis.append({
                    "word": hyp_words[j-1]["word"],
                    "status": "SELF_CORRECTION",
                    "confidence": 0,
                    "start": hyp_words[j-1]["start"],
                    "end": hyp_words[j-1]["end"],
                    "info": "Self-correction attempt"
                })
            else:
                analysis.append({
                    "word": hyp_words[j-1]["word"],
                    "status": "INSERTION",
                    "confidence": 0,
                    "start": hyp_words[j-1]["start"],
                    "end": hyp_words[j-1]["end"],
                    "info": "Extra word"
                })
            j -= 1
    
    # Reverse since we built it backwards
    analysis.reverse()
    
    return analysis


def detect_hesitations(analysis):
    """
    Detect hesitations and infer self-corrections based on:
    1. Abnormally long word duration (student struggling/retrying a word)
    2. Long gaps BEFORE a word (pause before reading)
    3. Long gaps AFTER a word (pause after reading, uncertainty)
    4. Words before explicit self-corrections
    5. Inferred self-corrections from extremely long word durations
       (when ASR merges a failed attempt + correction into one word)
    """
    # Calculate statistics
    durations = []
    gaps = []
    for i in range(len(analysis)):
        if 'start' in analysis[i] and 'end' in analysis[i]:
            dur = analysis[i]['end'] - analysis[i]['start']
            if dur > 0:
                durations.append((dur, i))
        if i > 0 and 'start' in analysis[i] and 'end' in analysis[i-1]:
            gap = analysis[i]['start'] - analysis[i-1]['end']
            if gap > 0:
                gaps.append(gap)
    
    # Use median for robust thresholds
    dur_values = [d[0] for d in durations]
    sorted_dur = sorted(dur_values)
    median_duration = sorted_dur[len(sorted_dur) // 2] if sorted_dur else 0.5
    avg_gap = sum(gaps) / len(gaps) if gaps else 0.2
    
    # Base thresholds - increased to reduce false positives
    base_duration_hesitation = max(1.2, median_duration * 3.0)   # Raised to 3.0x
    base_duration_self_corr = max(2.5, median_duration * 5.0)    # Raised to 5.0x
    base_gap_threshold = max(0.85, avg_gap * 2.0)                # Raised minimum to 0.85s
    
    print(f"[Detection] Median word duration: {median_duration:.2f}s")
    print(f"[Detection] Base Hesitation threshold: {base_duration_hesitation:.2f}s")
    print(f"[Detection] Base Self-correction threshold: {base_duration_self_corr:.2f}s")
    print(f"[Detection] Avg gap: {avg_gap:.2f}s, Base Gap threshold: {base_gap_threshold:.2f}s")
    
    hesitation_count = 0
    self_correction_count = 0
    
    prev_was_problematic = False # Track if previous word had issues
    
    for idx, word in enumerate(analysis):
        word['hesitation'] = False
        
        # Punctuation check - allow finding natural pauses
        word_text = word['word']
        has_punctuation = word_text.endswith(('،', '.', ':', '؛', '!', '؟', ','))
        
        # Dynamic thresholds
        current_gap_threshold = base_gap_threshold
        current_duration_threshold = base_duration_hesitation
        
        # 1. Punctuation Logic: Allow longer gaps after punctuation
        if has_punctuation:
            current_gap_threshold = max(current_gap_threshold, 1.5) # Allow 1.5s natural pause
            
        # 2. Ripple Effect: if previous word was problematic, be more sensitive now
        if prev_was_problematic:
            current_duration_threshold = current_duration_threshold * 0.8
            # Only lower gap threshold if NO punctuation (don't punish recovery pause)
            if not has_punctuation:
                current_gap_threshold = current_gap_threshold * 0.8
        
        # 3. Post-Correction Recovery: If previous word was a SELF-CORRECTION, fit is highly likely to be a hesitation
        # This catches cases like "Wa-tam-tama.... bi-alfadh" where the struggle continues
        if idx > 0 and analysis[idx-1].get('status') == 'SELF_CORRECTION':
            current_duration_threshold = median_duration * 1.2 # Aggressive: just slightly above median
            current_gap_threshold = 0.2 # Aggressive: any gap is suspicious
            print(f"[Hesitation] Aggressive threshold for '{word_text}' due to prev correction")
            
        is_problematic = False
        
        word_duration = 0
        if 'start' in word and 'end' in word:
            word_duration = word['end'] - word['start']
        
        # === Method 1: Word duration analysis ===
        if word_duration > 0:
            # Extremely long word → likely contains a self-correction attempt
            if word_duration > base_duration_self_corr:
                word['hesitation'] = True
                # Mark as self-correction if it's currently CORRECT
                if word.get('status') == 'CORRECT':
                    word['status'] = 'SELF_CORRECTION'
                    word['info'] = f'تصحيح ذاتي (مدة: {word_duration:.1f}ث)'
                    self_correction_count += 1
                    print(f"[Self-Correction] Inferred: '{word['word']}' took {word_duration:.2f}s")
                else:
                    hesitation_count += 1
                    print(f"[Hesitation] Very slow: '{word['word']}' took {word_duration:.2f}s")
                is_problematic = True
                prev_was_problematic = True
                continue
            
            # Moderately long word → hesitation
            elif word_duration > current_duration_threshold:
                word['hesitation'] = True
                hesitation_count += 1
                is_problematic = True
                print(f"[Hesitation] Slow word: '{word['word']}' took {word_duration:.2f}s (>{current_duration_threshold:.2f}s)")
                
        # === Method 2: Gap BEFORE word ===
        if idx > 0 and 'start' in word and 'end' in analysis[idx-1]:
            gap_before = word['start'] - analysis[idx-1]['end']
            
            # Check punctuation of PREVIOUS word for "Gap Before"
            prev_word_text = analysis[idx-1]['word']
            prev_has_punct = prev_word_text.endswith(('،', '.', ':', '؛', '!', '؟', ','))
            before_gap_threshold = base_gap_threshold
            if prev_has_punct:
                before_gap_threshold = max(before_gap_threshold, 1.5)
            
            if gap_before > before_gap_threshold:
                word['hesitation'] = True
                if not is_problematic: 
                    hesitation_count += 1
                is_problematic = True
                print(f"[Hesitation] Gap before: '{word['word']}' has {gap_before:.2f}s gap (Threshold: {before_gap_threshold:.2f}s)")
        
        # === Method 3: Gap AFTER word (hesitation on the word just read) ===
        if idx + 1 < len(analysis) and 'end' in word and 'start' in analysis[idx+1]:
            gap_after = analysis[idx+1]['start'] - word['end']
            if gap_after > current_gap_threshold:
                word['hesitation'] = True
                if not is_problematic:
                    hesitation_count += 1
                is_problematic = True
                print(f"[Hesitation] Gap after: '{word['word']}' has {gap_after:.2f}s gap after it (Threshold: {current_gap_threshold:.2f}s)")
        
        # === Method 4: Explicit Self-Correction ===
        if idx + 1 < len(analysis):
            if analysis[idx + 1].get('status') == 'SELF_CORRECTION':
                word['hesitation'] = True
                if not is_problematic:
                    hesitation_count += 1
                is_problematic = True
                print(f"[Hesitation] Pre-correction: '{word['word']}'")
        
        prev_was_problematic = is_problematic
    
    print(f"[Detection] Hesitations: {hesitation_count}, Self-corrections inferred: {self_correction_count}")
    return analysis


def calculate_stats(analysis):
    """Calculate reading accuracy and detailed stats"""
    total_ref = 0
    correct = 0
    substitutions = 0
    omissions = 0
    insertions = 0
    self_corrections = 0
    hesitations = 0
    
    for word in analysis:
        if word['status'] in ['INSERTION']:
            insertions += 1
            continue
        if word['status'] == 'SELF_CORRECTION':
            self_corrections += 1
            continue
        
        total_ref += 1
        if word['status'] == 'CORRECT':
            correct += 1
        elif word['status'] == 'SUBSTITUTION':
            substitutions += 1
        elif word['status'] == 'OMISSION':
            omissions += 1
        
        if word.get('hesitation', False):
            hesitations += 1
    
    # Also count hesitations on non-reference words
    for word in analysis:
        if word['status'] in ['INSERTION', 'SELF_CORRECTION'] and word.get('hesitation', False):
            hesitations += 1
    
    accuracy = (correct / total_ref * 100) if total_ref > 0 else 0
    
    return {
        'accuracy': accuracy,
        'total_words': total_ref,
        'correct': correct,
        'substitutions': substitutions,
        'omissions': omissions,
        'insertions': insertions,
        'self_corrections': self_corrections,
        'hesitations': hesitations
    }


def generate_html(analysis, stats):
    """Generate HTML visualization"""
    accuracy = stats['accuracy']
    html = f"""
    <div style="font-family: 'Arial'; direction: rtl; text-align: right; font-size: 20px; line-height: 2.5; background: #f9f9f9; padding: 20px; border-radius: 10px;">
        <h3>نتائج التحليل - WhisperX (Large-V3)</h3>
        <table style="font-size: 16px; margin: 10px 0; border-collapse: collapse; width: 100%;">
            <tr>
                <td style="padding: 5px 10px;"><strong>الدقة (Accuracy):</strong> {accuracy:.1f}%</td>
                <td style="padding: 5px 10px;"><strong>الكلمات الصحيحة:</strong> {stats['correct']}/{stats['total_words']}</td>
            </tr>
            <tr>
                <td style="padding: 5px 10px;"><strong>أخطاء الاستبدال:</strong> {stats['substitutions']}</td>
                <td style="padding: 5px 10px;"><strong>كلمات محذوفة:</strong> {stats['omissions']}</td>
            </tr>
            <tr>
                <td style="padding: 5px 10px;"><strong>التردد (Hesitations):</strong> {stats['hesitations']}</td>
                <td style="padding: 5px 10px;"><strong>التصحيح الذاتي:</strong> {stats['self_corrections']}</td>
            </tr>
        </table>
        <p><strong>الألوان:</strong> 
            <span style="color: green;">✓ صحيح</span> | 
            <span style="color: #FF5722;">✗ خطأ</span> | 
            <span style="background: #F8BBD0; padding: 2px 5px;">تردد</span> |
            <span style="background: #FFCCBC; padding: 2px 5px;">تصحيح ذاتي</span>
        </p>
        <hr>
        <p>
    """
    
    for word in analysis:
        text = word['word']
        status = word['status']
        hesitation = word.get('hesitation', False)
        
        # Default styling
        color = "green"
        bg = "transparent"
        tooltip = ""
        
        if status == "OMISSION":
            color = "#9E9E9E"
            tooltip = "لم تُقرأ"
        elif status == "SUBSTITUTION":
            color = "#FF5722"
            tooltip = word.get('info', '')
        elif status == "SELF_CORRECTION":
            color = "#E64A19"
            bg = "#FFCCBC"
            tooltip = "تصحيح ذاتي"
        elif status == "INSERTION":
            color = "#795548"
            tooltip = "كلمة إضافية"
        
        if hesitation:
            bg = "#F8BBD0"
            tooltip = "تردد - " + tooltip
        
        html += f'<span title="{tooltip}" style="color: {color}; background-color: {bg}; padding: 2px 5px; border-radius: 4px; margin: 2px;">{text}</span> '
    
    html += "</p></div>"
    return html


def analyze(audio):
    """Main analysis function for Gradio"""
    if audio is None:
        return "No audio uploaded", {}
    
    try:
        analysis, transcript = analyze_reading_whisperx(audio, REFERENCE_TEXT)
        stats = calculate_stats(analysis)
        html = generate_html(analysis, stats)
        
        return html, {
            "accuracy": f"{stats['accuracy']:.1f}%",
            "correct_words": f"{stats['correct']}/{stats['total_words']}",
            "substitutions": stats['substitutions'],
            "omissions": stats['omissions'],
            "hesitations": stats['hesitations'],
            "self_corrections": stats['self_corrections'],
            "raw_transcript": transcript,
            "word_analysis": analysis
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", {}


# Custom CSS for Premium Look
custom_css = """
body { background-color: #f7f9fc; }
.gradio-container { max-width: 1000px !important; margin: 0 auto !important; font-family: 'Inter', 'Roboto', sans-serif !important; }
.gr-button-primary { background: linear-gradient(135deg, #6e8efb, #a777e3) !important; border: none !important; border-radius: 8px !important; transition: transform 0.2s ease !important; }
.gr-button-primary:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(110, 142, 251, 0.4); }
.header-logo { display: block; margin-left: auto; margin-right: auto; width: 150px; border-radius: 50%; box-shadow: 0 10px 30px rgba(0,0,0,0.1); margin-bottom: 20px; }
.app-title { text-align: center; color: #1e3a8a; font-weight: 800; font-size: 2.5rem; margin-bottom: 5px; }
.app-subtitle { text-align: center; color: #64748b; font-size: 1.1rem; margin-bottom: 30px; }
.analysis-card { background: white; border-radius: 12px; padding: 25px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); border-top: 5px solid #6e8efb; }
"""

# Gradio Interface
with gr.Blocks(title="Kawkab AI - تقييم فصاحة القراءة", css=custom_css) as demo:
    with gr.Column(elem_id="header-container"):
        if os.path.exists("FinallLogo-02.avif"):
            gr.Image("FinallLogo-02.avif", elem_classes="header-logo", show_label=False, interactive=False, container=False)
        gr.Markdown("# Kawkab AI", elem_classes="app-title")
        gr.Markdown("### تقييم فصاحة القراءة بالذكاء الاصطناعي", elem_classes="app-subtitle")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(type="filepath", label="الرجاء تسجيل أو رفع ملف صوتي")
            analyze_btn = gr.Button("🔍 بدء تحليل القراءة", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### النص المطلوب قراءته:")
            gr.Markdown(f"> {REFERENCE_TEXT}")
    
    with gr.Row():
        with gr.Column(elem_classes="analysis-card"):
            html_output = gr.HTML(label="نتائج التحليل")
        with gr.Column():
            json_output = gr.JSON(label="البيانات المفصلة")
    
    analyze_btn.click(
        fn=analyze,
        inputs=[audio_input],
        outputs=[html_output, json_output]
    )

if __name__ == "__main__":
    # For VPS deployment, ensure share=False and server_name="0.0.0.0"
    demo.launch(share=True, server_name="0.0.0.0")
