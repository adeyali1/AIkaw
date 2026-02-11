# POC: Arabic Reading Fluency Assessment - Methods & Tools

Daqiq is a Proof-of-Concept (POC) tool designed to assess Arabic reading fluency by comparing a student's spoken audio against a reference text. It utilizes State-of-the-Art (SOTA) AI for speech recognition and a custom-tuned algortihmic approach for grading and feedback.

## 1. Core Tools & Technologies

The following libraries and frameworks form the backbone of the system:

### AI & Signal Processing
*   **Hugging Face Transformers (`transformers`)**: Used to load and inference the pre-trained Automatic Speech Recognition (ASR) model.
    *   **Model**: `jonatasgrosman/wav2vec2-large-xlsr-53-arabic`
    *   **Architecture**: Wav2Vec 2.0 (Self-supervised learning for speech).
*   **PyTorch (`torch`)**: The deep learning framework used to run the neural network model.
*   **Librosa (`librosa`)**: Standard library for audio analysis. Used here for loading audio files, converting them to mono, and resampling them to the **16kHz** sample rate required by the model.
*   **NumPy (`numpy`)**: Used for high-performance array manipulations, specifically for constructing the scoring matrices in the alignment algorithm.

### Application & Interface
*   **Gradio (`gradio`)**: Used to build the web-based user interface. It handles audio recording (microphone), file uploading, and rendering the rich HTML feedback report.
*   **Python Standard Libraries**: `difflib` (for string similarity), `re` (regex for text normalization).

---

## 2. Methodology & Pipeline

The assessment process follows a linear pipeline:

### Step 1: Text Normalization (Preprocessing)
Before any comparison, Arabic text is complex and requires normalization to ensure fair grading.
*   **Normalization**: Consolidates different forms of Alef (`أ`, `إ`, `آ` → `ا`), normalizes text delimiters, and handles Ta Marbuta/Ha.
*   **Stripping**: A separate version of text is created by stripping all diacritics (Short vowels/Harakat) to allow for "Base Matching" checks.

### Step 2: Automatic Speech Recognition (ASR)
The system converts the user's voice into text tokens.
1.  **Input**: Audio file (WAV/MP3).
2.  **Processing**: Audio is resampled to 16,000Hz.
3.  **Inference**: The Wav2Vec2 model outputs logical probabilities (logits) for each time step (~20ms).
4.  **CTC Decoding**:
    *   The model uses Connectionist Temporal Classification (CTC).
    *   We implement a custom decoding loop to convert these frame-level predictions into words.
    *   This custom decoder is crucial because it allows us to capture the exact **start and end timestamps** for every spoken word, which is needed for the "Hesitation" detection.

### Step 3: Sequence Alignment (The Grading Core)
This is the "Brain" of the POC. We do not simply compare strings; we align sequences of words to find what was skipped, changed, or added.
*   **Algorithm**: **Needleman-Wunsch**.
    *   This is a dynamic programming algorithm typically used in bioinformatics to align DNA sequences.
    *   We use it to align the Reference Sentence (Truth) with the Hypothesis Sentence (Spoken).
*   **Scoring Matrix**:
    *   **Exact Match**: +3.0 Points.
    *   **Gap (Omission/Insertion)**: -2.0 Points.
    *   **Mismatch**: -1.0 Points.
    *   The algorithm finds the "Global Optimal Alignment" path that maximizes the total score.

### Step 4: Classification & Error Categorization
Once aligned, every word is characterized based on its alignment status and string similarity (`difflib`):

| Error Type | Method of Detection | Visual Feedback |
| :--- | :--- | :--- |
| **CORRECT** | Exact string match (normalized). | <span style="color:green">Green</span> |
| **TASHKEEL** | Base letters match, but diacritics differ (Similarity > 85%). | <span style="color:red">Red/Pink</span> |
| **SUBSTITUTION** | Word exists in place of another, but is significantly different (Similarity > 60%). | <span style="color:blue">Blue</span> |
| **OMISSION** | The alignment algorithm determined a word was skipped in the spoken sequence. | <span style="color:orange; background:#FFCCBC">Orange Highight</span> |
| **INSERTION** | The alignment algorithm determined an extra word was spoken. | <span style="color:orange; background:#FFF9C4">Yellow Highlight</span> |
| **REPETITION (Takrar)**| Defined as an **INSERTION** where the inserted word is identical to the immediately preceding word. | <span style="color:blue; background:#B3E5FC">Light Blue</span> |
| **HESITATION** | Detected if the silence gap between the end of the previous word and start of current word > **2.0 seconds**. | <span style="background: #E91E63; color: white">(Pause Icon)</span> |

---

## 3. Why this approach? (POC Rationale)

*   **Robustness vs. Exactness**: Standard string distance (Levenshtein) is too simple. It doesn't tell you *what* kind of error happened (e.g., did they miss a vowel or say a totally wrong word?). Needleman-Wunsch allows us to define "Costs" for these specific pedagocial events.
*   **Word-Level Timing**: Using the raw model logits allows us to measure hesitation, which is a key metric for fluency.
*   **Hybrid Matching**: We mix strict matching (for 100% accuracy) with fuzzy matching (`difflib`) to distinguish between "Reading Wrong" and "Reading with bad accent/vowels".
