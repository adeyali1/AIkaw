"""
Download the best Whisper model for Arabic with resume support
"""
import os
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Use standard download

from huggingface_hub import snapshot_download
import sys

print("=" * 60)
print("Downloading Whisper Large-V3 (Best for Arabic)")
print("This is ~3GB, please wait...")
print("=" * 60)

# Try models in order of preference
models_to_try = [
    ("Systran/faster-whisper-large-v3", "Large-V3 (Best)"),
    ("Systran/faster-whisper-medium", "Medium (Good)"),
]

for repo_id, name in models_to_try:
    print(f"\n🔄 Trying: {name} ({repo_id})...")
    try:
        path = snapshot_download(
            repo_id=repo_id,
            resume_download=True,
            max_workers=2,  # Fewer workers for stability
        )
        print(f"\n✅ SUCCESS! {name} downloaded to: {path}")
        print("\nYou can now run the app!")
        sys.exit(0)
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        continue

print("\n❌ All downloads failed. Please check your internet connection.")
sys.exit(1)
