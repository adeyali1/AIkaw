
import shutil
import os
import imageio_ffmpeg
import sys

# Get source path
src = imageio_ffmpeg.get_ffmpeg_exe()
dst = os.path.join(os.getcwd(), "ffmpeg.exe")

print(f"Source: {src}")
print(f"Destination: {dst}")

try:
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        print("✅ ffmpeg.exe created successfully!")
    else:
        print("ℹ️ ffmpeg.exe already exists.")
        
    # Verify it works
    import subprocess
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("✅ ffmpeg command works!")
    
except Exception as e:
    print(f"❌ Error: {e}")
