
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

try:
    import imageio_ffmpeg
    import os
    import subprocess

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    ffmpeg_dir = os.path.dirname(ffmpeg_exe)
    
    print(f"FFmpeg exe: {ffmpeg_exe}")
    print(f"Adding to PATH: {ffmpeg_dir}")
    
    os.environ["PATH"] += os.pathsep + ffmpeg_dir
    
    print("Testing ffmpeg call...")
    subprocess.run(["ffmpeg", "-version"], check=True)
    print("✅ FFmpeg is working correctly via subprocess!")
    
except Exception as e:
    print(f"❌ Error: {e}")
