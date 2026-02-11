import os
import shutil
import subprocess
import videoio
import numpy as np

class VideoWriter:
    def __init__(self, path, fps, audio_source=None, src_audio_path=None, quality="high", high_quality=False, **kwargs):
        self.path = path
        self.fps = fps
        # Handle audio source from different kwarg names
        self.audio_source = audio_source if audio_source else src_audio_path
        
        # Handle quality
        self.quality = quality
        self.high_quality = high_quality
        
        self.writer = None
        self.resolution = None
        self.failed = False

    def write(self, image):
        if self.failed:
            return

        if self.writer is None:
            # Determine resolution from first frame
            if image.ndim == 3:
                h, w = image.shape[:2]
            else:
                # Assuming gray scale
                h, w = image.shape
                
            self.resolution = (w, h)
            
            # Map quality to videoio parameters
            # videoio default preset is 'slow'
            preset = "slow"
            lossless = False
            
            # Check if we want high quality
            if self.quality == "high" or self.high_quality:
                # For videoio, maybe use slower preset or higher bitrate if possible?
                pass
                
            self.writer = videoio.VideoWriter(
                self.path,
                resolution=self.resolution,
                fps=self.fps,
                preset=preset,
                lossless=lossless
            )
        
        # Ensure image is compatible (uint8)
        if image.dtype != np.uint8:
             # Assuming float 0-1 if not uint8
             image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
             
        try:
            self.writer.write(image)
        except BrokenPipeError:
            print(f"[VideoWriter] Error: BrokenPipeError when writing frame with shape {image.shape}. FFmpeg process might have died.")
            self.writer = None
            self.failed = True
        except Exception as e:
            print(f"[VideoWriter] Error writing video frame: {e}")
            self.failed = True

    def release(self):
        if self.writer:
            self.writer.close()
            self.writer = None
        
        # If audio_source is provided, mux audio
        if self.audio_source and os.path.exists(self.audio_source):
            self._mux_audio()

    def close(self):
        self.release()

    def _mux_audio(self):
        # Mux audio using ffmpeg command line
        temp_path = self.path + ".temp.mp4"
        if os.path.exists(self.path):
            os.rename(self.path, temp_path)
            
            cmd = [
                "ffmpeg",
                "-y",
                "-i", temp_path,
                "-i", self.audio_source,
                "-c:v", "copy",
                "-c:a", "aac",
                "-strict", "experimental",
                "-shortest", # Stop when shortest stream ends
                self.path
            ]
            
            try:
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.remove(temp_path)
            except subprocess.CalledProcessError as e:
                print(f"Failed to mux audio: {e}")
                if os.path.exists(temp_path):
                    os.rename(temp_path, self.path)
