import os
import platform
import warnings
import sounddevice as sd
import soundfile as sf
import torch
import threading
import queue
import numpy as np
from kokoro import KPipeline
from typing import  Optional, Generator, Tuple, Any,Tuple
# Add this at the top of your script, after imports
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")
"""Warning 1: RNN Dropout Warning
Issue: The warning about dropout expects num_layers > 1 but got num_layers=1 from the Kokoro model's internal architecture.

Warning 2: Weight Norm Deprecation Warning
Issue: torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
"""

class kokoroTTS:
    def __init__(self) -> None:
        #FIRST - check for internet so we can change our flag. KPipeline will still call HF on load without
        _ = self.check_internet()

        #now continue with imports


        self.KPipeline = KPipeline
        self.sf = sf
        self.torch = torch
        self.sd = sd
        
        self.pipeline: Optional[Any] = None  # will hold the TTS pipeline once initialized

        # Voice definitions - US English
        VOICES_FEMALE: list[str] = [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"
        ]
        
        VOICES_MALE: list[str] = [
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
            "am_michael", "am_onyx", "am_puck", "am_santa"
        ]

        self.accent: str = 'a'  # 'a' for American English, 'b' for British English
        self.voice: str = 'af_sky'  # Default
        self.speech_speed: float = 1.0  # Normal speed
        self.stop_queue = queue.Queue()

    def check_internet(self):
        import socket
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):
            # Set the environment variable to enable offline mode
            os.environ["HF_HUB_OFFLINE"] = "1"
            return False


    def _initialize_pipeline(self) -> None:
        """Initialize the Kokoro pipeline with MPS fallback for Mac."""
        try:
            # Set MPS fallback for Mac M1/M2/M3/M4
            if platform.system() == "Darwin" and self.torch.backends.mps.is_available():
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            # Explicitly specify repo_id to suppress warning
            self.pipeline = self.KPipeline(self.accent, repo_id='hexgrad/Kokoro-82M')

            print("Kokoro pipeline initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Kokoro pipeline: {e}")
            raise

    def synthesize_speech(self, text: str, auto_play: bool = False) -> Optional[Tuple[np.ndarray, int]]:
        """Synthesize speech from text and play it through the default audio device or return audio data."""

        # Generate audio
        generator: Generator[Tuple[Any, Any, Optional['np.ndarray']], None, None] = self.pipeline(
            text, voice=self.voice, speed=self.speech_speed
        )
            
        # Process and save audio
        audio_data: np.ndarray = np.array([])
        samplerate: int = 0
        for i, (graphemes, phonemes, audio) in enumerate(generator):
            if audio is not None:
                # Append audio to the array
                audio_data = np.concatenate((audio_data, audio))
                samplerate = 24000  # Assuming all audio has the same samplerate

                if auto_play:
                    # Play audio using sounddevice
                    try:
                        print("Playing Audio Chunk")
                        self.play_audio(audio, samplerate)
                        
                    except Exception as e:
                        print(f"Playback failed: {e}")
                        return None

        if not auto_play:
            return audio_data, samplerate

        return None
    
    def stop_playback(self) -> None:
        """Stop the current audio playback."""
        print("HOLD THE BUS")
        self.stop_queue.put_nowait('stop')

    def play_audio(self, audio: np.ndarray, samplerate: int) -> None:
        """Play audio data using sounddevice."""
        try:
            #play the entire audio chunk
            sd.play(audio, samplerate)
            
            # Wait for playback to finish or until stop is requested
            while sd.get_stream().active:
                if not self.stop_queue.empty():
                    signal = self.stop_queue.get_nowait()
                    if signal == 'stop':
                        print("Stop signal received. Stopping playback.")
                        sd.stop()
                        break
                time.sleep(0.01)  # Check every 10ms
                
        except Exception as e:
            print(f"Audio playback error: {e}")
        finally:
            # Clear the queue
            while not self.stop_queue.empty():
                self.stop_queue.get_nowait()
       

if __name__ == "__main__":
    import time
    
    kokoro = kokoroTTS()
    kokoro._initialize_pipeline()
    
    # Create a longer text that will take several seconds to play
    sample_text: str = """
    Hello, this is a test of the Kokoro text to speech synthesis model.
    This is a longer piece of text that will take several seconds to play back.
    We want to test the stop functionality by interrupting the playback 
    in the middle of this sentence. If everything works correctly, 
    the audio should stop abruptly when we call the stop method.
    This extra text ensures we have enough audio duration to test properly.
    """
    
    # Start playback in a separate thread so we can interrupt it
    playback_thread = threading.Thread(
        target=kokoro.synthesize_speech, 
        args=(sample_text,), 
        kwargs={'auto_play': True}
    )
    
    print("Starting playback...")
    playback_thread.start()

    
    # Wait 2 seconds, then stop playback
    time.sleep(5)
    print("Stopping playback after 2 seconds...")
    kokoro.stop_playback()
    
    # Wait for the thread to finish
    playback_thread.join()
    print("Playback thread completed")
    
    # Optional: Start another playback to verify everything still works
    print("\nTesting second playback (should play completely)...")
    time.sleep(1)
    kokoro.synthesize_speech("Second test playback. This should play fully.", auto_play=True)
    print("Done!")