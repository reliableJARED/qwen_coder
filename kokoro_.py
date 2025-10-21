import os
import platform
import warnings
import sounddevice as sd
import soundfile as sf
import torch
import threading
import queue
import time
import numpy as np
from kokoro import KPipeline
from typing import Optional, Generator, Tuple, Any

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.modules.rnn")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.utils.weight_norm")

def kokoroTTS_playback(play_queue, stop_queue, shutdown_event):
    """Play audio data using sounddevice."""
    while not shutdown_event.is_set():
        try:
            # Get audio from queue (THIS WAS THE BUG - was using put_nowait instead of get_nowait)
            data = play_queue.get(timeout=0.1)
            audio = data[0]
            samplerate = data[1]
            
            print(f"Playing audio chunk of length {len(audio)}")
            
            # Play the audio chunk
            sd.play(audio, samplerate)
            
            # Wait for playback to finish or until stop is requested
            while not shutdown_event.is_set():
                # Check if stop signal received
                try:
                    signal = stop_queue.get_nowait()
                    if signal == 'stop':
                        print("Stop signal received. Stopping playback.")
                        sd.stop()
                        # Clear any remaining items from stop and play queues
                        while not stop_queue.empty():
                            stop_queue.get_nowait()
                        while not play_queue.empty():
                            play_queue.get_nowait()
                        break
                except queue.Empty:
                    pass
                
                # Check if playback is still active
                if not sd.get_stream().active:
                    break
                    
                time.sleep(0.01)  # Check every 10ms
                
        except queue.Empty:
            time.sleep(0.01)
        except Exception as e:
            if not shutdown_event.is_set():
                print(f"Playback error: {e}")
            time.sleep(0.01)
    
    # Clean shutdown
    sd.stop()
    print("Playback thread shutting down cleanly")


class kokoroTTS:
    def __init__(self) -> None:
        # Check for internet connection
        _ = self.check_internet()

        self.KPipeline = KPipeline
        self.sf = sf
        self.torch = torch
        self.sd = sd
        
        
        # Voice definitions reference
        #https://github.com/hexgrad/kokoro/tree/main/kokoro.js/voices
        VOICES_FEMALE: list[str] = [
            "af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", 
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky"
        ]
        
        VOICES_MALE: list[str] = [
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", 
            "am_michael", "am_onyx", "am_puck", "am_santa"
        ]

        self.lang_code: str = 'a'  # 'a' for American English
        self.voice: str = 'af_sky,af_jessica'#  Single voice can be requested (e.g. 'af_sky') or multiple voices (e.g. 'af_bella,af_jessica'). If multiple voices are requested, they are averaged.
        self.speech_speed: float = 1.0  # Normal speed

        self.pipeline = None
        self._initialize_pipeline()#will set pipeline

        
        # Create queues for thread communication
        self.stop_queue = queue.Queue(maxsize=2)
        self.play_queue = queue.Queue(maxsize=25)
        self.shutdown_event = threading.Event()
        
        # Start playback thread as daemon so it exits when main program exits
        self.playback_thread = threading.Thread(
            target=kokoroTTS_playback, 
            args=(self.play_queue, self.stop_queue, self.shutdown_event),
            daemon=True
        )
        self.playback_thread.start()

    def check_internet(self):
        import socket
        """Check if internet connection is available."""
        try:
            socket.create_connection(("huggingface.co", 443), timeout=5)
            return True
        except (socket.timeout, socket.error, OSError):
            os.environ["HF_HUB_OFFLINE"] = "1"
            return False

    def _initialize_pipeline(self) -> None:
        """Initialize the Kokoro pipeline with MPS fallback for Mac.
        https://github.com/hexgrad/kokoro/blob/main/kokoro/pipeline.py
        """
        try:
            # Set MPS fallback for Mac M1/M2/M3/M4
            if platform.system() == "Darwin" and self.torch.backends.mps.is_available():
                os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
            
            self.pipeline = self.KPipeline(self.lang_code, repo_id='hexgrad/Kokoro-82M')
            print("Kokoro pipeline initialized successfully")
        except Exception as e:
            print(f"Failed to initialize Kokoro pipeline: {e}")
            raise

    def synthesize_speech(self, text: str, auto_play: bool = False) -> Optional[Tuple[np.ndarray, int]]:
        """Synthesize speech from text and optionally play it."""
        # Generate audio
        generator: Generator[Tuple[Any, Any, Optional[np.ndarray]], None, None] = self.pipeline(
            text, voice=self.voice, speed=self.speech_speed
        )
            
        # Process audio
        audio_data: np.ndarray = np.array([])
        samplerate: int = 24000#smaller, slower/lower - bigger faster/higher

        for i, (graphemes, phonemes, audio) in enumerate(generator):
            if audio is not None:
                audio_data = np.concatenate((audio_data, audio))

                if auto_play:
                    try:
                        print(f"Queuing audio chunk {i}")
                        self.play_queue.put((audio, samplerate))
                    except Exception as e:
                        print(f"Failed to queue audio: {e}")
                        return None

        if not auto_play:
            return audio_data, samplerate

        return None
    
    def stop_playback(self) -> None:
        """Stop the current audio playback."""
        print("Stopping playback...")
        try:
            self.stop_queue.put('stop', timeout=1)
        except queue.Full:
            print("Stop queue full, clearing and retrying")
            while not self.stop_queue.empty():
                self.stop_queue.get_nowait()
            self.stop_queue.put('stop')

    def play_audio(self, audio: np.ndarray, samplerate: int) -> None:
        """Queue audio for playback."""
        self.play_queue.put((audio, samplerate))

    def shutdown(self) -> None:
        """Cleanly shutdown the TTS system."""
        print("Shutting down TTS system...")
        self.stop_playback()
        self.shutdown_event.set()
        
        # Wait for playback thread to finish (with timeout)
        self.playback_thread.join(timeout=2.0)
        
        if self.playback_thread.is_alive():
            print("Warning: Playback thread did not shutdown cleanly")
        else:
            print("TTS system shutdown complete")


if __name__ == "__main__":
    import signal
    import sys
    
    # Global reference for signal handler
    kokoro_instance = None
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\nCtrl+C detected! Shutting down gracefully...")
        if kokoro_instance:
            kokoro_instance.shutdown()
        sys.exit(0)
    
    # Register signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    kokoro_instance = kokoroTTS()
    kokoro_instance._initialize_pipeline()
    
    # Test with longer text
    sample_text: str = """
    Hello, this is a test of the Kokoro text to speech synthesis model.
    This is a longer piece of text that will take several seconds to play back.
    We want to test the stop functionality by interrupting the playback 
    in the middle of this sentence. If everything works correctly, 
    the audio should stop abruptly when we call the stop method.
    This extra text ensures we have enough audio duration to test properly.
    """
    
    try:
        print("Starting playback...")
        kokoro_instance.synthesize_speech(sample_text, auto_play=True)
        
        # Wait, then stop playback
        print("\nStopping playback after 4 seconds...")
        time.sleep(5)
        kokoro_instance.stop_playback()
        
        # Test second playback
        print("\nTesting second playback (should play completely)...")
        # Wait a moment
        time.sleep(1)
        kokoro_instance.synthesize_speech("Second test playback. This should play fully.", auto_play=True)
        
        # Let it finish
        time.sleep(5)
        print("Done!")
        
    finally:
        # Always cleanup
        kokoro_instance.shutdown()