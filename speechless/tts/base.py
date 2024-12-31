from loguru import logger
from transformers import AutoTokenizer
from scipy.io import wavfile 
from pydub import AudioSegment
import io
import numpy as np

def numpy_to_audio_segment(audio_arr, sampling_rate):
    """Convert numpy array to AudioSegment"""
    # Convert to 16-bit PCM
    audio_int16 = (audio_arr * 32767).astype(np.int16)
    
    # Create WAV file in memory
    byte_io = io.BytesIO()
    wavfile.write(byte_io, sampling_rate, audio_int16)
    byte_io.seek(0)
    
    # Convert to AudioSegment
    return AudioSegment.from_wav(byte_io)


class TTSGenerator:
    @classmethod
    def build_Generator(cls, model_path, device="cpu"):
        """Build TTS Generator"""
        if "parler" in model_path.lower():
            from .parler import ParlerTTSGenerator
            return ParlerTTSGenerator(model_path, device)
        elif "bark" in model_path.lower():
            from .bark import BarkTTSGenerator
            return BarkTTSGenerator(model_path, device)
        elif "chattts" in model_path.lower():
            from .chattts import ChatTTSGenerator
            return ChatTTSGenerator(model_path, device)
        else:
            raise ValueError(f"Unknown TTS model: {model_path}")


    def generate_speaker_audio(text , device=None):
        raise NotImplementedError

    def generate_audio_segment(self, text, speaker=0):
        audio_arr, rate = self.generate_speaker_audio(text, speaker=speaker)
        audio_segment = numpy_to_audio_segment(audio_arr, rate)

        # from pydub.playback import play as play_snd
        # play_snd(audio_segment)

        return audio_segment

# final_audio = None
# audio_arr, rate = generate_speaker_audio(text)

# audio_segment = numpy_to_audio_segment(audio_arr, rate)
# play_snd(audio_segment)