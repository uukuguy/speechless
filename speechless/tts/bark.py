
from loguru import logger
from transformers import BarkModel, AutoProcessor 

from .base import TTSGenerator
class BarkTTSGenerator(TTSGenerator):
    def __init__(self, model_path, device="cpu", sampling_rate=24000):
        # BARK_MODEL="/opt/local/TTS/bark"
        self.model_path=model_path
        self.device = device
        self.bark_model = BarkModel.from_pretrained(model_path).to(device)
        self.bark_processor = AutoProcessor.from_pretrained(model_path)
        self.bark_sampling_rate = sampling_rate

    def generate_speaker_audio(text, voice_preset="v2/zh_speaker_1", device=None):
        """Generate audio using Bark for Speaker 2"""
        # logger.debug(f"bark generating speaker audio...")
        device = device or self.device
        inputs = self.bark_processor(text, voice_preset=voice_preset).to(device)
        speech_output = self.bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
        audio_arr = speech_output[0].cpu().numpy()
        # logger.info(f"bark generating speaker audio done.")

        return audio_arr, self.bark_sampling_rate