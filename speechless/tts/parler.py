from transformers import AutoTokenizer
from loguru import logger

from .base import TTSGenerator
class ParlerTTSGenerator(TTSGenerator):
    def __init__(self, model_path, device="cpu"):
        self.model_path = model_path
        self.device = device

        # PARLER_MODEL="/opt/local/TTS/parler-tts-mini-v1.1"
        from parler_tts import ParlerTTSForConditionalGeneration
        self.parler_model = ParlerTTSForConditionalGeneration.from_pretrained(model_path).to(device)
        self.parler_tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.speaker_description = """
        Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
        """

    def generate_speaker_audio(self, text, speaker_description=None, device=None):
        """Generate audio using ParlerTTS for Speaker 1"""
        # logger.debug(f"parler generating speaker audio...")

        device = device or self.device
        speaker_description = speaker_description or self.speaker_description

        input_ids = self.parler_tokenizer(speaker_description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = self.parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
        generation = self.parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        # logger.info(f"parler generating speaker audio done.")
        return audio_arr, self.parler_model.config.sampling_rate