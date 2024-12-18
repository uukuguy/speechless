# Import necessary libraries and configure settings
import torch
import torchaudio
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision('high')

import ChatTTS
from .base import TTSGenerator

class ChatTTSGenerator(TTSGenerator):
    def __init__(self, model_path="/opt/local/TTS/ChatTTS", device="cpu", sampling_rate=24000, compile=False):
        self.model_path = model_path
        self.device = device
        self.sampling_rate = sampling_rate

        self.chat_tts = ChatTTS.Chat()
        self.chat_tts.load(source="custom", custom_path=model_path, device=device, compile=compile) # Set to True for better performance

    def generate_speaker_audio(self, text, , device=None):
        # ###################################
        # # Sample a speaker from Gaussian.

        # rand_spk = chat.sample_random_speaker()
        # print(rand_spk) # save it for later timbre recovery

        # params_infer_code = ChatTTS.Chat.InferCodeParams(
        #     spk_emb = rand_spk, # add sampled speaker 
        #     temperature = .3,   # using custom temperature
        #     top_P = 0.7,        # top P decode
        #     top_K = 20,         # top K decode
        # )

        # ###################################
        # # For sentence level manual control.

        # # use oral_(0-9), laugh_(0-2), break_(0-7) 
        # # to generate special token in text to synthesize.
        # params_refine_text = ChatTTS.Chat.RefineTextParams(
        #     prompt='[oral_2][laugh_0][break_6]',
        # )

        # wavs = chat.infer(
        #     texts,
        #     params_refine_text=params_refine_text,
        #     params_infer_code=params_infer_code,
        # )

        wavs = self.chat_tts.infer(texts)
        audio_arr = wavs[0]

        # from IPython.display import Audio
        # Audio(audio_arr, rate=24_000, autoplay=True)

        # # Save the generated audio 
        # torchaudio.save("output.wav", torch.from_numpy(audio_arr), 24000)
        

        return audio_arr, self.sampling_rate