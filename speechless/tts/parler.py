from loguru import logger
from scipy.io import wavfile 
from pydub import AudioSegment
from pydub.playback import play as play_snd
from transformers import AutoTokenizer
from parler_tts import ParlerTTSForConditionalGeneration

PARLER_MODEL="/opt/local/TTS/parler-tts-mini-v1.1"
parler_model = ParlerTTSForConditionalGeneration.from_pretrained(PARLER_MODEL).to(device)
parler_tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL)

speaker1_description = """
Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
"""

def generate_speaker_audio(text):
    """Generate audio using ParlerTTS for Speaker 1"""
    logger.debug(f"parler generating speaker audio...")
    input_ids = parler_tokenizer(speaker1_description, return_tensors="pt").input_ids.to(device)
    prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
    generation = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    audio_arr = generation.cpu().numpy().squeeze()
    logger.info(f"parler generating speaker audio done.")
    return audio_arr, parler_model.config.sampling_rate

    
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

final_audio = None
audio_arr, rate = generate_speaker_audio(text)

audio_segment = numpy_to_audio_segment(audio_arr, rate)
play_snd(audio_segment)