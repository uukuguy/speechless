
from loguru import logger
from transformers import BarkModel, AutoProcessor, AutoTokenizer
from scipy.io import wavfile 
from pydub import AudioSegment
from pydub.playback import play as play_snd

BARK_MODEL="/opt/local/TTS/bark"
bark_processor = AutoProcessor.from_pretrained(BARK_MODEL)
# bark_model = BarkModel.from_pretrained(BARK_MODEL, torch_dtype=torch.float16).to(device)
bark_model = BarkModel.from_pretrained(BARK_MODEL).to(device)
bark_sampling_rate = 24000

def generate_speaker_audio(text, device):
    """Generate audio using Bark for Speaker 2"""
    logger.debug(f"bark generating speaker audio...")
    inputs = bark_processor(text, voice_preset="v2/zh_speaker_1").to(device)
    speech_output = bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
    audio_arr = speech_output[0].cpu().numpy()
    logger.info(f"bark generating speaker audio done.")
    return audio_arr, bark_sampling_rate

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