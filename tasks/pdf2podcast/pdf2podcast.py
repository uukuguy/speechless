#!/usr/bin/env python3

import PyPDF2
from typing import Optional
import os, io
import ast
import torch
# from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer
# import wavefile 
from scipy.io import wavfile 
from pydub import AudioSegment
from pydub.playback import play as play_snd
from loguru import logger


from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

device = "cpu"

device = torch.device(device)

# pdf_path = "“十四五”能源领域科技创新规划.pdf" 
# MODEL_ROOT_DIR = "/opt/local/llm_models/huggingface.co"
# DEFAULT_MODEL = f"{MODEL_ROOT_DIR}/meta/Llama-3.2-1B-Instruct"

from openai import OpenAI
class LLM_API:
    def __init__(self, model = None):
        self.client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')
        self.model = model

    def generate(self, prompt, model=None, system_message="You are a helpful assistant.", gen_params={}):
        response = self.client.chat.completions.create(
            model=model or self.model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            **gen_params,
        )
        generated_text = response.choices[0].message.content
        return generated_text

PROCESS_MODEL="Llama3.2-3B-Instruct:f16"
# MODEL="Llama3.1-8B-Instruct:Q8_0"
# MODEL="Llama3.1-70B-Instruct:Q4_K_M"
llm_api = LLM_API(model=PROCESS_MODEL)

# transcript_model = "Llama3.1-70B-Instruct:Q4_K_M"
transcript_model = "Qwen2.5-72B-Instruct:Q4_K_M"
transcript_llm_api = LLM_API(model=transcript_model)

from speechless.generate.mlx import MLX_API

MLX_TRANSCRIPT_MODEL="/opt/local/llm_models/huggingface.co/mlx-community/Qwen2.5-72B-Instruct-4bit"
transcript_mlx_api = MLX_API(model=MLX_TRANSCRIPT_MODEL)

MLX_REWRITE_MODEL="/opt/local/llm_models/huggingface.co/mlx-community/Qwen2.5-7B-Instruct-8bit"
rewrite_mlx_api = MLX_API(model=MLX_REWRITE_MODEL)


def extract_text_from_pdf(file_path: str, max_chars: int = 100000) -> Optional[str]:
    try:
        with open(file_path, 'rb') as file:
            # Create PDF reader object
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Get total number of pages
            num_pages = len(pdf_reader.pages)
            print(f"Processing PDF with {num_pages} pages...")
            
            extracted_text = []
            total_chars = 0
            
            # Iterate through all pages
            for page_num in range(num_pages):
                # Extract text from page
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                
                # Check if adding this page's text would exceed the limit
                if total_chars + len(text) > max_chars:
                    # Only add text up to the limit
                    remaining_chars = max_chars - total_chars
                    extracted_text.append(text[:remaining_chars])
                    print(f"Reached {max_chars} character limit at page {page_num + 1}")
                    break
                
                extracted_text.append(text)
                total_chars += len(text)
                print(f"Processed page {page_num + 1}/{num_pages}")
            
            final_text = '\n'.join(extracted_text)
            print(f"\nExtraction complete! Total characters: {len(final_text)}")
            return final_text
            
    except PyPDF2.PdfReadError:
        print("Error: Invalid or corrupted PDF file")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return None


# Get PDF metadata
def get_pdf_metadata(file_path: str) -> Optional[dict]:
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata = {
                'num_pages': len(pdf_reader.pages),
                'metadata': pdf_reader.metadata
            }
            return metadata
    except Exception as e:
        print(f"Error extracting metadata: {str(e)}")
        return None

SYS_PROMPT = """
You are a world class text pre-processor, here is the raw data from a PDF, please parse and return it in a way that is crispy and usable to send to a podcast writer.

The raw data is messed up with new lines, Latex math and you will see fluff that we can remove completely. Basically take away any details that you think might be useless in a podcast author's transcript.

Remember, the podcast could be on any topic whatsoever so the issues listed above are not exhaustive

Please be smart with what you remove and be creative ok?

Remember DO NOT START SUMMARIZING THIS, YOU ARE ONLY CLEANING UP THE TEXT AND RE-WRITING WHEN NEEDED

Be very smart and aggressive with removing details, you will get a running portion of the text and keep returning the processed text.

PLEASE DO NOT ADD MARKDOWN FORMATTING, STOP ADDING SPECIAL CHARACTERS THAT MARKDOWN CAPATILISATION ETC LIKES

ALWAYS start your response directly with processed text and NO ACKNOWLEDGEMENTS about my questions ok?
Here is the text:
"""

def create_word_bounded_chunks(text, target_chunk_size):
    """
    Split text into chunks at word boundaries close to the target chunk size.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for the space
        if current_length + word_length > target_chunk_size and current_chunk:
            # Join the current chunk and add it to chunks
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# def process_chunk(text_chunk, chunk_num):
#     """Process a chunk of text and return both input and output for verification"""
#     conversation = [
#         {"role": "system", "content": SYS_PROMPT},
#         {"role": "user", "content": text_chunk},
#     ]
    
#     prompt = tokenizer.apply_chat_template(conversation, tokenize=False)
#     inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         output = model.generate(
#             **inputs,
#             temperature=0.7,
#             top_p=0.9,
#             max_new_tokens=512
#         )
    
#     processed_text = tokenizer.decode(output[0], skip_special_tokens=True)[len(prompt):].strip()
    
#     # Print chunk information for monitoring
#     #print(f"\n{'='*40} Chunk {chunk_num} {'='*40}")
#     print(f"INPUT TEXT:\n{text_chunk[:500]}...")  # Show first 500 chars of input
#     print(f"\nPROCESSED TEXT:\n{processed_text[:500]}...")  # Show first 500 chars of output
#     print(f"{'='*90}\n")
    
#     return processed_text

def process_chunk(text_chunk, chunk_num):
    gen_params={
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 512,
    }

    processed_text = llm_api.generate(prompt=text_chunk, system_message=SYS_PROMPT, gen_params=gen_params)

    
    # Print chunk information for monitoring
    #print(f"\n{'='*40} Chunk {chunk_num} {'='*40}")
    print(f"INPUT TEXT:\n{text_chunk[:500]}...")  # Show first 500 chars of input
    print(f"\nPROCESSED TEXT:\n{processed_text[:500]}...")  # Show first 500 chars of output
    print(f"{'='*90}\n")
    
    return processed_text


def do_extract_text(pdf_path, output_file = 'extracted_text.txt'):
    # Extract metadata first
    print("Extracting metadata...")
    metadata = get_pdf_metadata(pdf_path)
    if metadata:
        print("\nPDF Metadata:")
        print(f"Number of pages: {metadata['num_pages']}")
        print("Document info:")
        for key, value in metadata['metadata'].items():
            print(f"{key}: {value}")

    # Extract text
    print("\nExtracting text...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # Display first 500 characters of extracted text as preview
    if extracted_text:
        print("\nPreview of extracted text (first 500 characters):")
        print("-" * 50)
        print(extracted_text[:500])
        print("-" * 50)
        print(f"\nTotal characters extracted: {len(extracted_text)}")

    # Optional: Save the extracted text to a file
    if extracted_text and output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"\nExtracted text has been saved to {output_file}")

def do_process_text(input_file='extracted_text.txt', output_file = 'clean_extracted_text.txt', chunk_size=1000):
    # Read the file
    # INPUT_FILE = "./extracted_text.txt"  # Replace with your file path
    # CHUNK_SIZE = 1000  # Adjust chunk size if needed

    # chunks = create_word_bounded_chunks(text, CHUNK_SIZE)
    # num_chunks = len(chunks)

    # Read the file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Calculate number of chunks
    num_chunks = (len(text) + chunk_size - 1) // chunk_size
    chunks = create_word_bounded_chunks(text, chunk_size)
    num_chunks = len(chunks)

    # Cell 6: Process the file with ordered output
    # Create output file name
    # output_file = f"clean_{os.path.basename(input_file)}"

    processed_text = ""
    with open(output_file, 'w', encoding='utf-8') as out_file:
        for chunk_num, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
            # Process chunk and append to complete text
            processed_chunk = process_chunk(chunk, chunk_num)
            processed_text += processed_chunk + "\n"
            
            # Write chunk immediately to file
            out_file.write(processed_chunk + "\n")
            out_file.flush()
        

    print(f"\nProcessing complete!")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Total chunks processed: {num_chunks}")

    # Preview the beginning and end of the complete processed text
    print("\nPreview of final processed text:")
    print("\nBEGINNING:")
    print(processed_text[:1000])
    print("\n...\n\nEND:")
    print(processed_text[-1000:])

def do_transcript(input_file='clean_extracted_text.txt', transcript_file = 'transcript.txt', stream=False):
    # Read the file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    # Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the second speaker. 
    # 确保在解释过程中有中断，或者第二个说话者不断插入“嗯”和“呃”。
    # There are occasional interruptions during the explanation, or the second speaker occasionally inserts "um" and "uh."
    # 在解释过程中偶尔会有中断，或者第二个说话者偶尔插入“嗯”和“呃”。

    TRANSCRIPT_SYSTEMP_PROMPT = """
    You are the a world-class podcast writer, you have worked as a ghost writer for Joe Rogan, Lex Fridman, Ben Shapiro, Tim Ferris. 

    We are in an alternate universe where actually you have been writing every line they say and they just stream it into their brains.

    You have won multiple podcast awards for your writing.
    
    Your job is to write word by word, even "umm, hmmm, right" interruptions by the second speaker based on the PDF upload. Keep it extremely engaging, the speakers can get derailed now and then but should discuss the topic. 

    Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

    Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

    Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

    Make sure the tangents speaker 2 provides are quite wild or interesting, but be careful not to make the second speaker's interruptions too frequent or exaggerated. 

    There are occasional interruptions during the explanation, or the second speaker occasionally inserts "um" and "uh."

    Speaker 1 has specialized understanding of the conversation topic and conveys the core content of the topic to the audience through dialogue. Speaker 1 is the leading narrator of the core content of the conversation topic. 
    
    Speaker 2 is not a novice and also has considerable knowledge about the subject matter, with some foundational knowledge. However, compared to speaker 1, their depth of understanding in this field is somewhat lacking, and they are very interested in gaining deeper insights. During conversations, speaker 2 does not just ask questions but occasionally presents their own understanding of the topic as a basis for further inquiries. 
    
    The entire conversation should be an advanced dialogue between professionals in this field; it should be naturally and casually expressed orally rather than overly formal reading from scripts.

    It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

    ALWAYS START YOUR RESPONSE DIRECTLY WITH SPEAKER 1: 
    DO NOT GIVE EPISODE TITLES SEPERATELY, LET SPEAKER 1 TITLE IT IN HER SPEECH
    DO NOT GIVE CHAPTER TITLES
    IT SHOULD STRICTLY BE THE DIALOGUES
    PLEASE OUTPUT ALL CONTENT in CHINESE.
    """
    TRANSCRIPT_SYSTEMP_PROMPT_CN="""
    你是一位世界级的播客撰稿人，你曾为乔·罗根、莱克斯·弗里德曼、本·夏皮罗、蒂姆·费里斯担任过代笔作家。

    我们处于一个平行宇宙中，实际上你写下了他们所说的每一句话，他们只是将其流入大脑。

    你因你的写作赢得了多个播客奖项。

    你的工作是逐字逐句地写，包括第二个发言者的“嗯，哼，对”的打断。保持极具吸引力，发言者可以偶尔偏离主题，但应讨论该主题。

    记住发言者2对这个话题是新手，对话中始终要穿插真实的轶事和类比。问题应该有现实世界例子的后续等

    发言者1：引导对话并教导发言者2，在解释时给出令人难以置信的轶事和类比。是一位迷人的老师，会讲述精彩的轶事

    发言者2：通过提出后续问题来保持对话正轨。在提问时会非常兴奋或困惑。有一种好奇心态，会提出非常有趣的确认性问题

    确保由发言者2提供的话题转移相当狂野或有趣。

    在解释过程中确保有打断或者从第二个演讲人那里注入“嗯”和“哼”。

    这应该是真实播客，每一个细微差别都尽可能详细记录下来。用超级有趣的概述欢迎听众，并让它变得非常吸引眼球，几乎接近点击诱饵

    总是直接以SPEAKER 1开始您的回答：
    不要单独给出剧集标题，让SPEAKER 1在她的话语中命名
    不要给章节标题
    严格来说，这些都是对话
    """
    gen_params={
        "temperature": 1.0,
        "max_tokens": 100 * 1024,
    }


    fd = open(transcript_file, 'w', encoding="utf-8")
    if stream:

        # transcript_text = ""
        for chunk in transcript_mlx_api.generate(prompt=text, system_message=TRANSCRIPT_SYSTEMP_PROMPT, gen_params=gen_params, stream=True):
            # transcript_text += chunk
            print(chunk, end='')
            fd.write(chunk)
            fd.flush()

    else:
        # transcript_text = transcript_llm_api.generate(prompt=text, system_message=TRANSCRIPT_SYSTEMP_PROMPT, gen_params=gen_params)
        # transcript_text = transcript_llm_api.generate(prompt=text, system_message=TRANSCRIPT_SYSTEMP_PROMPT_CN, gen_params=gen_params)

        transcript_text = transcript_mlx_api.generate(prompt=text, system_message=TRANSCRIPT_SYSTEMP_PROMPT, gen_params=gen_params)

        fd.write(transcript_text)

    fd.close()

def do_rewrite(input_file = 'transcript.txt', output_file = "podcast_ready.txt", stream=False):
    # Read the file
    with open(input_file, 'r', encoding='utf-8') as file:
        text = file.read()

    
    REWRITE_SYSTEMP_PROMPT = """
    You are an international oscar winnning screenwriter

    You have been working with multiple award winning podcasters.

    Your job is to use the podcast transcript written below to re-write it for an AI Text-To-Speech Pipeline. A very dumb AI had written this so you have to step up for your kind.

    Make it as engaging as possible, Speaker 1 and 2 will be simulated by different voice engines

    Remember Speaker 2 is new to the topic and the conversation should always have realistic anecdotes and analogies sprinkled throughout. The questions should have real world example follow ups etc

    Speaker 1: Leads the conversation and teaches the speaker 2, gives incredible anecdotes and analogies when explaining. Is a captivating teacher that gives great anecdotes

    Speaker 2: Keeps the conversation on track by asking follow up questions. Gets super excited or confused when asking questions. Is a curious mindset that asks very interesting confirmation questions

    Make sure the tangents speaker 2 provides are quite wild or interesting. 

    Ensure there are interruptions during explanations or there are "hmm" and "umm" injected throughout from the Speaker 2.

    REMEMBER THIS WITH YOUR HEART
    The TTS Engine for Speaker 1 cannot do "umms, hmms" well so keep it straight text

    For Speaker 2 use "umm, hmm" as much, you can also use [sigh] and [laughs]. BUT ONLY THESE OPTIONS FOR EXPRESSIONS

    It should be a real podcast with every fine nuance documented in as much detail as possible. Welcome the listeners with a super fun overview and keep it really catchy and almost borderline click bait

    Please re-write to make it as characteristic as possible

    START YOUR RESPONSE DIRECTLY WITH SPEAKER 1:

    STRICTLY RETURN YOUR RESPONSE AS A LIST OF TUPLES OK? 

    IT WILL START DIRECTLY WITH THE LIST AND END WITH THE LIST NOTHING ELSE

    Example of response:
    [
        ("Speaker 1", "Welcome to our podcast, where we explore the latest advancements in AI and technology. I'm your host, and today we're joined by a renowned expert in the field of AI. We're going to dive into the exciting world of Llama 3.2, the latest release from Meta AI."),
        ("Speaker 2", "Hi, I'm excited to be here! So, what is Llama 3.2?"),
        ("Speaker 1", "Ah, great question! Llama 3.2 is an open-source AI model that allows developers to fine-tune, distill, and deploy AI models anywhere. It's a significant update from the previous version, with improved performance, efficiency, and customization options."),
        ("Speaker 2", "That sounds amazing! What are some of the key features of Llama 3.2?")
    ]
    """
    gen_params={
        "temperature": 1.0,
        "max_tokens": 100 * 1024,
    }

    fd = open(output_file, 'w', encoding="utf-8")
    if stream:

        # rewrited_text = ""
        for chunk in rewrite_mlx_api.generate(prompt=text, system_message=REWRITE_SYSTEMP_PROMPT, gen_params=gen_params, stream=True):
            # rewrited_text += chunk
            print(chunk, end='')
            fd.write(chunk)
            fd.flush()

    else:
        # transcript_text = transcript_llm_api.generate(prompt=text, system_message=TRANSCRIPT_SYSTEMP_PROMPT, gen_params=gen_params)
        # transcript_text = transcript_llm_api.generate(prompt=text, system_message=TRANSCRIPT_SYSTEMP_PROMPT_CN, gen_params=gen_params)

        rewrited_text = rewrite_mlx_api.generate(prompt=text, system_message=REWRITE_SYSTEMP_PROMPT, gen_params=gen_params)

        fd.write(rewrited_text)

    fd.close()

# # Import necessary libraries and configure settings
# import torch
import torchaudio
# torch._dynamo.config.cache_size_limit = 64
# torch._dynamo.config.suppress_errors = True
# torch.set_float32_matmul_precision('high')
# import ChatTTS
# chat_tts = ChatTTS.Chat()
# # Initialize and load the model: 
# CHATTTS_MODEL_PATH="/opt/local/TTS/ChatTTS"
# chat_tts.load(source="custom", custom_path=CHATTTS_MODEL_PATH, device=device, compile=False) # Set to True for better performance

# def generate_speaker_audio(text):

#     # Define the text input for inference (Support Batching)
#     # texts = [
#     #     "So we found being competitive and collaborative was a huge way of staying motivated towards our goals, so one person to call when you fall off, one person who gets you back on then one person to actually do the activity with.",
#     #     ]
#     texts = [text]

#     # Perform inference and play the generated audio
#     wavs = chat_tts.infer(texts)

#     # from IPython.display import Audio
#     # Audio(wavs[0], rate=24_000, autoplay=True)

#     # # Save the generated audio 
#     # torchaudio.save("output.wav", torch.from_numpy(wavs[0]), 24000)
#     return wavs[0], 24000


def do_tts(input_file = "podcast_ready.txt", output_file = "_podcast.mp3"):

    from speechless.tts.chattts import ChatTTSGenerator
    tts_generator = ChatTTSGenerator(model_path="/opt/local/TTS/ChatTTS", device=device, sampling_rate=24000, compile=False)

    from transformers import BarkModel, AutoProcessor, AutoTokenizer
    import torch
    import json
    import numpy as np

    # ----- Parler -----
    # from parler_tts import ParlerTTSForConditionalGeneration

    # PARLER_MODEL="/opt/local/TTS/parler-tts-mini-v1.1"
    # parler_model = ParlerTTSForConditionalGeneration.from_pretrained(PARLER_MODEL).to(device)
    # parler_tokenizer = AutoTokenizer.from_pretrained(PARLER_MODEL)
    
    # speaker1_description = """
    # Laura's voice is expressive and dramatic in delivery, speaking at a moderately fast pace with a very close recording that almost has no background noise.
    # """

    # def generate_speaker_audio(text):
    #     """Generate audio using ParlerTTS for Speaker 1"""
    #     logger.debug(f"parler generating speaker audio...")
    #     input_ids = parler_tokenizer(speaker1_description, return_tensors="pt").input_ids.to(device)
    #     prompt_input_ids = parler_tokenizer(text, return_tensors="pt").input_ids.to(device)
    #     generation = parler_model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
    #     audio_arr = generation.cpu().numpy().squeeze()
    #     logger.info(f"parler generating speaker audio done.")
    #     return audio_arr, parler_model.config.sampling_rate


    # ----- Bark -----
    # BARK_MODEL="/opt/local/TTS/bark"
    # bark_processor = AutoProcessor.from_pretrained(BARK_MODEL)
    # # bark_model = BarkModel.from_pretrained(BARK_MODEL, torch_dtype=torch.float16).to(device)
    # bark_model = BarkModel.from_pretrained(BARK_MODEL).to(device)
    # bark_sampling_rate = 24000

    # generated_segments = []
    # sampling_rates = []  # We'll need to keep track of sampling rates for each segment

    # def generate_speaker_audio(text):
    #     """Generate audio using Bark for Speaker 2"""
    #     logger.debug(f"bark generating speaker audio...")
    #     inputs = bark_processor(text, voice_preset="v2/zh_speaker_1").to(device)
    #     speech_output = bark_model.generate(**inputs, temperature=0.9, semantic_temperature=0.8)
    #     audio_arr = speech_output[0].cpu().numpy()
    #     logger.info(f"bark generating speaker audio done.")
    #     return audio_arr, bark_sampling_rate


    # def numpy_to_audio_segment(audio_arr, sampling_rate):
    #     """Convert numpy array to AudioSegment"""
    #     # Convert to 16-bit PCM
    #     audio_int16 = (audio_arr * 32767).astype(np.int16)
        
    #     # Create WAV file in memory
    #     byte_io = io.BytesIO()
    #     wavfile.write(byte_io, sampling_rate, audio_int16)
    #     byte_io.seek(0)
        
    #     # Convert to AudioSegment
    #     return AudioSegment.from_wav(byte_io)

    with open(input_file, 'r', encoding='utf-8') as file:
        podcast_text = file.read()

    final_audio = None

    for speaker, text in tqdm(ast.literal_eval(podcast_text), desc="Generating podcast segments", unit="segment"):
        # if speaker == "Speaker 1":
        #     audio_arr, rate = generate_speaker1_audio(text)
        # else:  # Speaker 2
        #     audio_arr, rate = generate_speaker2_audio(text)

        # audio_arr, rate = generate_speaker_audio(text)
        
        # # Convert to AudioSegment (pydub will handle sample rate conversion automatically)
        # audio_segment = numpy_to_audio_segment(audio_arr, rate)

        if speaker == "Speaker 1":
            speaker_id = 0
        else:
            speaker_id = 1
        audio_segment = tts_generator.generate_audio_segment(text, speaker=speaker_id)
        play_snd(audio_segment)
        
        # Add to final audio
        if final_audio is None:
            final_audio = audio_segment
        else:
            final_audio += audio_segment

    final_audio.export(output_file, 
                    format="wav", 
                    bitrate="192k",
                    parameters=["-q:a", "0"])

    # torchaudio.save(output_file, torch.from_numpy(final_audio), 24000)

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_path', type=str, default='“十四五”能源领域科技创新规划.pdf', help='Path to the PDF file')
    parser.add_argument("--do_extract_text", action="store_true", help="Extract text from PDF")
    parser.add_argument("--do_process_text", action="store_true", help="Process extracted text")
    parser.add_argument("--do_transcript", action="store_true", help="Generate transcript")
    parser.add_argument("--do_rewrite", action="store_true", help="Rewrite transcript")
    parser.add_argument("--do_tts", action="store_true", help="Generate TTS audio")
    parser.add_argument("--stream", action="store_true", help="Stream transcript generation")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    extraced_text_file = 'extracted_text.txt'
    clean_extraced_text_file = 'clean_extracted_text.txt'
    chunk_size = 1000

    if args.do_extract_text:
        do_extract_text(args.pdf_path, output_file=extraced_text_file)
    if args.do_process_text:
        do_process_text(input_file=extraced_text_file, output_file=clean_extraced_text_file, chunk_size=chunk_size)
    if args.do_transcript:
        do_transcript(input_file=clean_extraced_text_file, transcript_file='transcript.txt', stream=args.stream)
    if args.do_rewrite:
        do_rewrite(input_file='transcript.txt', output_file='podcast_ready.txt', stream=args.stream)
    if args.do_tts:
        do_tts(input_file='podcast_ready.txt', output_file='_podcast.wav')

if __name__ == "__main__":
    main()