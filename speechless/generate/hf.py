#!/usr/bin/env python
import torch,os
import time
from loguru import logger
from transformers import AutoConfig, AutoTokenizer,AutoModelForCausalLM,pipeline,BitsAndBytesConfig

# from langchain.llms import HuggingFacePipeline
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# Define the model name or path for the pretrained model
# os.environ["HF_TOKEN"] = [your own token should be here]

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def generate_by_hf(args):
    """
    Generate text using a Hugging Face model.

    Args:
        args: An object containing the following attributes:
            - model_path (str): Path to the pre-trained model.
            - prompt_file (str, optional): Path to a file containing the prompt.
            - prompt (str, optional): The prompt text if not using a file.
            - temperature (float): The temperature for text generation.
            - max_new_tokens (int): The maximum number of new tokens to generate.
            - top_p (float): The top-p value for nucleus sampling.

    This function performs the following steps:
    1. Loads the tokenizer and model from the specified path.
    2. Prepares the input prompt (either from a file or directly provided).
    3. Sets up generation parameters.
    4. Generates text using the model.
    5. Decodes the generated tokens and prints the result.
    6. Calculates and prints the tokens per second (TPS) for the generation.

    The generated text is printed to the console along with the TPS metric.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": device,
            "trust_remote_code": True,
        }
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)

        prompt = open(args.prompt_file).read().strip() if args.prompt_file else args.prompt

        generate_kwargs = {
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "do_sample": True,
            "top_p": args.top_p,
        }

        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            gen_tokens = model.generate(**inputs, pad_token_id=tokenizer.pad_token_id, **generate_kwargs)
        
        generated_text = tokenizer.decode(gen_tokens[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        tps = (gen_tokens.shape[1] - inputs["input_ids"].shape[1]) / elapsed_time
        
        logger.info(f"Generated text: {generated_text}")
        logger.info(f"TPS (Tokens Per Second): {round(tps, 2)}")

        return generated_text

    except Exception as e:
        logger.error(f"Error in generate_by_hf: {str(e)}")
        return None


# def generate_by_hf(args):
#     # https://github.com/huggingface/blog/blob/main/accelerate-large-models.md
#     # from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
#     # config = AutoConfig.from_pretrained(args.model_name_or_path)
#     # with init_empty_weights():
#     #     model = AutoModelForCausalLM.from_config(config)

#     # # device_map = infer_auto_device_map(model, dtype="bfloat16")
#     # from .default_device_map import llama_3_1_405b_device_map
#     # device_map = llama_3_1_405b_device_map
#     # # device_map="balanced_low_0"
#     # # device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"])
#     # # device_map = "auto"
#     # print(f"{device_map=}")

#     # Count the number of GPUs available
#     gpu_count = torch.cuda.device_count()
#     device = 'auto'
#     # Initialize the tokenizer with the specified model
#     tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

#     # Configure BitsAndBytes for model quantization to reduce memory usage
#     # This includes enabling 4-bit quantization, setting the quantization type, and using double quantization
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_compute_dtype=torch.float16
#     )

#     # Load the model from the pretrained path with specified configurations
#     # This includes enabling SafeTensors for memory efficiency, setting the quantization config,
#     # specifying the device map for GPU usage, allowing remote code execution, and setting the tensor data type
#     model_kwargs = {
#         "torch_dtype": torch.bfloat16,
#         "device_map": device, #device_map,
#         "trust_remote_code": True,
#     }
#     # model_kwargs["attn_implementation"] = "flash_attention_2"
#     model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path,
#                                                 # quantization_config=bnb_config,
#                                                 **model_kwargs)

#     # model = load_checkpoint_and_dispatch(
#     #     model, checkpoint=model_name_or_path, device_map="balanced_low_0"
#     # )

#     # model.hf_device_map
#     logger.debug(f"Start model.compile()...")
#     # model.compile()
#     # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
#     model = torch.compile(model)
#     # model = torch.compile(model, mode="max-autotune")
#     logger.debug(f"model.compile() done.")

#     generate_kwargs = {
#         "temperature": args.temperature,
#         "max_new_tokens": args.max_new_tokens,
#         "top_p": args.top_p,
#         # "stop": args.stop,
#         "seed": args.seed,
#         "stream": args.stream,
#     }

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_length=2500,
#         return_full_text=True,
#         do_sample=True,
#         repetition_penalty=1.15,
#         num_return_sequences=1,
#         pad_token_id=2,
#         model_kwargs = generate_kwargs
#         # model_kwargs={"temperature": 0.3,
#         #                             "top_p":0.95,
#         #                             "top_k":40,
#         #                             "max_new_tokens":2500},
#     )
#     llm = HuggingFacePipeline(pipeline=pipe)
#     template = """You are a Sport Analyst Assistant. Generally excelling at providing abstract commentary to elaborate a football action.
#     Question: {query}
#     Answer: """

#     prompt_template = PromptTemplate(
#         input_variables=["query"],
#         template=template
#     )
#     #instantiate the chain
#     llm_chain = LLMChain(prompt=prompt_template, llm=llm)

#     logger.info(f"Start to inference")
#     resp=llm_chain.invoke("Kevin Debruyne is a Manchester City FC Midfielder, he just scored a goal against Liverpool FC")['text']
#     logger.info(f"Inference done. {resp=}")
#     # print(resp)

# model_name_or_path = "/opt/local/llm_models/huggingface.co/unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"
def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for text generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for text generation")
    parser.add_argument("--top_k", type=int, default=40, help="Top k for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=2500, help="Maximum number of new tokens to generate")
    parser.add_argument("--prompt", type=str, help="Prompt to run")
    parser.add_argument("--prompt_file", type=str, help="Prompt file")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stream", action="store_true", help="Enable streaming")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    args = get_args()
    generate_by_hf(args)


if __name__ == "__main__":
    main()