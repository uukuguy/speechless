#!/usr/bin/env python
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import torch,os
from loguru import logger
from langchain.llms import HuggingFacePipeline
from transformers import AutoConfig, AutoTokenizer,AutoModelForCausalLM,pipeline,BitsAndBytesConfig

# Define the model name or path for the pretrained model
# os.environ["HF_TOKEN"] = [your own token should be here]


def generate_by_hf(args):
    # https://github.com/huggingface/blog/blob/main/accelerate-large-models.md
    from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
    config = AutoConfig.from_pretrained(model_name_or_path)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # device_map = infer_auto_device_map(model, dtype="bfloat16")
    from .default_device_map import llama_3_1_405b_device_map
    device_map = llama_3_1_405b_device_map
    # device_map="balanced_low_0"
    # device_map = infer_auto_device_map(model, no_split_module_classes=["OPTDecoderLayer"])
    # device_map = "auto"
    print(f"{device_map=}")

    # Count the number of GPUs available
    gpu_count = torch.cuda.device_count()
    device = 'auto'
    # Initialize the tokenizer with the specified model
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    # Configure BitsAndBytes for model quantization to reduce memory usage
    # This includes enabling 4-bit quantization, setting the quantization type, and using double quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )

    # Load the model from the pretrained path with specified configurations
    # This includes enabling SafeTensors for memory efficiency, setting the quantization config,
    # specifying the device map for GPU usage, allowing remote code execution, and setting the tensor data type
    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    model_kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                # quantization_config=bnb_config,
                                                **model_kwargs)

    # model = load_checkpoint_and_dispatch(
    #     model, checkpoint=model_name_or_path, device_map="balanced_low_0"
    # )

    # model.hf_device_map
    logger.debug(f"Start model.compile()...")
    # model.compile()
    # model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    model = torch.compile(model)
    # model = torch.compile(model, mode="max-autotune")
    logger.debug(f"model.compile() done.")

    generate_kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        # "stop": args.stop,
        "seed": args.seed,
        "stream": args.stream,
    }

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2500,
        return_full_text=True,
        do_sample=True,
        repetition_penalty=1.15,
        num_return_sequences=1,
        pad_token_id=2,
        model_kwargs = generate_kwargs
        # model_kwargs={"temperature": 0.3,
        #                             "top_p":0.95,
        #                             "top_k":40,
        #                             "max_new_tokens":2500},
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    template = """You are a Sport Analyst Assistant. Generally excelling at providing abstract commentary to elaborate a football action.
    Question: {query}
    Answer: """

    prompt_template = PromptTemplate(
        input_variables=["query"],
        template=template
    )
    #instantiate the chain
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)

    logger.info(f"Start to inference")
    resp=llm_chain.invoke("Kevin Debruyne is a Manchester City FC Midfielder, he just scored a goal against Liverpool FC")['text']
    logger.info(f"Inference done. {resp=}")
    # print(resp)

# model_name_or_path = "/opt/local/llm_models/huggingface.co/unsloth/Meta-Llama-3.1-405B-Instruct-bnb-4bit"
def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for text generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p for text generation")
    parser.add_argument("--top_k", type=int, default=40, help="Top k for text generation")
    parser.add_argument("--max_new_tokens", type=int, default=2500, help="Maximum number of new tokens to generate")
    parser.add_argument("--prompt", type=str, help="Prompt to run")
    parser.add_argument("--prompt_file", type=str, help="Prompt file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    return parser.parse_args()

def main():
    args = get_args()
    generate_by_hf(args)


if __name__ == "__main__":
    main()