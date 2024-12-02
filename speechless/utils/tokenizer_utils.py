from typing import Dict
import transformers
from transformers import AutoTokenizer

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def load_tokenizer(args):
    # Tokenizer
    tokenizer_kwargs = {
        "cache_dir": args.cache_dir,
        "padding_side": "left",
        "use_fast": False,
        "trust_remote_code": True,
    }
    print(f"---------- Original tokens----------")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, **tokenizer_kwargs)
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")

    # if "qwen" in args.model_name_or_path.lower():
    if False:
        tokenizer.eos_token = "<|endoftext|>"
        # tokenizer.unk_token = "<|extra_3|>"
        tokenizer.bos_token = "<|extra_2|>"
        tokenizer.pad_token = "<|extra_1|>"
    elif "glm-4" in args.model_name_or_path.lower():
        tokenizer.eos_token = "<sop>"
        tokenizer.unk_token = "<sop>"
    else:
        if tokenizer.bos_token_id is None:
            tokenizer.bos_token_id = 1
            tokenizer.bos_token = "<s>"
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = 2
            tokenizer.eos_token = "</s>"
        if tokenizer.unk_token_id is None:
            tokenizer.unk_token_id = 0
            tokenizer.unk_token = "<unk>"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = 0 # tokenizer.eos_token_id
            tokenizer.pad_token = tokenizer._convert_id_to_token(tokenizer.pad_token_id) #tokenizer.eos_token

    print(f"---------- Fixed tokens ----------")
    print(f"{tokenizer.pad_token=},{tokenizer.pad_token_id=}")
    # print(f"{tokenizer.unk_token=},{tokenizer.unk_token_id=}")
    print(f"{tokenizer.bos_token=},{tokenizer.bos_token_id=}")
    print(f"{tokenizer.eos_token=},{tokenizer.eos_token_id=}")

    return tokenizer
