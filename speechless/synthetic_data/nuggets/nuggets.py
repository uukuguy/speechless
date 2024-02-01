# Modified from https://github.com/pldlgb/nuggets
import os, json
import argparse
import gc, ctypes
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from loguru import logger
from transformers import AutoTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM

from .meta_optimizer import AttnOptimWrapper
from .tasks.alpaca import AlpacaProbInference

os.environ['TOKENIZERS_PARALLELISM']="false"

def tabular_pretty_print(grid):
    lens = [max(map(len, col)) for col in zip(*grid)]

    fmt = " | ".join("{{:{}}}".format(x) for x in lens)
    table = [fmt.format(*row) for row in grid]

    sep = ["~" * len(table[0])]
    table = sep + table + sep

    res = []
    for idx, line in enumerate(table):
        if idx == 0 or idx == len(table) - 1:
            ps = "* {} *".format(line)
        else:
            ps = "| {} |".format(line)
        res.append(ps)
    return res


class AdvantageLogger:

    def __init__(self, direction="up"):
        self.log = []
        self.cur_best = 0.0
        self.is_better = np.greater_equal if direction == "up" else np.less

    def submit(self, idx, value):
        value = float(value)
        if self.is_better(value, self.cur_best):
            self.cur_best = value
            self.log.append((value, idx))
            return True

        return False

    def pretty_print(self):
        table = [["At", "Metric"]]
        for v, idx in self.log:
            table.append([str(idx), str(v)])

        for line in tabular_pretty_print(table):
            yield line


def init_random(seed):
    import random
    import os
    import numpy as np
    import torch

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


from pathlib import Path

checkpoints_root = Path("huggingface_cache")


def build_tokenizer(model_path=None, padding_side="left", use_fast=False):
    if not use_fast:
        tok = AutoTokenizer.from_pretrained(model_path, padding_side=padding_side, cache_dir=str(checkpoints_root))
    else:
        tok = PreTrainedTokenizerFast.from_pretrained(model_path, padding_side=padding_side, cache_dir=str(checkpoints_root))
    return tok


def build_model(model_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=str(checkpoints_root),
        device_map="auto",
        load_in_8bit=True,
    )
    model.eval()
    return model


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict,
    tokenizer,
    model,
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


# init_random(seed=SEED)


def the_shape(pack):
    if isinstance(pack, (list, tuple)):
        return f"{len(pack)} * {the_shape(pack[0])}"
    if isinstance(pack, torch.Tensor):
        return pack.size()


@torch.no_grad()
def do_infer_probs_zero(batched_choices_input):
    batched_choices_logprobs = []
    for batched_one_choice_input in batched_choices_input:
        batch_input_ids, batch_attention_mask, batch_choice_start, batch_choice_end = batched_one_choice_input
        bs = len(batch_input_ids)

        batched_logits = model(
            input_ids=batch_input_ids,  # [B, L']
            attention_mask=batch_attention_mask
        ).logits
        batched_output = F.log_softmax(batched_logits, dim=-1)  # [B, L', Vocab]

        batched_one_choice_logprobs = []
        for input_ids, choice_start, choice_end, lm_logprobs in zip(
            batch_input_ids, batch_choice_start, batch_choice_end, batched_output
        ):
            choice_tokens = input_ids[choice_start:choice_end].unsqueeze(1)  # [L, 1]
            choice_logprobs = lm_logprobs[choice_start - 1:choice_end - 1]  # [L, Vocab]

            extracted = torch.gather(choice_logprobs, -1, choice_tokens).squeeze(-1)

            choice_length = choice_end - choice_start
            lm_log_p = torch.sum(extracted).item()
            norm_lm_log_p = (lm_log_p / choice_length).item()

            choice_lm_info = {
                "lm_log_p": lm_log_p,
                "norm_lm_log_p": norm_lm_log_p
            }
            batched_one_choice_logprobs.append(choice_lm_info)
        batched_choices_logprobs.append(batched_one_choice_logprobs)
    return batched_choices_logprobs


@torch.no_grad()
def do_infer_probs(exemplar_attn_kv, exemplar_attn_mask, batched_choices_input):
    batched_choices_logprobs = []
    for batched_one_choice_input in batched_choices_input:
        batch_input_ids, batch_attention_mask, batch_choice_start, batch_choice_end = batched_one_choice_input
        bs = len(batch_input_ids)

        merged_attn_mask = torch.cat((exemplar_attn_mask.expand(bs, -1), batch_attention_mask), dim=1)
        # [B, #Heads, Length, Hidden]
        expand_exemplar_attn_kv = [[layer_k.expand((bs, -1, -1, -1)),
                                    layer_v.expand((bs, -1, -1, -1))] for layer_k, layer_v in exemplar_attn_kv]

        batched_logits = model(
            input_ids=batch_input_ids,  # [B, L']
            attention_mask=merged_attn_mask,  # [B, L + L']
            past_key_values=expand_exemplar_attn_kv,  # num_layers * 2 * [B, num_heads, L, H]
        ).logits
        batched_output = F.log_softmax(batched_logits, dim=-1)  # [B, L', Vocab]

        batched_one_choice_logprobs = []
        for input_ids, choice_start, choice_end, lm_logprobs in zip(
            batch_input_ids, batch_choice_start, batch_choice_end, batched_output
        ):
            choice_tokens = input_ids[choice_start:choice_end].unsqueeze(1)  # [L, 1]
            choice_logprobs = lm_logprobs[choice_start - 1:choice_end - 1]  # [L, Vocab]

            extracted = torch.gather(choice_logprobs, -1, choice_tokens).squeeze(-1)

            choice_length = choice_end - choice_start
            lm_log_p = torch.sum(extracted).item()
            norm_lm_log_p = (lm_log_p / choice_length).item()

            choice_lm_info = {
                "lm_log_p": lm_log_p,
                "norm_lm_log_p": norm_lm_log_p
            }
            batched_one_choice_logprobs.append(choice_lm_info)
        batched_choices_logprobs.append(batched_one_choice_logprobs)
    return batched_choices_logprobs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt_version", type=str, default="default")
    # parser.add_argument("--dataset", type=str, choices=task_mapper.keys())
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--debug", type=str2bool, default=False)

    parser.add_argument("--model_path", type=str)

    parser.add_argument("--prompt_path", type=str, default="/opt/local/datasets/alpaca_gpt4/alpaca_gpt4_data.json")
    parser.add_argument("--test_path", type=str, default="/opt/local/datasets/alpaca_gpt4/alpaca_gpt4_kmeans_100.json")
    parser.add_argument("--save_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=0)  # 0 for auto-detect, -1 for FORCE auto-detect
    parser.add_argument("--in_8bit", type=str2bool, default=False)
    parser.add_argument("--no_console", action="store_true", default=False)

    parser.add_argument("--exemplar_method", type=str, default="random", choices=["random", "written", "stratified"])
    # if `num_base_shot` is set, `num_k_shot * num_base_shot` is the number of exemplars to be sampled
    parser.add_argument("--num_k_shots", type=int, default=1)
    parser.add_argument("--start", type=int, default=0, help="start index of the exemplar set")
    parser.add_argument("--pace", type=int, default=0, help="start + pace is the end index of the exemplar set")
    parser.add_argument("--num_eval", type=float, default=1)
    parser.add_argument("--num_prompt", type=float, default=1.0)

    parser.add_argument("--kv_iter", type=int, default=1)
    parser.add_argument("--step_size", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    task_name = f"seed{args.seed}_main{args.kv_iter}"
    # task_name += f"_{args.prompt_version}"
    task_name += f"_{args.exemplar_method}{'' if args.exemplar_method == 'written' else args.num_k_shots}"
    task_name += f"_eps{args.step_size}_beta{args.momentum}"

    logger.info(f"Task Prepared: {task_name}")

    # 1. load model, tokenizer
    tokenizer = build_tokenizer(model_path=args.model_path, padding_side="right")
    model = build_model(model_path=args.model_path)
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict={"pad_token": "[PAD]"},
        tokenizer=tokenizer,
        model=model,
    )
    torch.autograd.set_grad_enabled(False)
    logger.info(f"Model loaded: {args.model_path}")

    # 2. load dataset (with demonstrations)
    TaskHandler = AlpacaProbInference
    task_agent = TaskHandler(args.prompt_version, args.prompt_path, args.test_path)
    task_agent.set_seed(args.seed)
    task_agent.do_load()
    dataset = task_agent.mk_result_dataset(tokenizer, args)

    logger.info(f"Selected batch_size: {args.batch_size}")

    loader = DataLoader(dataset, shuffle=False, drop_last=False, batch_size=args.batch_size, num_workers=2)

    logger.info("Running ...")

    # Zero Shot Forward
    generated_zero_info = []
    for batch_input in tqdm(loader, desc=f"Zero Shot Forward"):
        batch_input = [[e.cuda() for e in batch_choice] for batch_choice in batch_input]
        # batch_input = [[e for e in batch_choice] for batch_choice in batch_input]
        batch_output = do_infer_probs_zero(batch_input, )  # [batch_of_choice0, batch_of_choice1, ...]
        zipped_zero_logprobs = list(zip(*batch_output))  # batch * (choice0, choice1, ...)
        generated_zero_info.extend(zipped_zero_logprobs)

    # Set demonstrations
    if args.exemplar_method == "written":
        exemplar_str = task_agent.handcrafted_exemplars()
    elif args.exemplar_method == "random":
        exemplar_str = task_agent.random_selected_exemplars(args.num_k_shots)
    elif args.exemplar_method == "stratified":
        exemplar_str = task_agent.stratified_sampling(args.num_k_shots)
    else:
        raise ValueError(f"Unknown `args.exemplar_method == {args.exemplar_method}`")

    # Demonstrations Slice
    # logger.info("before slice : ", len(exemplar_str))
    exemplar_str = exemplar_str[:int(len(exemplar_str) * args.num_prompt)]
    # logger.info("after slice : ", len(exemplar_str))

    text_width = os.get_terminal_size().columns - 30

    rate_dict = {}
    score_dict = {}
    start = args.start
    if args.pace > 0:
        end = min(start + args.pace, len(exemplar_str))
    else:
        end = len(exemplar_str)
    logger.info(str(start) + "----------------" + str(end))

    for i in tqdm(range(start, end)):
        rate_dict[i] = []
        exemplar_input_ids, exemplar_attn_mask = [e.cuda() for e in dataset.tokenize_demonstration(exemplar_str[i])]
        meta_optim = AttnOptimWrapper(model, step_size=args.step_size, momentum=args.momentum)
        meta_optim.init()

        # trace_logger = AdvantageLogger()

        for idx in range(args.kv_iter):
            exemplar_kv = meta_optim.step(exemplar_input_ids)

            generated_info = []  # question * [choice0_prob, choice1_prob]
            for batch_input in tqdm(loader, desc=f"idx={idx}"):
                batch_input = [[e.cuda() for e in batch_choice] for batch_choice in batch_input]
                # batch_input = [[e for e in batch_choice] for batch_choice in batch_input]
                batch_output = do_infer_probs(
                    exemplar_kv,
                    exemplar_attn_mask.unsqueeze(0),
                    batch_input,
                )  # [batch_of_choice0, batch_of_choice1, ...]
                zipped_logprobs = list(zip(*batch_output))  # batch * (choice0, choice1, ...)
                generated_info.extend(zipped_logprobs)

            rate, metric, score = task_agent.post_process(
                generated_info, metric_output=False, generated_zero_info=generated_zero_info
            )
            rate_dict[i].append(rate[0])
            score_dict[i] = [list(i) for i in score]
            metric_s = json.dumps(metric, indent=None)
            logger.info(f"Iter={idx+1: <3} | {metric_s}")
            # trace_logger.submit(idx + 1, metric["lm_log_p"])
            # gc.collect()

        # for line in trace_logger.pretty_print():
        #     logger.info(line)

    json_data = json.dumps(rate_dict)

    # 将JSON字符串写入文件
    os.makedirs(args.save_path, exist_ok=True)
    with open(f'{args.save_path}/{start}_{end}_score.json', 'w') as file:
        file.write(json_data)

    score_data = json.dumps(score_dict)
    with open(f'{args.save_path}/{start}_{end}_raw_score.json', 'w') as file:
        file.write(score_data)
