import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main(args):
    # Initialize vLLM model
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Set generation parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.n_samples,
        top_p=args.top_p,
    )

    import json

    data = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    prompts = [item[args.prompt_key] for item in data]
    if args.system_prompt is None:
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False
            )
            for prompt in prompts
        ]
    else:
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
            )
            for prompt in prompts
        ]
    # Batch generation
    outputs = llm.generate(prompts, sampling_params)

    # Print results
    for index, output in enumerate(outputs):
        data[index]["test_outputs"] = [out.text for out in output.outputs]

    # Save results to file
    with open(args.output_file, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Test")

    # Model related parameters
    parser.add_argument("--model", type=str, required=True, help="Model path or name")
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=1, help="Model parallel size"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.8,
        help="GPU memory utilization, range 0-1",
    )

    # Sampling parameters
    parser.add_argument("--system_prompt", type=str, default=None, help="System prompt")
    parser.add_argument(
        "--temperature", type=float, default=0, help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=1, help="Top-p sampling threshold"
    )
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling count")
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=6144,
        help="Maximum token count for generation",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Number of samples to generate for each prompt",
    )

    # Dataset parameters
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input prompt file path, one prompt per line",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results.jsonl",
        help="Output result save path",
    )
    parser.add_argument(
        "--prompt_key", type=str, default="prompt", help="Output result save path"
    )

    args = parser.parse_args()
    main(args)
