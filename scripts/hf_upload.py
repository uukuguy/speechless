import os
from huggingface_hub import HfApi

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--gguf_file', type=str)
    parser.add_argument('--orgnization', type=str, default='uukuguy')
    parser.add_argument('--repo_type', type=str, default='model')
    parser.add_argument("--revision", type=str, default='main')
    parser.add_argument("--local_dir", type=str, default='/opt/local/llm_models/huggingface.co')
    parser.add_argument("--mirror_url", type=str, default="https://hf-mirror.com")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    api = HfApi()

    model_id = f"{args.orgnization}/{os.path.basename(args.model_name_or_path)}"
    api.create_repo(model_id, exist_ok=True, repo_type=args.repo_type)
    api.upload_file(
        path_or_fileobj=args.model_name_or_path,
        path_in_repo=args.gguf_file,
        repo_id=model_id,
    )

if __name__ == '__main__':  
    main()