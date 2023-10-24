from huggingface_hub import snapshot_download
import os
# os.environ['HF_ENDPOINT']="https://hf-mirror.com"
# MODEL_NAME='THUDM/agentlm-7b'
# snapshot_download(repo_id=MODEL_NAME,
#                   repo_type='model',
#                   local_dir='/opt/local/llm_models/huggingface.co/',
#                   resume_download=True)

def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--repo_type', type=str, default='model')
    parser.add_argument("--revision", type=str, default='main')
    parser.add_argument("--local_dir", type=str, default='/opt/local/llm_models/huggingface.co')
    parser.add_argument("--mirror_url", type=str, default="https://hf-mirror.com")

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    if args.mirror_url:
        os.environ['HF_ENDPOINT']=args.mirror_url
    snapshot_download(repo_id=args.model_name_or_path,
                      repo_type=args.repo_type,
                      revision=args.revision,
                      local_dir=f"{args.local_dir}/{args.model_name_or_path}",
                      resume_download=True)

if __name__ == '__main__':  
    main()