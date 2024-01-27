import os
import ollama


def ollama_create(args):
    """
    # Simply specify the file path of gguf to generate the default-named ollama model, avoiding the tedious steps and lengthy parameters of using ollama create.
    # Default ollama_model_name = os.path.basename(gguf_file).replace('.', ':').replace('gguf', '')
    ```
    python -m speechless.infer \
        ollama_create \
        --gguf_file /path/to/deepseek-coder-6.7b-instruct.Q8_0.gguf \
        --ollama_model_name deepseek-coder-6.7b-instruct:Q8_0 
    ```

    ```
    # create_deepseek_coder:
    ollama create \
        deepseek_coder:6.7b-instruct.Q8_0 \
        -f Modelfile.deepseek_coder_6.7b_instruct_Q8_0
    ```
    """
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    modelfile_path = os.path.join(tmp_dir, "Modelfile." + os.path.basename(args.gguf_file))
    with open(modelfile_path, "w") as f:
        f.write(f"FROM {args.gguf_file}")

    ollama_model_name = args.ollama_model_name or os.path.basename(args.gguf_file).replace('.Q', ':Q').replace('.f16', ':f16').replace('.gguf', '')

    modelfile=open(modelfile_path).read()
    ollama.create(model=ollama_model_name, modelfile=modelfile)
    # cmd = f"ollama create {ollama_model_name} -f {modelfile_path}"
    # os.system(cmd)

    os.remove(modelfile_path)


def litellm_proxy(args):
    """
    # When using the ollama backend, there is no need to manually write a litellm configuration file. You can directly load all currently available models in ollama, including the default ChatGPT model.
    # Default litellm port 18341
    python -m speechless.infer \
        litellm_proxy \
        --litellm_port 18341 \
        
        
    # start_litellm_proxy:
    litellm --port 18341 \
        --drop_params \
        --config litellm_proxy.yaml
    """
    # tmp_dir = os.environ.get("TMPDIR", "/tmp")
    tmp_dir = os.path.expanduser("~/.speechless")
    os.makedirs(tmp_dir, exist_ok=True)
    config_file = os.path.join(tmp_dir, "litellm_proxy.yaml")

    # ollama_models = []
    # ollama_models_dir = os.path.expanduser("~/.ollama/models/manifests/registry.ollama.ai")
    # for dirpath, dirnames, filenames in os.walk(ollama_models_dir):
    #     if len(filenames) == 1 and len(dirnames) == 0:
    #         items = dirpath.split("/")
    #         repo_id = items[-2]
    #         model_name = items[-1]
    #         tag = filenames[0]
    #         if repo_id != "library":
    #             ollama_model_name = f"{model_name}:{tag}"
    #         else:
    #             ollama_model_name = f"{repo_id}/{model_name}:{tag}"
    #         litellm_model_name = f"{model_name}:{tag}"
    #         ollama_models.append((litellm_model_name, ollama_model_name))
    ollama_models = [ (os.path.basename(m['name']), m['name']) for m in ollama.list()['models']]

    with open (config_file, "w") as f:
        f.write("model_list:\n")
        f.write("  - model_name: gpt-3.5-turbo-1106\n")
        f.write("    litellm_params:\n")
        f.write("      model: openai/gpt-3.5-turbo-1106\n")
        f.write("      api_base: https://api.openai-proxy.org/v1\n")

        for litellm_model_name, ollama_model_name in ollama_models:
            f.write(f"  - model_name: {litellm_model_name}\n")
            f.write(f"    litellm_params:\n")
            f.write(f"      model: ollama/{ollama_model_name}\n")

    cmd = f"litellm --port {args.litellm_port} --drop_params --config {config_file}"
    os.system(cmd)


commands = {
    "ollama_create": ollama_create,
    "litellm_proxy": litellm_proxy,
}


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("cmd", type=str, choices=commands.keys(), help="command to run")

    parser.add_argument("--gguf_file", type=str)
    parser.add_argument("--ollama_model_name", type=str)

    parser.add_argument("--litellm_port", type=int, default=18341)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    func = commands[args.cmd]
    func(args)


if __name__ == "__main__":
    main()
