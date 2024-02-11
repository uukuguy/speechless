#!/usr/bin/env python
import os, json
import shutil
import torch
import transformers
from peft import PeftModel

script_path = os.path.dirname(os.path.realpath(__file__))


def merge_sparsetral_lora(base_model, trained_weights_dir, output_dir, num_experts=16, topk=4, adapter_dim=512):
    from speechless.finetune.moe.sparsetral.configuration_sparsetral import SparsetralConfig
    from speechless.finetune.moe.sparsetral.modeling_sparsetral import MistralForCausalLM

    model_config = SparsetralConfig.from_pretrained(base_model)
    model_config.pretraining_tp = 1  ## without tensor parallelism rank

    # Sparsetral Config
    model_config.moe_dtype = "bfloat16"
    model_config.adapter_dim = adapter_dim
    model_config.topk = topk
    model_config.moe_scaling = 1
    model_config.num_experts = num_experts
    model_config.output_router_logits = False

    moe_model = os.path.join(trained_weights_dir, "moe_model.bin")
    adapter_model = os.path.join(trained_weights_dir, "adapter_model")

    model = MistralForCausalLM.from_pretrained(
        base_model,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_model)
    model = model.merge_and_unload()

    moe_state_dict = torch.load(moe_model, map_location="cpu")
    new_moe_state_dict = {}
    for k, v in moe_state_dict.items():
        new_moe_state_dict[k.replace("base_model.model.", "")] = v

    model.load_state_dict(new_moe_state_dict, strict=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config_path = os.path.join(output_dir, "config.json")
    config = json.load(open(config_path, "r"))
    config["architectures"] = ["modeling_sparsetral.MistralForCausalLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_sparsetral.SparsetralConfig",
        "AutoModel": "modeling_sparsetral.MistralModel",
        "AutoModelForCausalLM": "modeling_sparsetral.MistralForCausalLM"
    }
    config["model_type"] = "sparsetral"
    config.pop("_name_or_path", None)
    json.dump(config, open(config_path, "w"), indent=2)

    shutil.copy2(
        f"{script_path}/sparsetral/configuration_sparsetral.py",
        os.path.join(output_dir, "configuration_sparsetral.py")
    )
    shutil.copy2(f"{script_path}/sparsetral/modeling_sparsetral.py", os.path.join(output_dir, "modeling_sparsetral.py"))


def merge_camelidae_lora(base_model, trained_weights_dir, output_dir, num_experts=8, topk=2, adapter_dim=64):
    from speechless.finetune.moe.camelidae.configuration_camelidae import CamelidaeConfig
    from speechless.finetune.moe.camelidae.modeling_camelidae import LlamaForCausalLM

    model_config = CamelidaeConfig.from_pretrained(base_model)
    model_config.pretraining_tp = 1  ## without tensor parallelism rank

    # Sparsetral Config
    model_config.moe_dtype = "bfloat16"
    model_config.adapter_dim = adapter_dim
    model_config.topk = topk
    model_config.moe_scaling = 1
    model_config.num_experts = num_experts
    model_config.output_router_logits = False

    moe_model = os.path.join(trained_weights_dir, "moe_model.bin")
    adapter_model = os.path.join(trained_weights_dir, "adapter_model")

    model = LlamaForCausalLM.from_pretrained(
        base_model,
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, adapter_model)
    model = model.merge_and_unload()

    moe_state_dict = torch.load(moe_model, map_location="cpu")
    new_moe_state_dict = {}
    for k, v in moe_state_dict.items():
        new_moe_state_dict[k.replace("base_model.model.", "")] = v

    model.load_state_dict(new_moe_state_dict, strict=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    config_path = os.path.join(output_dir, "config.json")
    config = json.load(open(config_path, "r"))
    config["architectures"] = ["modeling_sparsetral.LlamaForCausalLM"]
    config["auto_map"] = {
        "AutoConfig": "configuration_camelidae.CamelidaeConfig",
        "AutoModel": "modeling_camelidae.LlamaModel",
        "AutoModelForCausalLM": "modeling_camelidae.LlamaForCausalLM"
    }
    config["model_type"] = "camelidae"
    config.pop("_name_or_path", None)
    json.dump(config, open(config_path, "w"), indent=2)

    shutil.copy2(
        f"{script_path}/camelidae/configuration_camelidae.py", os.path.join(output_dir, "configuration_camelidae.py")
    )
    shutil.copy2(f"{script_path}/camelidae/modeling_camelidae.py", os.path.join(output_dir, "modeling_camelidae.py"))


def merge_moe_lora(base_model, trained_weights_dir, output_dir, num_experts=16, topk=4, adapter_dim=512):
    if "mistral" in base_model:
        merge_sparsetral_lora(base_model, trained_weights_dir, output_dir, num_experts, topk, adapter_dim)
    else:
        merge_camelidae_lora(base_model, trained_weights_dir, output_dir, num_experts, topk, adapter_dim)


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--trained_weights_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_experts", type=int, default=16)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--adapter_dim", type=int, default=512)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    merge_moe_lora(
        args.base_model, args.trained_weights_dir, args.output_dir, args.num_experts, args.topk, args.adapter_dim
    )


if __name__ == "__main__":
    main()
