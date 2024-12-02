import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
import bitsandbytes as bnb
from loguru import logger

def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': f'cuda:{local_rank}'}
        max_memory = {'': max_memory[local_rank]}


    logger.info(f'loading base model {args.model_name_or_path}...')

    config = transformers.AutoConfig.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )

    # if "qwen" in args.model_name_or_path.lower():
    #     if args.bf16:
    #         config.bf16 = True
    #     elif args.fp16:
    #         config.fp16 = True

    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model_kwargs = {
        "cache_dir": args.cache_dir,
        # transformers-4.39.0.dev0
        # ValueError: You can't pass `load_in_4bit`or `load_in_8bit` as a kwarg when passing `quantization_config` argument at the same time.
        # "load_in_4bit": args.bits == 4,
        # "load_in_8bit": args.bits == 8,
        "device_map": device_map if not args.deepspeed else None,
        "max_memory": max_memory if not args.deepspeed else None,
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ) if args.bits in (4, 8) else None,
        "torch_dtype": (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        "trust_remote_code": args.trust_remote_code,
    }

    if args.flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    logger.info(f"{model_kwargs=}")
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, config=config, **model_kwargs)
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print('='*80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('='*80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype=(torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if checkpoint_dir is not None:
        logger.info("Loading adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, os.path.join(checkpoint_dir, 'adapter_model'), is_trainable=True)
    else:
        logger.info(f'adding LoRA modules...')
        modules = find_all_linear_names(args, model)
        print(f"LoRA modules: {modules}")
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if "norm" in name:
            module.to(compute_dtype)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                module.to(compute_dtype)

    return model