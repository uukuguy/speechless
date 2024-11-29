import yaml
import re

PROMPT_FORMATS = ['mistral', 'chatml', 'llama3', 'alpaca', 'phi', 'gemma']


def get_prompt_formatter(prompt_format):
    #@TODO: Make this configuration driven but still import base in a way that is secure
    if prompt_format == 'mistral':
        from .prompt_templates.mistral import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'chatml':
        from .prompt_templates.chatml import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'llama3':
        from .prompt_templates.llama3 import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'alpaca':
        from .prompt_templates.alpaca import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'phi':
        from .prompt_templates.phi import TrainingRecordHandler
        return TrainingRecordHandler
    elif prompt_format == 'gemma':
        from .prompt_templates.gemma import TrainingRecordHandler
        return TrainingRecordHandler

yaml_loader = yaml.SafeLoader
yaml_loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))


CONFIG_DEFAULTS = {
    "num_tokens": 100,
    "write_every": 1,
    "prompt": None,
    "train": False,
    "data": "data/",
    "temp": 0.6,
    "top_p": 1.0,
    "num_layers": 16,
    "evals_per_epoch": None,
    "saves_per_epoch": 2,
    "batch_size": 4,
    "iters": -1,
    "epochs": -1,
    "val_batches": 25,
    "learning_rate": 1e-5,
    "steps_per_report": 10,
    "steps_per_eval": 200,
    "resume_adapter_file": None,
    "test": False,
    "test_batches": 500,
    "max_seq_length": 2048,
    "seed": 0,
    "max_tokens": 100,
    "save_every": 100,
    "validation_scale": 5,
    "adapter_path": "adapters",
    "reporting_interval_proportion": 0.01, #10/1000

    "validation_interval_proportion": 0.2, #200/1000
    "validations_per_train_item": .5,
    "eval_proportion_of_total": 0,#0.25,

    "adapter_save_interval_proportion": .1,
    "ignore_chat_template": False,
    "colorize": False,
    "trust_remote_code": False,
    "lora_parameters": {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0},
    "grad_checkpoint": False
}
