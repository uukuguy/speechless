from .alpaca import AlpacaProbInference

task_mapper = {
    "alpaca": AlpacaProbInference,
}


def load_task(name):
    if name not in task_mapper.keys():
        raise ValueError(f"Unrecognized dataset `{name}`")

    return task_mapper[name]
