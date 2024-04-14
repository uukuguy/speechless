from pprint import pprint

from .base import Task
from .code.mbpp import MBPP
from .code.humaneval import HumanEval
from .math.gsm8k import GSM8K
from .math.math import MATH
from .csqa.csqa import CommonsenseQA


TASK_REGISTRY = {
    'mbpp': MBPP,
    'humaneval': HumanEval,
    'gsm8k': GSM8K,
    'math': MATH,
    'csqa': CommonsenseQA
}


def get_task(task_name, args=None) -> Task:
    try:
        kwargs = {}
        return TASK_REGISTRY[task_name](**kwargs)
    except KeyError:
        print("Available tasks:")
        pprint(TASK_REGISTRY)
        raise KeyError(f"Missing task {task_name}")
    