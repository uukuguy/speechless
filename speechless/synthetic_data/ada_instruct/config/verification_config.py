import sys
import os
from pprint import pprint


VERIFICATIONARGS_REGISTRY = {
    'mbpp': (
        (),
        {
            'n_workers': 8,
            'timeout': 3.0
        }
    ),
    'humaneval': (
        (),
        {
            'n_workers': 8,
            'timeout': 3.0
        }
    )
}


def get_verification_config(verification_config_name):
    try:
        args, kwargs = VERIFICATIONARGS_REGISTRY[verification_config_name]
        return args, kwargs
        
    except KeyError:
        print("Available verification configs:")
        pprint(VERIFICATIONARGS_REGISTRY)
        raise KeyError(f"Missing prompter {verification_config_name}")
    