"""
Preprocess the GSM8k dataset to parquet format using generic preprocessing
Backwards compatible with original usage.

Usage:
	python data_prepare.py --local_dir data/verl/gsm8k
"""

import re
from data_prepare_base import DatasetConfig, generic_main

def extract_gsm8k_solution(solution_str: str) -> str:
    """Default solution extractor for GSM8K format"""
    solution = re.search(r"#### ([\-]?[0-9\.\,]+)", solution_str)
    assert solution is not None, "No solution found in GSM8K format"
    final_solution = solution.group(1).replace(',', '')
    return final_solution


def format_gsm8k_prompt(question: str) -> str:
    """Format GSM8K prompt with instruction"""
    return f"{question} Let's think step by step and output the final answer after \"\"\"."


def get_gsm8k_config() -> DatasetConfig:
    """Get configuration matching original GSM8K setup"""
    return DatasetConfig(
        data_source='data/openai/gsm8k',
        dataset_name='main',
        input_key='question',
        output_key='answer',
        ability='math',
        reward_style='rule',
        extract_answer_fn=extract_gsm8k_solution,
        prompt_format_fn=format_gsm8k_prompt
    )


def gsm8k_main():
    """Main function for backward compatibility with original file"""
    config = get_gsm8k_config()
    
    # Define optional custom arguments specific to GSM8K
    custom_args = {
        'local_dir': {'default': '~/data/gsm8k', 'help': 'Output directory for GSM8K datasets'},
    }
    
    return generic_main(config, custom_args)


if __name__ == '__main__':
    gsm8k_main()
