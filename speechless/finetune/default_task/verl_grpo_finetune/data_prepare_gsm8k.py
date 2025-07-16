"""
Preprocess the GSM8k dataset to parquet format using generic preprocessing
Backwards compatible with original usage.

Usage:
	python data_prepare.py --local_dir data/verl/gsm8k
"""

import re
from data_prepare import DatasetConfig, preprocess_verl_dataset

def extract_gsm8k_solution(solution_str: str) -> str:
    """Default solution extractor for GSM8K format"""
    solution = re.search(r"#### ([\-]?[0-9\.\,]+)", solution_str)
    assert solution is not None, "No solution found in GSM8K format"
    final_solution = solution.group(1).replace(',', '')
    return final_solution


def get_gsm8k_config() -> DatasetConfig:
    """Get configuration matching original GSM8K setup"""
    return DatasetConfig(
        data_source='data/openai/gsm8k',
        dataset_name='main',
        input_key='question',
        output_key='answer',
        instruction_suffix="Let's think step by step and output the final answer after \"\"\".",
        ability='math',
        reward_style='rule',
        extract_answer_fn=extract_gsm8k_solution
    )


def gsm8k_main():
    """Main function for backward compatibility with original file"""
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/gsm8k', help='Output directory for processed datasets')

    args = parser.parse_args()

    config = get_gsm8k_config()
    processed = preprocess_verl_dataset(
        config=config,
        output_dir=args.local_dir,
        splits=['train', 'test']
    )
    
    print(f"Successfully processed GSM8K datasets: {list(processed.keys())}")
    
    # Original logic stops here - no HDFS support currently
    return processed


if __name__ == '__main__':
    gsm8k_main()
