
"""
# verl.trainer.main_ppo.get_custom_reward_fn
custom_reward_function.path=./reward_score.py
custom_reward_function.name=compute_score

verl.workers.reward_manager.NaiveRewardManager
verl.workers.reward_manager.PrimeRewardManager
"""
def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info=None):
    """
    GRPO finetune reward score function
    data_source: the data source
    solution_str: the solution text.
    ground_truth: the ground truth text.
    extra_info: you can use it to pass additional information
    """
    return 0
