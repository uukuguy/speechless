"""
Math Verification Reward Function

This module provides backward compatibility for the compute_score function
by importing the MathVerifyReward class from the reward_functions package.
"""

from speechless.reasoning.general_reasoner.reward_functions import CombinedReward, LengthReward, MathVerifyReward
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Legacy function to compute math verification score.
    
    This function is maintained for backward compatibility.
    It's recommended to use the MathVerifyReward class instead.
    
    Args:
        data_source: Source of the problem
        solution_str: Model's solution string
        ground_truth: Ground truth answer
        extra_info: Additional information
        
    Returns:
        Verification score between 0 and 1
    """
    length_reward = LengthReward(min_length=256, max_length=2048)
    math_reward = MathVerifyReward()

    combined_reward = CombinedReward(reward_functions=[length_reward, math_reward])

    return combined_reward.compute_reward(solution_str, reference=ground_truth)


# Example usage
if __name__ == "__main__":
    print("This module provides backward compatibility for the compute_score function.")
    print("For examples, see the example_usage function in reward_functions.utils module")
    
    # Simple example
    answer = "The answer is 42."
    ground_truth = "42"
    score = compute_score(None, answer, ground_truth)
    print(f"Score: {score}")
