# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Training Module for General Reasoner

This module provides functionality to train a general reasoning model using 
Proximal Policy Optimization (PPO) with Ray for distributed computing.
It handles model initialization, worker setup, reward function configuration,
and training execution.
"""

import os
import sys
import importlib.util
from typing import Optional, Dict, Any, Callable, Type

import ray
import hydra
from omegaconf import OmegaConf
from pprint import pprint

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager, Role
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer, hf_processor


def load_custom_reward_function(file_path: str, function_name: str, reward_kwargs: Dict[str, Any]) -> Callable:
    """
    Load a custom reward function from a specified file.
    
    Args:
        file_path: Path to the Python file containing the reward function
        function_name: Name of the function to load
        reward_kwargs: Additional keyword arguments to pass to the reward function
        
    Returns:
        A callable reward function that includes the specified keyword arguments
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist
        AttributeError: If the function doesn't exist in the module
        RuntimeError: If there's an error loading the module
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Reward function file '{file_path}' not found.")

    spec = importlib.util.spec_from_file_location("custom_module", file_path)
    module = importlib.util.module_from_spec(spec)
    
    try:
        sys.modules["custom_module"] = module
        spec.loader.exec_module(module)
    except Exception as e:
        raise RuntimeError(f"Error loading module from '{file_path}': {e}")

    if not hasattr(module, function_name):
        raise AttributeError(f"Reward function '{function_name}' not found in '{file_path}'.")

    print(f"Using customized reward function '{function_name}' from '{file_path}'")
    raw_fn = getattr(module, function_name)

    def wrapped_fn(*args, **kwargs):
        return raw_fn(*args, **kwargs, **reward_kwargs)

    return wrapped_fn


def get_custom_reward_fn(config: Dict[str, Any]) -> Optional[Callable]:
    """
    Extract and load a custom reward function from the configuration.
    
    Args:
        config: Configuration dictionary containing reward function details
        
    Returns:
        A callable reward function or None if no custom function is specified
    """
    reward_fn_config = config.get("custom_reward_function") or {}
    file_path = reward_fn_config.get("path")
    
    if not file_path:
        return None

    function_name = reward_fn_config.get("name")
    reward_kwargs = dict(reward_fn_config.get("reward_kwargs", {}))
    
    return load_custom_reward_function(file_path, function_name, reward_kwargs)


def get_reward_manager_class(reward_manager_name: str) -> Type:
    """
    Get the appropriate reward manager class based on the configuration.
    
    Args:
        reward_manager_name: Name of the reward manager to use
        
    Returns:
        The reward manager class
        
    Raises:
        NotImplementedError: If the specified reward manager is not implemented
    """
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        return NaiveRewardManager
    elif reward_manager_name == 'prime':
        from verl.workers.reward_manager import PrimeRewardManager
        return PrimeRewardManager
    elif reward_manager_name == 'batch':
        from verl.workers.reward_manager import BatchRewardManager
        return BatchRewardManager
    elif reward_manager_name == 'dapo':
        from verl.workers.reward_manager import DAPORewardManager
        return DAPORewardManager
    else:
        raise NotImplementedError(f"Reward manager '{reward_manager_name}' is not implemented")


def get_worker_classes(strategy: str):
    """
    Get the appropriate worker classes and worker group class based on the strategy.
    
    Args:
        strategy: The strategy to use ('fsdp' or 'megatron')
        
    Returns:
        A tuple containing (ActorRolloutRefWorker, CriticWorker, ray_worker_group_cls)
        
    Raises:
        NotImplementedError: If the specified strategy is not implemented
    """
    if strategy == 'fsdp':
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        return ActorRolloutRefWorker, CriticWorker, RayWorkerGroup
    elif strategy == 'megatron':
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        return ActorRolloutRefWorker, CriticWorker, NVMegatronRayWorkerGroup
    else:
        raise NotImplementedError(f"Strategy '{strategy}' is not implemented")


def setup_reward_model_worker(config):
    """
    Set up the reward model worker based on the configuration.
    
    Args:
        config: Configuration dictionary containing reward model details
        
    Returns:
        The reward model worker class
        
    Raises:
        NotImplementedError: If the specified strategy is not implemented
    """
    if not config.reward_model.enable:
        return None
        
    if config.reward_model.strategy == 'fsdp':
        from verl.workers.fsdp_workers import RewardModelWorker
    elif config.reward_model.strategy == 'megatron':
        from verl.workers.megatron_workers import RewardModelWorker
    elif config.reward_model.strategy == 'verifier':
        from verifier import RewardModelWorker
    else:
        raise NotImplementedError(f"Reward model strategy '{config.reward_model.strategy}' is not implemented")
    
    return RewardModelWorker


def create_reward_function(config, tokenizer, compute_score, num_examine=0):
    """
    Create a reward function based on the configuration.
    
    Args:
        config: Configuration dictionary
        tokenizer: The tokenizer to use
        compute_score: Custom compute score function
        num_examine: Number of examples to examine
        
    Returns:
        A reward function
    """
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    reward_manager_cls = get_reward_manager_class(reward_manager_name)
    
    reward_kwargs = dict(config.reward_model.get("reward_kwargs", {}))
    
    return reward_manager_cls(
        tokenizer=tokenizer,
        num_examine=num_examine,
        compute_score=compute_score,
        reward_fn_key=config.data.reward_fn_key,
        **reward_kwargs
    )


@ray.remote(num_cpus=1)  # Ensure main_task is not scheduled on head node
class TaskRunner:
    """
    Ray remote class for running the PPO training task.
    """
    
    def run(self, config):
        """
        Run the PPO training task with the given configuration.
        
        Args:
            config: Configuration for the training task
            
        Returns:
            None
        """
        # Print and resolve configuration
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # Download checkpoint from HDFS and initialize tokenizer
        local_path = copy_to_local(config.actor_rollout_ref.model.path)
        trust_remote_code = config.data.get('trust_remote_code', False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, use_fast=True)  # Used for multimodal LLM

        # Get worker classes based on strategy
        strategy = config.actor_rollout_ref.actor.strategy
        assert strategy == config.critic.strategy, "Actor and critic strategies must match"
        
        ActorRolloutRefWorker, CriticWorker, ray_worker_group_cls = get_worker_classes(strategy)

        # Set up role-worker mapping
        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
        }

        # Configure resource pools
        global_pool_id = 'global_pool'
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
        }

        # Set up reward model if enabled
        RewardModelWorker = setup_reward_model_worker(config)
        if RewardModelWorker:
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # Set up reference policy if needed
        if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
            role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
            mapping[Role.RefPolicy] = global_pool_id

        # Create reward functions
        compute_score = get_custom_reward_fn(config)
        reward_fn = create_reward_function(config, tokenizer, compute_score)
        
        # Always use function-based RM for validation
        val_reward_fn = create_reward_function(config, tokenizer, compute_score, num_examine=1)
        
        # Set up resource pool manager
        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, 
            mapping=mapping
        )

        # Initialize and run trainer
        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn
        )
        
        trainer.init_workers()
        trainer.fit()


def run_ppo(config) -> None:
    """
    Run the PPO training with the given configuration.
    
    Args:
        config: Configuration for the training
        
    Returns:
        None
    """
    # Set environment variables to resolve SGLang conflict with Ray devices isolation
    # TODO: Find a permanent solution for this issue
    os.environ["ENSURE_CUDA_VISIBLE_DEVICES"] = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(runtime_env={
            'env_vars': {
                'TOKENIZERS_PARALLELISM': 'true',
                'NCCL_DEBUG': 'WARN',
                'VLLM_LOGGING_LEVEL': 'WARN'
            }
        })

    # Create and run the task runner
    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@hydra.main(config_path='./verl/trainer/config', config_name='ppo_trainer', version_base=None)
def main(config):
    """
    Main entry point for the PPO training script.
    
    Args:
        config: Hydra configuration object
        
    Returns:
        None
    """
    run_ppo(config)


if __name__ == '__main__':
    main()
