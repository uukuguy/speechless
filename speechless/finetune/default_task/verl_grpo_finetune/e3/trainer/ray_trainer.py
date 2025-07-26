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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict, List, Union, Optional
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

import copy
import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
import tensordict
from tensordict import TensorDict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import (
    RayResourcePool,
    RayWorkerGroup,
    RayClassWithInitArgs,
)
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.utils.seqlen_balancing import (
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

from verl.trainer.ppo.ray_trainer import Role, AdvantageEstimator, ResourcePoolManager
from ..data.rl_dataset import RLHFDataset, collate_fn

WorkerType = Type[Worker]


import torch
from verl.utils.torch_functional import masked_mean


def repeat(
    batch,
    non_tensor_batch,
    meta_info,
    repeat_times: Union[int, List[int]] = 2,
    interleave=True,
):
    """
    Repeat the batch data a specified number of times.

    Args:
        repeat_times (int): Number of times to repeat the data.
        interleave (bool): Whether to interleave the repeated data.

    Returns:
        DataProto: A new DataProto with repeated data.
    """
    if isinstance(repeat_times, int):
        if batch is not None:
            if interleave:
                # Interleave the data
                repeated_tensors = {
                    key: tensor.repeat_interleave(repeat_times, dim=0)
                    for key, tensor in batch.items()
                }
            else:
                # Stack the data
                repeated_tensors = {
                    key: tensor.unsqueeze(0)
                    .expand(repeat_times, *tensor.shape)
                    .reshape(-1, *tensor.shape[1:])
                    for key, tensor in batch.items()
                }

            repeated_batch = TensorDict(
                source=repeated_tensors,
                batch_size=(batch.batch_size[0] * repeat_times,),
            )
        else:
            repeated_batch = None

        repeated_non_tensor_batch = {}
        for key, val in non_tensor_batch.items():
            if interleave:
                repeated_non_tensor_batch[key] = np.repeat(val, repeat_times, axis=0)
            else:
                repeated_non_tensor_batch[key] = np.tile(
                    val, (repeat_times,) + (1,) * (val.ndim - 1)
                )

        return DataProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=meta_info,
        )
    else:
        assert len(repeat_times) == batch.batch_size[0]
        repeated_tensors = {}
        for key, tensor in batch.items():
            tensor_list = []
            for n, item in zip(repeat_times, tensor):
                expanded_item = item.unsqueeze(0).expand(n, *item.shape)
                tensor_list.append(expanded_item)
            repeated_tensors[key] = torch.cat(tensor_list, dim=0)

        repeated_batch = TensorDict(
            source=repeated_tensors,
            batch_size=(sum(repeat_times),),
        )

        repeated_non_tensor_batch = {}
        for key, val in non_tensor_batch.items():
            total_size = sum(repeat_times)
            repeated_val = np.empty(total_size, dtype=object)
            current_idx = 0
            for n, item in zip(repeat_times, val):
                for i in range(n):
                    repeated_val[current_idx + i] = copy.deepcopy(item)
                current_idx += n
            repeated_non_tensor_batch[key] = repeated_val

        return DataProto(
            batch=repeated_batch,
            non_tensor_batch=repeated_non_tensor_batch,
            meta_info=meta_info,
        )


def apply_kl_penalty(
    data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"
):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch["attention_mask"]
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {
        "actor/reward_kl_penalty": current_kl,
        "actor/reward_kl_penalty_coeff": beta,
    }

    return data, metrics


def compute_response_mask(data: DataProto):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    length_per_prompt: Optional[torch.Tensor] = None,
    length_penalty_coeff: Optional[float] = 0,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    def get_panenty(length: torch.Tensor) -> torch.Tensor:
        return length * length_penalty_coeff / (length * length_penalty_coeff + 1)

    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}
    id2length = defaultdict(list)

    all_error_count = 0
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2length[index[i]].append(length_per_prompt[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                ts = torch.tensor(id2score[idx])
                if torch.all(ts == 0):
                    all_error_count += 1
                if torch.all(ts > 1):
                    if length_penalty_coeff > 0:
                        panenty = get_panenty(torch.tensor(id2length[idx]))
                        ts = ts - panenty

                id2mean[idx] = torch.mean(ts)
                id2std[idx] = torch.std(ts)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores, all_error_count


def compute_advantage(
    data: DataProto,
    adv_estimator,
    gamma=1.0,
    lam=1.0,
    length_penalty_coeff: Optional[float] = 0,
):
    # Back-compatible with trainers that do not compute response mask in fit
    all_error_count = 0
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            eos_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        data = get_length(data)
        advantages, returns, all_error_count = compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            length_per_prompt=data.batch["response_length_per_prompt"],
            length_penalty_coeff=length_penalty_coeff,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            eos_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            eos_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data, all_error_count


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


def get_length(data: DataProto):
    responses = data.batch["responses"]
    # rank = [i for i in data.non_tensor_batch["rank"]]
    # rank = torch.tensor(rank, dtype=torch.float32)
    # rank[rank == -100] = 0
    response_length = responses.size(1)
    # token_level_scores = data.batch["token_level_scores"]  # (bsz,response_length)
    # token_level_rewards = data.batch["token_level_rewards"]
    attention_mask = data.batch["attention_mask"]

    response_mask = attention_mask[:, -response_length:]
    response_length_per_prompt = response_mask.sum(dim=-1)
    data.batch["response_length_per_prompt"] = response_length_per_prompt
    return data


def apply_length_penalty(
    data: DataProto, penalty_coeff: float = 1e-3, penalty_threshold: float = 0.2
):
    def get_panenty(length: torch.Tensor) -> torch.Tensor:
        return length * penalty_coeff / (length * penalty_coeff + 1)

    responses = data.batch["responses"]
    rank = [i for i in data.non_tensor_batch["rank"]]
    rank = torch.tensor(rank, dtype=torch.float32)
    rank[rank == -100] = 1
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]  # (bsz,response_length)
    token_level_rewards = data.batch["token_level_rewards"]
    attention_mask = data.batch["attention_mask"]

    response_mask = attention_mask[:, -response_length:]
    response_length_per_prompt = response_mask.sum(dim=-1)
    length_penalty = get_panenty(response_length_per_prompt)

    # length_penalty = (1 - rank) * length_penalty  # (bsz,)
    zero_mask = rank > penalty_threshold
    length_penalty[zero_mask] = 0

    length_penalty = length_penalty.unsqueeze(-1).tile(
        [1, token_level_scores.shape[1]]
    )  # (bsz,response_length)

    score_mask = token_level_scores != 0
    token_level_rewards[score_mask] = (
        token_level_scores[score_mask] - length_penalty[score_mask]
    )
    data.batch["token_level_rewards"] = token_level_rewards
    return data


def get_rollout_n_per_prompt(
    rank: np.ndarray,
    total_rollout_n: int,
    rollout_n_min: int = 2,
    rollout_n_max: int = 16,
) -> List[int]:
    """
    dynamic rollout_n
    difficult samples rollout_n will be larger
    """

    rollout_counts = np.full_like(rank, rollout_n_min)
    mask = rank == -100
    rank = rank.copy()  # 创建副本避免修改原始数据
    rank[mask] = 0

    # 归一化rank
    rank_sum = rank.sum()
    if rank_sum > 0:  # 避免除以零
        rank = rank / (rank_sum + 1e-8)

    # 计算额外的rollout分配
    extra_counts = (total_rollout_n - rollout_counts.sum()) * rank
    rollout_counts = rollout_counts + extra_counts

    # numpy数组转整数
    rollout_counts = np.floor(rollout_counts).astype(np.int64)

    # 处理舍入误差，确保总和等于total_rollout_n
    diff = total_rollout_n - rollout_counts.sum()
    if diff > 0:
        # 按rank降序排列索引
        sorted_indices = np.argsort(-rank)  # 降序排列
        for i in sorted_indices:
            if diff > 0:
                if rollout_counts[i] < rollout_n_max:
                    add_count = min(diff, rollout_n_max - rollout_counts[i])
                    rollout_counts[i] += add_count
                    diff -= add_count
            if diff <= 0:
                break

    return rollout_counts.tolist()


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
    ):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        # dynamic rollout_n
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine

        self.rollout_n_min = config.actor_rollout_ref.rollout.n
        self.rollout_n_max = config.actor_rollout_ref.rollout.n
        self.rollout_n = config.actor_rollout_ref.rollout.n

        self.rollout_n_low = config.actor_rollout_ref.rollout.n_low
        self.rollout_n_high = config.actor_rollout_ref.rollout.n_high
        self.rollout_update = config.actor_rollout_ref.rollout.n_update

        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert (
                Role.ActorRollout in role_worker_mapping
            ), f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(
                config.algorithm.kl_ctrl
            )

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = (
            config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        )
        assert (
            real_train_batch_size % n_gpus == 0
        ), f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. "
                        f"Please remove '{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(
                config.critic.ppo_micro_batch_size,
                config.critic.ppo_micro_batch_size_per_gpu,
                "critic",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size,
                config.reward_model.micro_batch_size_per_gpu,
                "reward_model",
            )

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert (
                config.data.train_batch_size
                >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            )
            sp_size = config.actor_rollout_ref.actor.get(
                "ulysses_sequence_parallel_size", 1
            )
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert (
                    config.actor_rollout_ref.actor.ppo_mini_batch_size
                    % config.actor_rollout_ref.actor.ppo_micro_batch_size
                    == 0
                )
                assert (
                    config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size
                    >= n_gpus
                )

        if (
            config.algorithm.use_kl_in_reward
            and config.actor_rollout_ref.actor.use_kl_loss
        ):
            print(f"NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert (
                    config.critic.ppo_mini_batch_size
                    % config.critic.ppo_micro_batch_size
                    == 0
                )
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            if (
                config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
                > 1
                or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1)
                > 1
            ):
                assert (
                    config.actor_rollout_ref.model.use_remove_padding
                ), "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert (
                    config.critic.model.use_remove_padding
                ), "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert (
                config.actor_rollout_ref.rollout.temperature > 0
            ), "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(
            parquet_files=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
        )
        assert self.train_dataset.truncation == self.config.data.get(
            "truncation", "error"
        ), f'dataset truncation {self.train_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get("seed", 1))
            sampler = RandomSampler(
                data_source=self.train_dataset, generator=train_dataloader_generator
            )
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            num_workers=8,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=sampler,
        )

        self.val_dataset = RLHFDataset(
            parquet_files=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            image_key=self.config.data.get("image_key", "images"),
            max_prompt_length=self.config.data.max_prompt_length,
            filter_prompts=True,
            return_raw_chat=self.config.data.get("return_raw_chat", False),
            truncation=self.config.data.get("truncation", "error"),
            filter_overlong_prompts=self.config.data.filter_overlong_prompts,
        )
        assert self.val_dataset.truncation == self.config.data.get(
            "truncation", "error"
        ), f'dataset truncation {self.val_dataset.truncation} must be the same as config {self.config.data.get("truncation", "error")}'
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1
        assert (
            len(self.val_dataloader) == 1
        ), "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f"Size of train dataloader: {len(self.train_dataloader)}")

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = (
            len(self.train_dataloader) * self.config.trainer.total_epochs
        )

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = (
                total_training_steps
            )
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(
            self.config.trainer.logger, samples, self.global_steps
        )

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        length_per_source = {}
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            # we only do validation on rule-based rm
            if (
                self.config.reward_model.enable
                and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model"
            ):
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in input_ids
            ]
            sample_inputs.extend(input_texts)

            if "multi_modal_inputs" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=[
                        "raw_prompt_ids",
                        "multi_modal_data",
                        "multi_modal_inputs",
                    ],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
                test_gen_batch, self.actor_rollout_wg.world_size
            )
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(
                test_gen_batch_padded
            )

            # unpad
            test_output_gen_batch = unpad_dataproto(
                test_output_gen_batch_padded, pad_size=pad_size
            )
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [
                self.tokenizer.decode(ids, skip_special_tokens=True)
                for ids in output_ids
            ]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            for i in range(len(test_batch)):
                data_item = test_batch[i]
                prompt_length = data_item.batch["prompts"].shape[-1]
                data_source = data_item.non_tensor_batch["data_source"]
                valid_response_length = data_item.batch["attention_mask"][
                    prompt_length:
                ].sum()
                if data_source not in length_per_source:
                    length_per_source[data_source] = []
                length_per_source[data_source].append(valid_response_length)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(
                test_batch.non_tensor_batch.get(
                    "data_source", ["unknown"] * reward_tensor.shape[0]
                )
            )

        self._maybe_log_val_generations(
            inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
        )

        reward_tensor = (
            torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()
        )  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f"val/test_score/{data_source}"] = np.mean(rewards)

        # ============= 长度日志 =============
        for data_source in length_per_source:
            avg_length = sum(length_per_source[data_source]) / len(
                length_per_source[data_source]
            )
            if isinstance(avg_length, torch.Tensor):
                avg_length = avg_length.item()
            metric_dict[f"val/test_length/{data_source}"] = avg_length
        # ============= 长度日志 =============

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.ActorRollout
            )
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool][
                "actor_rollout"
            ] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(
                Role.RewardModel
            )
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(
                self.config.trainer.default_hdfs_dir,
                f"global_step_{self.global_steps}",
                "actor",
            )
        )

        remove_previous_ckpt_in_save = self.config.trainer.get(
            "remove_previous_ckpt_in_save", False
        )
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated, set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None)
            if not remove_previous_ckpt_in_save
            else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path,
            actor_remote_path,
            self.global_steps,
            max_ckpt_to_keep=max_actor_ckpt_to_keep,
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir,
                    f"global_step_{self.global_steps}",
                    "critic",
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path,
                critic_remote_path,
                self.global_steps,
                max_ckpt_to_keep=max_critic_ckpt_to_keep,
            )

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = (
                self.config.trainer.default_local_dir
            )  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(
                checkpoint_folder
            )  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(
                    self.config.trainer.resume_from_path, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_from_path
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path,
            del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path,
                del_local_after_load=self.config.trainer.del_local_ckpt_after_load,
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(
                dataloader_local_path, weights_only=False
            )
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(
                f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch"
            )

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = (
            batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()
        )  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor(
            [j for partition in global_partition_lst for j in partition]
        )
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst,
            partitions=global_partition_lst,
            prefix=logging_prefix,
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get(
            "val_before_train", True
        ):
            val_metrics = self._validate()
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="Training Progress",
        )

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                rollout_n_per_prompt = get_rollout_n_per_prompt(
                    batch_dict["rank"],
                    self.rollout_n * len(batch_dict["rank"]),
                    self.rollout_n_min,
                    self.rollout_n_max,
                )

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )
                batch = repeat(
                    batch.batch,
                    batch.non_tensor_batch,
                    batch.meta_info,
                    repeat_times=rollout_n_per_prompt,
                    interleave=True,
                )

                # 将数据按照 rollout_n_per_prompt 复制

                # pop those keys for generation
                if "multi_modal_inputs" in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=[
                            "raw_prompt_ids",
                            "multi_modal_data",
                            "multi_modal_inputs",
                        ],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(
                            gen_batch
                        )

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = (
                                self.actor_rollout_wg.generate_sequences(
                                    gen_baseline_batch
                                )
                            )

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(
                                batch
                            )
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_fn_res = self.reward_fn(batch, return_dict=True)
                        reward_tensor = reward_fn_res["reward_tensor"]
                        reward_extra_info = reward_fn_res["reward_extra_info"]
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available

                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch[
                                "token_level_scores"
                            ]
                        if (
                            self.config.algorithm.penalty_coeff > 0
                            and self.config.algorithm.penalty_threshold > 0
                        ):
                            assert self.config.algorithm.penalty_coeff < 1
                            # penalty all correct rollout where rank < penalty_threshold
                            batch = apply_length_penalty(
                                batch,
                                penalty_coeff=self.config.algorithm.penalty_coeff,
                                penalty_threshold=self.config.algorithm.penalty_threshold,
                            )

                        # compute advantages, executed on the driver process
                        batch, all_error_count = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            length_penalty_coeff=self.config.algorithm.penalty_coeff if self.config.algorithm.penalty_threshold == 0 else 0,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(
                            critic_output.meta_info["metrics"]
                        )
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(
                            actor_output.meta_info["metrics"]
                        )
                        metrics.update(actor_output_metrics)

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (
                            is_last_step
                            or self.global_steps % self.config.trainer.test_freq == 0
                        )
                    ):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(
                    compute_data_metrics(batch=batch, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics(batch=batch, timing_raw=timing_raw)
                )
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(
                    compute_throughout_metrics(
                        batch=batch, timing_raw=timing_raw, n_gpus=n_gpus
                    )
                )

                # TODO: make a canonical logger that supports various backend

                rank = [i for i in batch.non_tensor_batch["rank"]]
                metrics.update({"actor/rank": sum(rank) / len(rank)})
                metrics.update({"actor/all_error_count": all_error_count})
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                index = [i for i in batch.non_tensor_batch["index"]]

                reward = reward_extra_info["acc"]
                self.train_dataset.update_prompt_stats(index, reward)
                self.global_steps += 1

            # update rollout_n_min and rollout_n_max for dynamic rollout_n
            self.rollout_n_min = max(
                self.rollout_n_min - self.rollout_update,
                self.rollout_n_low,
            )
            self.rollout_n_max = min(
                self.rollout_n_max + self.rollout_update,
                self.rollout_n_high,
            )

            self.train_dataset.update_rank()
