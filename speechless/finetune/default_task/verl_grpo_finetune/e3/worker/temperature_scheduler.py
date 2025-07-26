from typing import Optional, Union, List
import math
import numpy as np


class TemperatureScheduler:
    def __init__(
        self,
        init_temperature: float,
        vocab_size: Optional[int] = None,
        enable_temperature_scheduler: bool = False,
        init_steps: int = 1,
        target_entropy: Optional[float] = None,
        enable_annealing: bool = False,
        max_steps: int = -1,
        annealing_ratio: float = 0.4,
        annealing_factor: float = 0.9,
    ):
        self.init_temperature = init_temperature
        self.init_steps = init_steps
        self.current_temperature = init_temperature
        self.steps = 1
        self.init_entropy = []
        self.target_entropy = target_entropy
        self.vocab_size = vocab_size
        self.factor = None
        if vocab_size is not None:
            self.set_vocab_size(vocab_size)
        self.enable_temperature_scheduler = enable_temperature_scheduler
        # annealing
        self.enable_annealing = enable_annealing
        self.max_steps = max_steps
        self.annealing_step = max_steps - int(max_steps * annealing_ratio)
        self.annealing_factor = annealing_factor
        self.initial_target_entropy = target_entropy

    def set_vocab_size(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.factor = np.log(self.vocab_size) + np.log(np.log(self.vocab_size))

    def step(self, entropy: Union[float, List[float]]):
        if self.steps <= self.init_steps:
            if isinstance(entropy, float):
                self.init_entropy.append(entropy)
            else:
                self.init_entropy.extend(entropy)
        elif self.enable_temperature_scheduler:
            if self.target_entropy is None:
                self.target_entropy = sum(self.init_entropy) / len(self.init_entropy)
                self.initial_target_entropy = self.target_entropy
            if isinstance(entropy, float):
                current_entropy = entropy
            elif isinstance(entropy, list):
                current_entropy = sum(entropy) / len(entropy)
            else:
                raise ValueError(f"Invalid entropy type: {type(entropy)}")
            alpha = self.target_entropy / current_entropy
            self.current_temperature = self.current_temperature * (
                1 + self.current_temperature * np.log(alpha) / self.factor
            )
            if self.enable_annealing and self.steps >= self.annealing_step:
                self.annealing()
        self.steps += 1

    def get_temperature(self):
        return self.current_temperature

    def annealing(self):
        progress = (self.steps - self.annealing_step) / (
            self.max_steps - self.annealing_step
        )
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        decay_factor = (
            1 - self.annealing_factor
        ) * cosine_decay + self.annealing_factor
        self.target_entropy = self.initial_target_entropy * decay_factor
