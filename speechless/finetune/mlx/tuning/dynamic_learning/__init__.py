"""
Framework for learning rate schedulers
Many taken from #https://mxnet.apache.org/versions/1.7/api/python/docs/tutorials/packages/gluon/training/learning_rates/learning_rate_schedules_advanced.html

[1] https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20
"""
from abc import ABC
import mlx.optimizers.schedulers as mlx_schedulers


class DynamicLearningRateSchedule(ABC):

    def __init__(self, learning_rate: float, total_iterations: int):
        self.learning_rate = learning_rate
        self.total_iterations = total_iterations

    def update(self, iteration: int) -> float:
        """
        Called before commencing with each iteration to provide the learning rate scheduler the chance
        to update the rate relative to iterations over time.

        Returns the (new or same) learning rate
        """
        pass


class ConstantLearningRateSchedule(DynamicLearningRateSchedule):
    """
    The default Learning Rate Manager, which does not make any changes to the learning rate
    """
    def update(self, iteration: int) -> float:
        return self.learning_rate

    @classmethod
    def from_configuration(cls, learning_rate, config, total_iterations):
        return learning_rate

    def __str__(self):
        return f"ConstantLearningRateSchedule: {self.learning_rate})"


class CosineWithWarmup:
    @classmethod
    def from_configuration(cls, learning_rate, config, total_iterations):
        param_dict = {k: v for k, v in config["learning_schedule"].items()}
        min_lr = param_dict["min_lr"]
        min_cos_lr = param_dict["min_cos_lr"] if "min_cos_lr" in param_dict else 0.0
        max_lr = param_dict["max_lr"] if "max_lr" in param_dict else learning_rate
        cycle_length = param_dict["cycle_length"]
        cycle_length = total_iterations if cycle_length == -1 else cycle_length
        length = param_dict["length"] if "length" in param_dict else int(param_dict["warmup_proportion"] *
                                                                         total_iterations)
        warmup_schedule = mlx_schedulers.linear_schedule(min_lr, max_lr, length)
        cosine_schedule = mlx_schedulers.cosine_decay(max_lr, cycle_length, min_cos_lr)
        cosine_w_warmup_schedule = mlx_schedulers.join_schedules([warmup_schedule, cosine_schedule], [length])
        return cosine_w_warmup_schedule


class Cosine:
    @classmethod
    def from_configuration(cls, learning_rate, config, total_iterations):
        param_dict = {k: v for k, v in config["learning_schedule"].items()}
        max_lr = param_dict["max_lr"] if "max_lr" in param_dict else learning_rate
        cycle_length = param_dict["cycle_length"]
        cycle_length = total_iterations if cycle_length == -1 else cycle_length
        return mlx_schedulers.cosine_decay(max_lr, cycle_length)


SCHEDULE_CONFIGURATION_TYPE_TO_CLASS = {
    "cosine": Cosine,
    "cosine_w_warmup": CosineWithWarmup,
    "constant": ConstantLearningRateSchedule
}
