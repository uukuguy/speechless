import os, sys, json, time
from datetime import timedelta

import torch

from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, has_length

# from .constants import LOG_FILE_NAME

LOG_FILE_NAME = "trainer_log.jsonl"

from transformers import TrainerControl, TrainerState, TrainingArguments


# from .logging import get_logger
# logger = get_logger(__name__)
from loguru import logger


# from .misc import fix_valuehead_checkpoint
# class FixValueHeadModelCallback(TrainerCallback):
#     def on_save(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
#         r"""
#         Event called after a checkpoint save.
#         """
#         if args.should_save:
#             fix_valuehead_checkpoint(
#                 model=kwargs.pop("model"),
#                 output_dir=os.path.join(args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, state.global_step)),
#                 safe_serialization=args.save_safetensors,
#             )


class LoggingCallback(TrainerCallback):
    def __init__(self, runner=None):
        self.runner = runner
        self.in_training = False
        self.start_time = time.time()
        self.cur_steps = 0
        self.max_steps = 0
        self.elapsed_time = ""
        self.remaining_time = ""

    def timing(self):
        cur_time = time.time()
        elapsed_time = cur_time - self.start_time
        avg_time_per_step = elapsed_time / self.cur_steps if self.cur_steps != 0 else 0
        remaining_time = (self.max_steps - self.cur_steps) * avg_time_per_step
        self.elapsed_time = str(timedelta(seconds=int(elapsed_time)))
        self.remaining_time = str(timedelta(seconds=int(remaining_time)))

    def on_train_begin(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the beginning of training.
        """
        if state.is_local_process_zero:
            self.in_training = True
            self.start_time = time.time()
            self.max_steps = state.max_steps

        if args.save_on_each_node:
            if not state.is_local_process_zero:
                return
        else:
            if not state.is_world_process_zero:
                return

        if os.path.exists(os.path.join(args.output_dir, LOG_FILE_NAME)) and args.overwrite_output_dir:
            logger.warning("Previous log file in this folder will be deleted.")
            os.remove(os.path.join(args.output_dir, LOG_FILE_NAME))

    def on_train_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of training.
        """
        if state.is_local_process_zero:
            self.in_training = False
            self.cur_steps = 0
            self.max_steps = 0

    def on_substep_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of an substep during gradient accumulation.
        """
        if state.is_local_process_zero and self.runner is not None and self.runner.aborted:
            control.should_epoch_stop = True
            control.should_training_stop = True

    def on_step_end(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called at the end of a training step.
        """
        if state.is_local_process_zero:
            self.cur_steps = state.global_step
            self.timing()
            if self.runner is not None and self.runner.aborted:
                control.should_epoch_stop = True
                control.should_training_stop = True

    def on_evaluate(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs):
        r"""
        Event called after an evaluation phase.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_predict(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", *other, **kwargs
    ):
        r"""
        Event called after a successful prediction.
        """
        if state.is_local_process_zero and not self.in_training:
            self.cur_steps = 0
            self.max_steps = 0

    def on_log(self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs) -> None:
        r"""
        Event called after logging the last logs.
        """
        if args.save_on_each_node:
            if not state.is_local_process_zero:
                return
        else:
            if not state.is_world_process_zero:
                return

        logs = dict(
            current_steps=self.cur_steps,
            total_steps=self.max_steps,
            loss=state.log_history[-1].get("loss", None),
            eval_loss=state.log_history[-1].get("eval_loss", None),
            predict_loss=state.log_history[-1].get("predict_loss", None),
            reward=state.log_history[-1].get("reward", None),
            accuracy=state.log_history[-1].get("rewards/accuracies", None),
            learning_rate=state.log_history[-1].get("learning_rate", None),
            epoch=state.log_history[-1].get("epoch", None),
            percentage=round(self.cur_steps / self.max_steps * 100, 2) if self.max_steps != 0 else 100,
            elapsed_time=self.elapsed_time,
            remaining_time=self.remaining_time,
        )
        if self.runner is not None:
            logger.info(
                "{{'loss': {:.4f}, 'learning_rate': {:2.4e}, 'epoch': {:.2f}}}".format(
                    logs["loss"] or 0, logs["learning_rate"] or 0, logs["epoch"] or 0
                )
            )

        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "trainer_log.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(logs) + "\n")

    def on_prediction_step(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ):
        r"""
        Event called after a prediction step.
        """
        eval_dataloader = kwargs.pop("eval_dataloader", None)
        if state.is_local_process_zero and has_length(eval_dataloader) and not self.in_training:
            if self.max_steps == 0:
                self.max_steps = len(eval_dataloader)
            self.cur_steps += 1
            self.timing()


import gc, ctypes
def clean_memory():
    for _ in range(3):
        gc.collect()
        if sys.platform == 'linux':
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        # mps backend
        if torch.backends.mps.is_available():
            torch.cuda.empty_cache()

class CleanMemoryCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        clean_memory()
        if state.is_local_process_zero:
            print("Clean GPU memory at step:", state.global_step) # Optional: for monitoring

    def on_evaluate(self, args, state, control, **kwargs):
        clean_memory()
        if state.is_local_process_zero:
            print("Clean GPU memory in evaluating stage") # Optional: for monitoring

# from transformers.trainer import ExportableState
class EarlyStoppingCallback(TrainerCallback):#, ExportableState):

    def __init__(self, early_stopping_train_epochs: int = 0):
        self.early_stopping_train_epochs = early_stopping_train_epochs

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self.early_stopping_train_epochs > 0:
            if state.epoch >= self.early_stopping_train_epochs:
                control.should_training_stop = True

    # def state(self) -> dict:
    #     return {
    #         "args": {
    #             "early_stopping_train_epochs": self.early_stopping_train_epochs,
    #         },
    #         "attributes": {
    #         }
    #     }


class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        logger.info('Saving PEFT checkpoint...')
        checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

        self._symlink_latest_checkpoint(checkpoint_folder)

    def _symlink_latest_checkpoint(self, checkpoint_folder):
        # if the latest checkpoint is a symlink, remove it
        output_dir = os.path.dirname(checkpoint_folder)
        latest_checkpoint = os.path.join(output_dir, "latest")
        if os.path.islink(latest_checkpoint):
            os.remove(latest_checkpoint)
        # symlink the latest checkpoint to the checkpoint folder
        os.symlink(os.path.basename(checkpoint_folder), latest_checkpoint)

    def on_save(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        if state.is_local_process_zero:
            touch(os.path.join(args.output_dir, 'completed'))
            self.save_model(args, state, kwargs)