from mlx_lm.tuner.trainer import TrainingCallback
try:
    import wandb
except ImportError:
    wandb = None


class BaseTrainingCallback(TrainingCallback):
    def __init__(self, progress_bar):
        self.iteration = 0
        self.progress_bar = progress_bar

    def update_progress(self, info):
        iteration = info["iteration"]
        self.progress_bar.update(iteration - self.iteration)
        self.iteration = iteration

    def on_train_loss_report(self, train_info):
        self.update_progress(train_info)

    def on_val_loss_report(self, val_info):
        self.update_progress(val_info)


class WandbCallback(BaseTrainingCallback):

    def on_train_loss_report(self, train_info):
        super().on_train_loss_report(train_info)
        if wandb is None:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        try:
            wandb.log(train_info, step=train_info["iteration"])
        except Exception as e:
            print(f"logging to wandb failed: {e}")

    def on_val_loss_report(self, val_info):
        super().on_train_loss_report(val_info)
        if wandb is None:
            raise ImportError('wandb module not available.  Install with `pip install wandb`')
        try:
            wandb.log(val_info, step=val_info["iteration"])
        except Exception as e:
            print(f"logging to wandb failed: {e}")