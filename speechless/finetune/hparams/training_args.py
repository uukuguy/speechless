from dataclasses import dataclass, field
from transformers import HfArgumentParser, Seq2SeqTrainingArguments

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    num_early_stopping_train_epochs: int = field(default=0, metadata={"help": 'Number of training epochs before early stopping.'})

    def __post_init__(self):
        super().__post_init__()
        if self.num_early_stopping_train_epochs <= 0:
            self.num_early_stopping_train_epochs = self.num_train_epochs
