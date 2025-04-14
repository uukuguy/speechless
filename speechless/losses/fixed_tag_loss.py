
from typing import List
import torch
from loguru import logger

class FixedTagLoss(torch.nn.Module):
    def __init__(self, tokenizer, fixed_tags: List):
        super(FixedTagLoss, self).__init__()
        self.tokenizer = tokenizer
        assert isinstance(fixed_tags, list) and len(fixed_tags) > 0, "fixed_tags should be a non-empty list"
        self.fixed_tags = fixed_tags
        self.fixed_tags_ids = None

    # def forward(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    def forward(self, outputs, labels, num_items_in_batch=None):
        if self.fixed_tags_ids is None:
            self.fixed_tags_ids = [self.tokenizer.encode(tag, add_special_tokens=False) for tag in self.fixed_tags]
            self.fixed_tags_ids = [torch.tensor(tag_id).to(labels.device) for tag_id in self.fixed_tags_ids]

        logits = outputs.logits
        # weights = torch.ones_like(labels, dtype=torch.float)
        weights = torch.tensor([1.0] * logits.size(-1), dtype=torch.float).to(labels.device)
        for fixed_tag_id in self.fixed_tags_ids:
            fixed_tag_len = len(fixed_tag_id)
            for i in range(labels.size(0)):
                for j in range(labels.size(1) - fixed_tag_len + 1):
                    if torch.equal(labels[i, j:j + fixed_tag_len], fixed_tag_id):
                        weights[i, j:j + fixed_tag_len] = 2.0

        # logger.debug(f"{logits.shape=}, {labels.shape=}, {weights.shape=}")

        # transformers/loss/loss_utils.py ForCausalLMLoss

        loss_fct = torch.nn.CrossEntropyLoss(weight=weights.view(-1), reduction="mean")
        # Flatten the tokens
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)
        # Enable model parallelism
        # labels = labels.to(logits.device)
        loss = loss_fct(logits, labels)

        return loss
