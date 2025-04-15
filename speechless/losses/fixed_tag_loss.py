
from typing import List
import torch
from loguru import logger

class FixedTagLoss(torch.nn.Module):
    def __init__(self, tokenizer, fixed_tags: List, fixed_tag_weight: float = 2.0, allowed_colors: List = None, allowed_token_weight: float = 1.0):
        super(FixedTagLoss, self).__init__()
        self.tokenizer = tokenizer
        assert isinstance(fixed_tags, list) and len(fixed_tags) > 0, "fixed_tags should be a non-empty list"
        self.fixed_tags = fixed_tags
        self.fixed_tags_ids = None
        self.fixed_tag_weight = fixed_tag_weight
        if allowed_colors is None:
            allowed_colors = ('Black', 'Blue', 'Red', 'Green', 'Yellow', 'Gray', 'Pink', 'Orange', 'Purple', 'Brown')
        self.allowed_colors = allowed_colors
        self.allowed_token_weight = allowed_token_weight

        self.allowed_token_ids = tokenizer.convert_tokens_to_ids(allowed_colors) if self.allowed_colors else []
        self.allowed_token_ids = [torch.tensor(token_id).to(tokenizer.device) for token_id in self.allowed_token_ids]

    # def forward(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    def forward(self, outputs, labels, num_items_in_batch=None):
        if self.fixed_tags_ids is None:
            self.fixed_tags_ids = [self.tokenizer.encode(tag, add_special_tokens=False) for tag in self.fixed_tags]
            self.fixed_tags_ids = [torch.tensor(tag_id).to(labels.device) for tag_id in self.fixed_tags_ids]

        logits = outputs.logits
        weights = torch.ones_like(labels, dtype=torch.float)
        # weights = torch.tensor([1.0] * logits.size(-1), dtype=torch.float).to(labels.device)

        fixed_tag_weight = self.fixed_tag_weight
        allowed_token_weight = self.allowed_token_weight
        # fixed_tag_weight = 2.0
        # allowed_token_weight = 1.0

        # fixed_tag_weight = 10.0
        # allowed_token_weight = 2.0

        def set_weights(tag_id, weight):
            tag_len = len(tag_id)
            for i in range(labels.size(0)):
                for j in range(labels.size(1) - tag_len + 1):
                    if torch.equal(labels[i, j:j + tag_len], tag_id):
                        weights[i, j:j + tag_len] = weight

        if fixed_tag_weight > 1.0:
            for fixed_tag_id in self.fixed_tags_ids:
                set_weights(fixed_tag_id, fixed_tag_weight)
                # fixed_tag_len = len(fixed_tag_id)
                # for i in range(labels.size(0)):
                #     for j in range(labels.size(1) - fixed_tag_len + 1):
                #         if torch.equal(labels[i, j:j + fixed_tag_len], fixed_tag_id):
                #             weights[i, j:j + fixed_tag_len] = fixed_tag_weight

        if self.allowed_token_ids and allowed_token_weight > 1.0:
            for allowed_token_id in self.allowed_token_ids:
                set_weights(allowed_token_id, allowed_token_weight)
                # for i in range(labels.size(0)):
                #     for j in range(labels.size(1)):
                #         if labels[i, j] == allowed_token_id:
                #             weights[i, j] = allowed_token_weight
        

        # logger.debug(f"{logits.shape=}, {labels.shape=}, {weights.shape=}")

        # transformers/loss/loss_utils.py ForCausalLMLoss

        from transformers.loss.loss_utils import ForCausalLMLoss
        loss = ForCausalLMLoss(logits, labels, vocab_size=logits.size(-1), num_items_in_batch=num_items_in_batch)
        weighted_loss = loss * weights.view(-1)
        loss = weighted_loss.mean()

        # # loss_fct = torch.nn.CrossEntropyLoss(weight=weights.view(-1), reduction="mean")
        # loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        # # Flatten the tokens
        # logits = logits.view(-1, logits.size(-1))
        # labels = labels.view(-1)
        # # Enable model parallelism
        # # labels = labels.to(logits.device)
        # loss = loss_fct(logits, labels)
        # # logger.debug(f"{loss.shape=}")
        # weighted_loss = loss * weights.view(-1)
        # # logger.debug(f"{weighted_loss.shape=}")
        # loss = weighted_loss.mean()

        return loss
