import numpy as np
import torch
from torch import Tensor

def multilabel_categorical_crossentropy(y_pred, y_true, reduction='none'):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
    1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
    不用加激活函数，尤其是不能加sigmoid或者softmax！预测
    阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
    本文。
    """
    y_pred = (1 - 2 * y_true) * y_pred # 将正例乘以-1，负例乘以1
    y_pred_neg = y_pred - y_true * 1e12 # 将正例变为负无穷，消除影响
    y_pred_pos = y_pred - (1 - y_true) * 1e12 # 将负例变为负无穷
    zeros = torch.zeros_like(y_pred[..., :1]) 
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1) # 0阈值
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    loss = neg_loss + pos_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))

def batch_gather(input: Tensor, indices: Tensor):
    """
    Args:
        input: label tensor with shape [batch_size, n, L] or [batch_size, L]
        indices: predict tensor with shape [batch_size, m, l] or [batch_size, l]

    Return:
        Note that when second dimention n != m, there will be a reshape operation to gather all value along this dimention of input 
        if m == n, the return shape is [batch_size, m, l]
        if m != n, the return shape is [batch_size, n, l*m]

    """
    if indices.dtype != torch.int64:
        indices = indices.type(torch.int64)
    results = []
    # breakpoint()
    for data, index in zip(input, indices):
        if len(index) < len(data):
            index = index.reshape(-1)
            results.append(data[..., index])
        else:
            indice_dim = index.ndim
            results.append(torch.gather(data, dim=indice_dim-1, index=index))
    return torch.stack(results)


def sparse_multilabel_categorical_crossentropy(pred: Tensor, label: Tensor, mask_zero=False, reduction='none'):
    """Sparse Multilabel Categorical CrossEntropy
        Reference: https://kexue.fm/archives/8888, https://github.com/bojone/bert4keras/blob/4dcda150b54ded71420c44d25ff282ed30f3ea42/bert4keras/backend.py#L272
        You should remove `[CLS]` token before call this function. 

    Args:
        pred: logits tensor with shape [batch_size, m, num_classes] or [batch_size, num_classes], don't use acivation.
        label: label tensor with shape [batch_size, n, num_positive] or [Batch_size, num_positive]
            should contain the indexes of the positive rather than a ont-hot vector.
        mask_zero: if label is used zero padding to align, please specify make_zero=True.
            when mask_zero = True, make sure the label start with 1 to num_classes, before zero padding.

    """
    zeros = torch.zeros_like(pred[..., :1])
    pred = torch.cat([pred, zeros], dim=-1)
    if mask_zero:
        infs = torch.ones_like(zeros) * np.inf
        pred = torch.cat([infs, pred], dim=-1)
    pos_2 = batch_gather(pred, label)
    pos_1 = torch.cat([pos_2, zeros], dim=-1)
    if mask_zero:
        pred = torch.cat([-infs, pred[..., 1:]], dim=-1)
        pos_2 = batch_gather(pred, label)
    pos_loss = torch.logsumexp(-pos_1, dim=-1)
    all_loss = torch.logsumexp(pred, dim=-1)
    aux_loss = torch.logsumexp(pos_2, dim=-1) - all_loss
    aux_loss = torch.clip(1 - torch.exp(aux_loss), 1e-16, 1)
    neg_loss = all_loss + torch.log(aux_loss)
    loss = pos_loss + neg_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise Exception('Unexpected reduction {}'.format(reduction))


if __name__ == '__main__':
    x = torch.tensor(np.arange(384).reshape(2, 3, 64))
    y = torch.tensor(np.arange(1024).reshape(2, 8, 64))
    indices = torch.tensor(
        [
            [[1, 2, 3, 4], [0, 1, 0, 0], [0, 0, 0, 0]],
            [[0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        ], dtype=torch.int64)
    print(indices.shape)
    print('='*80)
    res = batch_gather(x, indices)
    print(res.shape)
    print(res)
    print('='*80)
    res = batch_gather(y, indices)
    print(res.shape)
    print(res)


import math
import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from packaging import version
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import BertForSequenceClassification 
from transformers.modeling_outputs import SequenceClassifierOutput

class BertForSequenceMultiLabelsClassification(BertForSequenceClassification):
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # loss_fct = BCEWithLogitsLoss()
                # loss = loss_fct(logits, labels)
                # print(f"BCE loss: {loss}")
                # loss_fct = sparse_multilabel_categorical_crossentropy
                # loss = loss_fct(logits, labels, reduction="mean")
                # loss = loss.detach().cpu().numpy()[0]
                # print(f"sparse loss: {loss}")
                loss_fct = multilabel_categorical_crossentropy
                loss = loss_fct(logits, labels, reduction="mean")
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
