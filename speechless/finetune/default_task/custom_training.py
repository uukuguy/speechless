
from typing import List
import torch
from transformers import Seq2SeqTrainer, Trainer, TrainingArguments
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
import toml

# 自定义损失函数
class TaskLoss(torch.nn.Module):
    def __init__(self, tokenizer):
        super(TaskLoss, self).__init__()
        self.tokenizer = tokenizer

    def forward(self, model_output, labels):
        # 解码生成的文本
        generated_texts = self.tokenizer.batch_decode(
            torch.argmax(model_output.logits, dim=-1), 
            skip_special_tokens=True
        )
        # print(f"{generated_texts=}")
        
        # # 解码目标文本
        # target_text = self.tokenizer.decode(labels[0], skip_special_tokens=True)

        # 自定义损失计算（例如：文本相似度）
        # 这里只是一个示例，你可以替换为实际的损失计算逻辑
        loss = torch.tensor(0.0, requires_grad=True)  # 占位符
        for generated_text in generated_texts:
            task_loss = 0.0
            if "<tool_call>" in generated_text:
                toolcall_text = generated_text.split("<tool_call>")[1].split("</tool_call>")[0]
                try:
                    toolcall_text = toml.loads(toolcall_text)
                except Exception as e:
                    task_loss = 1.0
                    
            if task_loss > 0:
                loss = loss + task_loss
        
        return loss


class TaskTrainer(Seq2SeqTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_loss = TaskLoss(self.tokenizer)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        original_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        # 计算自定义损失
        _task_loss = self.task_loss(outputs, labels)
        if _task_loss > 0:
            if original_loss < 1.0:
                loss = original_loss * 0.5 + _task_loss
            else:
                loss = original_loss * 2
        else:
            loss = original_loss
        
        return (loss, outputs) if return_outputs else loss