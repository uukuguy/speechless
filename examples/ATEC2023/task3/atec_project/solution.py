import os
import json
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


class AlgSolution:

    def __init__(self):

        model_name = "/adabench_mnt/llm/chatglm2-6b"
        # 0.917505
        ptuning_path = "output/ptuing-chatglm2-6b-pt-128-2e-2/checkpoint-750/"
        # 0.917323
        ptuning_path = "output/ptuing-chatglm2-6b-pt-128-2e-2/checkpoint-500/"
        # 0.849785
        ptuning_path = "output/ptuing-chatglm2-6b-pt-128-2e-2/checkpoint-250/"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if ptuning_path is not None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=128)
            self.model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
            prefix_state_dict = torch.load(os.path.join(ptuning_path, "pytorch_model.bin"), map_location='cpu')
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
            self.model = self.model.half().cuda()
            self.model.transformer.prefix_encoder.float()
        else:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).half().cuda()
        self.model.eval()

    def pre_process(self, input_data: Dict) -> str:
        prompt = "请你帮我判断如下内容是人类撰写的还是机器生成的，内容如下《%s》，注意回复为该新闻是由人类撰写的或该新闻是由机器合成的。" % input_data['input']
        return prompt

    def generate(self, prompt: str) -> str:
        response, _ = self.model.chat(self.tokenizer, prompt, history=[])
        return response

    def post_process(self, response: str) -> str:
        return response

    def predicts(self, input_data: List[Dict], **kwargs) -> str:
        results = []
        for item in input_data:
            if isinstance(item['id'], str) and item['id'].startswith('subject'):
                result = self.generate(item['input'])
            else:
                prompt = self.pre_process(item)
                response = self.generate(prompt)
                result = self.post_process(response)
            results.append({
                'id': item['id'],
                'output': result
            })
        return results
