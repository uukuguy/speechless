# -*- coding: utf-8 -*-
import os
import json
from typing import Dict, List
from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch
from utils.manual_seed import setup_manual_seed
import ast

class AlgSolution():

    def __init__(self):
        self.model = None
        self.params = None

        # 固定随机种子，防止本地训练结果与云端不一致（选手可自定义）
        seed = 1234
        setup_manual_seed(seed)
        print(f'use random seed: {seed}')

        # 选择可用的 device
        #self.device = torch.device(
            #'cuda:1' if torch.cuda.is_available() else 'cpu')
        #print('use device: ', self.device)

    def load_model(self, model_path:str, params: Dict, **kwargs) -> bool:
        """需要选手加载前一阶段产出的模型参数与特征文件，并初始化打分环境（例如 model.eval() 等）。
            !!! 注意：
            !!!     - 此阶段不可额外读取其他预先准备的数据文件

        Args:
            model_path (str): 本地模型路径
            params (Dict): 模型输入参数。默认为conf/default.json

        Returns:
            bool: True 成功; False 失败
        """
        print("=== model_path", model_path)
        ##### 注意：传入的model_path可能不是你要的
        # 改下面的模型路径，根据你在Dockerfile里模型路径，譬如 /home/admin/atec_project/model_output
        # 或者底座模型 /adabench_mnt/llm/chatglm2-6b
        model_path = '/adabench_mnt/llm/chatglm2-6b'
        print("=== actual model_path", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model = model.half().cuda()
        self.model = model.eval()
        print('model loaded!')
        return True

    def predicts(self, sample_list: List[Dict], **kwargs) -> List[Dict]:
        """需要执行预测打分的流程，选手需将前一阶段加载的特征拼接在测试数据中，再送入模型执行打分。
            !!! 注意：
            !!!     - 此阶段不可额外读取其他预先准备的数据文件

        Args:
            sample_list (List[Dict]): 输入请求内容列表
            kwargs:
                __dataset_root_path (str):  测试集预测时，本地输入数据集路径
                __output_root_path (str):  本地输出路径

        Returns:
            List[Dict]: 输出预测结果列表
        """
        results = []
        temperature = 0.1
        for sample in sample_list:
            input_obj = sample['input']
            input_str = input_obj['prompt'] + '\\n' + input_obj['context'] + '\\n' + input_obj['current_turn']

            # limit the output length to 256
            tokens = self.tokenizer.tokenize(input_str)
            max_len = len(tokens) + 256

            response, history = self.model.chat(self.tokenizer, input_str, history=[], temperature=temperature, max_length=max_len)

            # response should be a string repr of dict
            try:
               response = ast.literal_eval(response)
            except Exception as e:
               response = {"response" : response}

            output_dict = {"id": sample["id"], "input": input_str, "output": response}
            results.append(output_dict)
        return results


if __name__ == '__main__':
    """
        以下代码仅本地测试使用，以流式打分方案提交后云端并不会执行。
    """
    config_path = './conf/default.json'
    params = json.load(open(config_path, 'r'))
    # 改下面的模型路径，根据你在Dockerfile里模型路径，譬如 /home/admin/atec_project/model_output
    # model_path = '/adabench_mnt/llm/chatglm2-6b'
    model_path = '/adabench_mnt/llm/chatglm2-6b'

    # 加载 checkpoint 文件，用于打分预测
    solution = AlgSolution()
    solution.load_model(model_path=model_path, params=params.copy())

    # 执行测试, 下行改成你的测试文件
    input_file = '/home/admin/workspace/job/input/test.jsonl'
    with open(input_file, 'r') as f:
        for line in f:
            test_data = []
            sample = json.loads(line)
            test_data.append(sample)
            result = solution.predicts(sample_list=test_data, params=params.copy())
            result = result[0]
            output_json = json.dumps(result, ensure_ascii=False).encode('utf8')
            print("=== OUTPUT")
            print(output_json.decode())
