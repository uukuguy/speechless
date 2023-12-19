from ostest import make_os_line
from transformers import AutoConfig, AutoTokenizer, AutoModel
import torch
from typing import Dict, List
import os

class AlgSolution():


    def __init__(self):
        self.base_model_dir = "./chatglm2"
        self.model = None
        self.tokenizer = None

    def train_model(self, input_data_path: str, output_model_path: str, params: Dict, **kwargs) -> bool:
        """使用数据集训练模型
        Args:
        input_data_path (str): 本地输入数据集路径
        output_model_path (str): 本地输出模型路径
        params (Dict): 训练输入参数。默认为conf/default.json
        Returns:
        bool: True 成功; False 失败
        """
        # load pretrained model if any
        # self.model = load_from_pretrained()
        train_data_path = os.path.join(input_data_path, 'train.jsonl')

        # 构建特征
        get_data_feature(train_data_path)

        # 执行文件

        make_os_line(self.base_model_dir, output_model_path)
        # train model
        # self.model.train(train_samples)
        # save model
        # self.model.save(output_model_path)
        return True
    def load_model(self, model_path: str, params: Dict, **kwargs) -> bool:

            tokenizer = AutoTokenizer.from_pretrained(self.base_model_dir, trust_remote_code=True)
            config = AutoConfig.from_pretrained(self.base_model_dir, trust_remote_code=True, pre_seq_len=128)
            model = AutoModel.from_pretrained(self.base_model_dir, config=config, trust_remote_code=True)
            # /mnt/data/xiaoyong/ATEC/run/ChatGLM2-6B/output/subcot-chatglm2-6b-pt-128-2e-2/checkpoint-3000
            CHECKPOINT_PATH = model_path + '/checkpoint-10/'
            prefix_state_dict = torch.load(os.path.join(CHECKPOINT_PATH, "pytorch_model.bin"))
            new_prefix_state_dict = {}
            for k, v in prefix_state_dict.items():
                if k.startswith("transformer.prefix_encoder."):
                    new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
            model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
            # model = model.quantize(4)
            model = model.cuda()
            model = model.eval()

            self.tokenizer = tokenizer
            self.model = model

            """从本地加载模型
            Args:
            model_path (str): 本地模型路径
            params (Dict): 模型输入参数。默认为conf/default.json
            Returns:
            bool: True 成功; False 失败
            """
            # self.model = load_from_trained_model()
            return True
    def predicts(self, sample_list: List[Dict], **kwargs) -> List[Dict]:
            """"
            批量预测
            Args:
            sample_list(List[Dict]): 输入请求内容列表
            kwargs:
            __dataset_root_path(str): 测试集预测时，本地输入数据集路径
            Returns:
            List[Dict]: 输出预测结果列表
            """
            dataset_root_path = kwargs.get('__dataset_root_path') # noqa
            results = []
            for sample in sample_list:
                # add_feature(sample['input'])
                response, history = self.model.chat(self.tokenizer, sample['input'], history=[])
                sample['output'] = response
            # label = self.model.predict([sample['input_text']])[0]
            # sample['label'] = label
                results.append(sample)
            return results




if __name__ == "__main__":

    创建AlgSolution实例
    alg_solution = AlgSolution()

    假设训练数据和模型存储的路径
    input_data_path = "./data"
    output_model_path = "output/test/"
    # train_params = {"param1": "value1", "param2": "value2"} # 根据需要设置训练参数

    调用train_model函数
    alg_solution.train_model(input_data_path, output_model_path, train_params)

    # 加载模型，这里假设模型已经训练并保存在指定的路径
    # model_path = "output/test/"
    # load_params = {"param1": "value1", "param2": "value2"} # 根据需要设置加载参数

    # alg_solution.load_model(output_model_path, load_params)

    # 进行预测，这里假设有一个模拟的样本列表
    # sample_list = [{"input": "sample1"}, {"input": "sample2"}] # 根据实际情况构造样本
    # predict_params = {"__dataset_root_path": "path/to/dataset"} # 根据需要设置预测参数

    # predictions = alg_solution.predicts(sample_list, **predict_params)

    # 打印预测结果
    # print(predictions)
