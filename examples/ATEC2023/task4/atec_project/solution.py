import os
import json
from typing import Dict, List
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm
import binascii
import scapy.all as scapy

class AlgSolution:

    def __init__(self):

        model_name = "/adabench_mnt/llm/chatglm2-6b"
        ptuning_path = "output/ptuning-model/checkpoint-397"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if ptuning_path is not None:
            config = AutoConfig.from_pretrained(model_name, trust_remote_code=True, pre_seq_len=64)
            self.model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True)
            prefix_state_dict = torch.load(
                os.path.join(ptuning_path, "pytorch_model.bin"), map_location='cpu')
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

        self.MAX_PACKET_NUMBER = 10
        self.MAX_PACKET_LENGTH_IN_FLOW = 256
        self.HEX_PACKET_START_INDEX = 0

    def build_pcap_data(self, pcap_file):
        packets = scapy.rdpcap(pcap_file)
        hex_stream = []
        for packet in packets[:self.MAX_PACKET_NUMBER]:
            packet_data = packet.copy()
            data = (binascii.hexlify(bytes(packet_data)))
            packet_string = data.decode()
            hex_stream.append(packet_string[self.HEX_PACKET_START_INDEX:min(len(packet_string), self.MAX_PACKET_LENGTH_IN_FLOW)])
        flow_data = "<pck>" + "<pck>".join(hex_stream)
        return flow_data

    def pre_process(self, input_data: Dict, dataset_root) -> str:
        prompt = input_data['instruction'] + \
                self.build_pcap_data(dataset_root + '/test/' + input_data['path'])
        return prompt

    def generate(self, prompt: str) -> str:
        response, _ = self.model.chat(self.tokenizer, prompt, history=[])
        return response

    def post_process(self, response: str) -> str:
        return response

    def predicts(self, input_data: List[Dict], **kwargs) -> str:
        dataset_root = kwargs["__dataset_root_path"]
        results = []
        for item in input_data:
            if isinstance(item['id'], str) and item['id'].startswith('subject'):
                result = self.generate(item['input'])
            else:
                prompt = self.pre_process(item, dataset_root)
                response = self.generate(prompt)
                result = self.post_process(response)
            results.append({
              'id': item['id'],
              'output': result,
            }) 
        return results
