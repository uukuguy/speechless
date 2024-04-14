# https://github.com/wangitu/Ada-Instruct/blob/main/synthesizer.py
import os, sys, json
from tqdm import tqdm
from accelerate import Accelerator

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import set_seed, DefaultDataCollator

from .config import get_generation_config
from .tasks import get_task


# from process_helper import ProcessHelperMixin
class ProcessHelperMixin:
    def __init__(self, accelerator: Accelerator):
        self.accelerator = accelerator
        
    @property
    def num_processes(self):
        return self.accelerator.num_processes

    @property
    def process_index(self):
        return self.accelerator.process_index

    @property
    def local_process_index(self):
        return self.accelerator.local_process_index

    @property
    def device(self):
        return self.accelerator.device
    
    @property
    def is_main_process(self):
        """True for one process only."""
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self):
        """True for one process per server."""
        return self.accelerator.is_local_main_process
    
    def pad_across_processes(self, tensor, dim=0, pad_index=0, pad_first=False):
        return self.accelerator.pad_across_processes(tensor, dim, pad_index, pad_first)
    
    def gather(self, tensor):
        return self.accelerator.gather(tensor)



class Synthesizer(ProcessHelperMixin):
    def __init__(self, args, accelerator, model, tokenizer):
        ProcessHelperMixin.__init__(self, accelerator)
        
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        
    def _parallel_generation(self, inputs):
        set_seed(42 + self.process_index)
        
        synthesize_num = self.args.synthesize_num
        
        class AllInOneDataset(Dataset):
            def __getitem__(self, index):
                return inputs
                
            def __len__(self):
                return synthesize_num
            
        
        dataloader = DataLoader(AllInOneDataset(), batch_size=self.args.batch_size, collate_fn=DefaultDataCollator())
        # we only wrap data loader to avoid extra memory occupation
        self.model.to(self.device)
        dataloader = self.accelerator.prepare(dataloader)
        
        generation_config = get_generation_config(self.args.task_name)
        
        output_tokens = []
        for batch in tqdm(dataloader, desc='synthesizing instructions...', disable=not self.is_local_main_process):
            # we could avoid `batch.to(self.device)` since we set the accelerator with `device_placement=True`
            output = self.model.generate(
                **batch,
                generation_config=generation_config
            )
            output_ids = output if isinstance(output, torch.Tensor) else output.sequences
            
            # pad across processes before gather
            output_ids = self.pad_across_processes(
                output_ids, dim=1, pad_index=self.tokenizer.pad_token_id
            )
            # gather across processes and offload to cpu
            output_ids = self.gather(output_ids).cpu()
            output_tokens.extend(output_ids)
            
        return output_tokens
        
    def synthesize(self, task_args=None):
        task_name, args = (task_args.task_name, task_args) if task_args is not None else (self.args.task_name, self.args)
        task = get_task(task_name, args)
        prompt = task.get_prompt()
        inputs = self.tokenizer(prompt, add_special_tokens=False)
        
        output_tokens = self._parallel_generation(inputs)
        outputs = []
        for i, output in enumerate(self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)):
            outputs.append({
                'id': i,
                'instruction': task.get_response(output)
            })
            
        if self.is_local_main_process:
            # if `self.model = None` or `del self.model`, there is still reference outside. Loading model in `__init__` could work
            self.model.to('cpu')
            torch.cuda.empty_cache()
                        
            retained, discarded = task.postprocess_synthesized_instructions(outputs)
            
            print(f"{len(retained)} retained, {len(discarded)} discarded")
            
            out_dir, out_file = os.path.split(self.args.out_file)
            out_file_name, ext = os.path.splitext(out_file)
            os.makedirs(out_dir, exist_ok=True)
            
            with open(self.args.out_file, 'w', encoding='utf-8') as f:
                json.dump(retained, f, ensure_ascii=False, indent=2)
                
            with open(os.path.join(out_dir, out_file_name + '_discarded' + ext), 'w', encoding='utf-8') as f:
                json.dump(discarded, f, ensure_ascii=False, indent=2)
    

from accelerate import Accelerator

# from utils import load_model_tokenizer_to_device
def mpExceptionHandler(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except BaseException as e:
            print(f"Process {multiprocessing.current_process().pid} is raising: {e}")
            raise e

    return decorated


@mpExceptionHandler
def load_model_tokenizer_to_device(args, i):

    from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)  
    
    tokenizer.padding_side = 'left'
    
    # If you wish faster inference, set `config.pretraining_tp` to 1, but at the cost of higher GPU memory usage
    # Reference: https://huggingface.co/docs/transformers/v4.32.0/en/model_doc/llama2
    config = LlamaConfig.from_pretrained(args.base_model)
    config.pretraining_tp = args.pretraining_tp
    config.use_cache = True
    
    model = LlamaForCausalLM.from_pretrained(args.base_model, config=config, torch_dtype=torch.float16, device_map={'': i})
    
    if model.generation_config.pad_token_id is None or model.config.pad_token_id is None:
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.pad_token_id
        elif tokenizer.eos_token_id is not None:
            model.generation_config.pad_token_id = model.config.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer

def synthesize_instructions():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--task_name', type=str, choices=['humaneval', 'mbpp', 'gsm8k', 'math', 'csqa'], required=True)
    parser.add_argument('--synthesize_num', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    parser.add_argument('--pretraining_tp', type=int, default=1)
    args = parser.parse_args()

    accelerator = Accelerator()
    model, tokenizer = load_model_tokenizer_to_device(args, accelerator.device)
    synthesizer = Synthesizer(args, accelerator, model, tokenizer)
    synthesizer.synthesize()
    

if __name__ == '__main__':
    synthesize_instructions()