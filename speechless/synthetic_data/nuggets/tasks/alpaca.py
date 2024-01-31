from .base import BaseProbInference
import logging
import json
from .loader import TokenizedForGenRightPad
import numpy as np
import os
from pathlib import Path

logger = logging.getLogger("task")

class AlpacaProbInference(BaseProbInference):
    def __init__(self, prompt_version, prompt_path=None, test_path=None):
        super().__init__(prompt_version)

        self.can_be_stratified = True
        self.num_base_shot = 1
        self.num_eval = 1
        self.prompt_path=prompt_path
        self.test_path=test_path

    def default_prompt_version(self):
        return "sp"

    def do_load(self):
        assert self.prompt_path is not None
        assert self.test_path is not None
        if os.path.exists(self.prompt_path): #PROMPT
            f_path, file_name = os.path.split(self.prompt_path)
            self.raw_data_sample = self.do_load_json(Path(f_path), file_name)
        if os.path.exists(self.test_path): # TEST
            f_path, file_name = os.path.split(self.test_path)
            self.raw_data_result = self.do_load_json(Path(f_path), file_name)

    def do_load_json(self, f_path, file_name):
        f_path = f_path.joinpath(file_name)
        with f_path.open("r") as f:
            raw_data = json.load(f)
            data = self.dataset_preprocess(raw_data)
            logger.info(f"Data loaded: {file_name}.")
        return data

    def mk_result_dataset(self, tokenizer, args=None):
        self.num_eval = int(args.num_eval*len(self.raw_data_result))
        return TokenizedForGenRightPad(self.raw_data_result[:self.num_eval], tokenizer, self.multiple_gen_promptify)

    def dataset_preprocess(self, raw_data):
        data = []
        for e in raw_data:
            data.append({"instruction": e["instruction"].strip(), "input": e["input"].strip(), "output": e["output"].strip()})
        return data

    def handcrafted_exemplars(self):
        raise NotImplementedError

    def exemplar_seperator(self):
        if self.prompt_version.startswith("sp"):
            return "\n\n"
        else:
            raise ValueError(f"TREC: Not supported prompt_version: {self.prompt_version}")

    def multiple_gen_promptify(self, instruction, input, output):
        if self.prompt_version.startswith("sp"):
            if input != "":
                with_query = f"Instruction:\n{instruction}\nInput:\n{input}\nResponse:\n"
            else:
                with_query = f"Instruction:\n{instruction}\nResponse:\n"

            with_query_and_choice = f"{with_query}{output}"
        else:
            raise ValueError(f"ALPACA_INSTRUCTION: Not supported prompt_version: {self.prompt_version}")
        return with_query, with_query_and_choice
    
    def stratified_sampling(self, num_k_shots):
        num_shots = num_k_shots

        if not self.can_be_stratified:
            logger.info("Cannot be stratified, fallback to random selection.")
            return self.random_selected_exemplars(num_shots)

        prefix = ""

        ex_list = [[e["instruction"], e["input"], e["output"]] for e in self.raw_data_sample]

        self._cached_prefix = prefix
        self._cached_ex_list = ex_list
        return self.build_exemplar_from_examples(prefix, ex_list)

    def build_exemplar_from_examples(self, prefix, ex_list):
        ex_prompted = []

        for instruction, input, output in ex_list:
            _, line = self.multiple_gen_promptify(instruction, input, output)  # query, <query_with_answer>
            if len(prefix):
                line = prefix + line
            ex_prompted.append(line)
        return ex_prompted

    def post_process(self, generated_info, metric_output=True, generated_zero_info=None):
        full_info = []
        num_tested = 0
        lm_log_p_zero, lm_log_p_icl = [], []
        norm_lm_log_p_zero, norm_lm_log_p_icl = [], []
        for zero_info, icl_info in zip(generated_zero_info, generated_info):
            zero_info = zero_info[0]
            icl_info = icl_info[0]
            lm_log_p_zero.append(zero_info["lm_log_p"])
            norm_lm_log_p_zero.append(zero_info["norm_lm_log_p"])
            lm_log_p_icl.append(icl_info["lm_log_p"])
            norm_lm_log_p_icl.append(icl_info["norm_lm_log_p"])
        lm_log_p_zero, lm_log_p_icl = np.array(lm_log_p_zero), np.array(lm_log_p_icl)
        norm_lm_log_p_zero, norm_lm_log_p_icl = np.array(norm_lm_log_p_zero), np.array(norm_lm_log_p_icl)

        win_rate = np.greater(lm_log_p_icl, lm_log_p_zero)
        print(np.sum(win_rate))
        win_rate = np.sum(win_rate)/len(win_rate)
        norm_win_rate = np.greater(norm_lm_log_p_icl, norm_lm_log_p_zero)
        norm_win_rate = np.sum(norm_win_rate)/len(norm_win_rate)

        if metric_output:
            logger.info("v" * 30)
            logger.info(f"Acc @ lm_log_p :  = {win_rate:.4f}")
            logger.info(f"Acc @ norm_lm_log_p :  = {norm_win_rate:.4f}")
            logger.info("^" * 30)

        acc_info = {"lm_log_p win rate": f"{(win_rate * 100 ):.4f}", "norm_lm_log_p": f"{(norm_win_rate * 100 ):.4f}"}

        lm_log_p_zero = np.round(lm_log_p_zero, decimals=2)
        lm_log_p_icl = np.round(lm_log_p_icl, decimals=2)

        return (win_rate, norm_win_rate), acc_info, np.vstack((lm_log_p_zero, lm_log_p_icl))

