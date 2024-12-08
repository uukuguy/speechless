from .prompting import pdelim, format
from typing import Dict

#https://github.com/OoriData/OgbujiPT/pull/70
LLAMA3_DELIMITERS = {
    pdelim.FIXED_PREAMBLE: '<|begin_of_text|>',
    pdelim.PRECONTEXT: '<|start_header_id|>system<|end_header_id|>\n\n',
    pdelim.POST_ALL_CONTEXT: '<|eot_id|>',
    pdelim.PREQUERY: '<|start_header_id|>user<|end_header_id|>\n\n',
    pdelim.POSTQUERY: '<|eot_id|>'
}


class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["instruction"], delimiters=cls.get_delimiters())

    @classmethod
    def get_output(cls, record) -> str:
        return f"<|start_header_id|>assistant<|end_header_id|>\n\n{record['response']}<|eot_id|>"

    @classmethod
    def get_delimiters(cls) -> Dict:
        return LLAMA3_DELIMITERS
