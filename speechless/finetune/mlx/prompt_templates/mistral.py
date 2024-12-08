from .prompting import pdelim, format
from typing import Dict

#https://github.com/OoriData/OgbujiPT/pull/70
MISTRAL_INSTRUCTION_DELIMITERS_NO_BOS = {
    pdelim.FIXED_PREAMBLE: '[INST]',
    pdelim.POSTQUERY: '\n[/INST]',
}


class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["instruction"], delimiters=cls.get_delimiters())

    @classmethod
    def get_output(cls, record) -> str:
        return record["response"]

    @classmethod
    def get_delimiters(cls) -> Dict:
        return MISTRAL_INSTRUCTION_DELIMITERS_NO_BOS
