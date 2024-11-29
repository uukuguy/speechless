from .prompting import pdelim, format
from typing import Dict

ALPACA_DELIMITERS = {
    pdelim.PREQUERY: '### Instruction:\n',
    pdelim.POSTQUERY: '\n### Response:\n',
}


class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["input"], delimiters=cls.get_delimiters())

    @classmethod
    def get_output(cls, record) -> str:
        return ALPACA_DELIMITERS[pdelim.POSTQUERY] + record['output']

    @classmethod
    def get_delimiters(cls) -> Dict:
        return ALPACA_DELIMITERS
