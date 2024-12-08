from .prompting import format, CHATML_DELIMITERS
from typing import Dict

class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["instruction"], delimiters=cls.get_delimiters())

    @classmethod
    def get_output(cls, record) -> str:
        return record["response"]

    @classmethod
    def get_delimiters(cls) -> Dict:
        return CHATML_DELIMITERS
