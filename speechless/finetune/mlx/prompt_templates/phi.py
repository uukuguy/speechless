from .prompting import pdelim, format
from typing import Dict


class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["input"], delimiters=cls.get_delimiters())

    @classmethod
    def get_output(cls, record) -> str:
        return f"<|assistant|>\n{record['output']}"

    @classmethod
    def get_delimiters(cls) -> Dict:
        return {
            pdelim.PREQUERY: '<|user|>\n',
            pdelim.POSTQUERY: '<|end|>\n'}

