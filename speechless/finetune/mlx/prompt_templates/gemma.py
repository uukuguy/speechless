from .prompting import pdelim, format
from typing import Dict


class TrainingRecordHandler:
    @classmethod
    def get_input(cls, record) -> str:
        return format(record["input"], delimiters=cls.get_delimiters())

    @classmethod
    def get_output(cls, record) -> str:
        return f"<start_of_turn>model\n{record['output']}<end_of_turn>"

    @classmethod
    def get_delimiters(cls) -> Dict:
        return {
            pdelim.PREQUERY: '<bos><start_of_turn>user\n',
            pdelim.POSTQUERY: '<end_of_turn>\n'}

