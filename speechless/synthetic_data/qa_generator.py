#!/usr/bin/env python
# Inspired by: https://github.com/virattt/financial-datasets
import os, json, time
import random
from io import BytesIO
from typing import List, Tuple

import requests
from PyPDF2 import PdfReader
import instructor
from instructor import patch
from langchain_text_splitters import TokenTextSplitter
from openai import OpenAI
from litellm import completion
from tqdm import tqdm


from pydantic.main import BaseModel


class QADatasetItem(BaseModel):
    question: str
    answer: str
    context: str


class QADataset(BaseModel):
    items: List[QADatasetItem]


default_prompt = """
You are an expert at understanding and analyzing power grid documents. 
Your role is to generate question and ground truth answer pairs based on the provided power grid knowledge text. The generated question and answer must be using Chinese and must be relevant to the power grid context.

When generating questions and answers, adhere to the following guidelines:
1. Your ground truth answers must be directly derived from the content within the provided text. Do not make up, hallucinate, or generate answers that are not explicitly supported by the given text.
2. Ensure that the questions you generate are relevant to the financial context and can be answered based on the information provided in the text.
3. Include the relevant 'context' paragraph from which you generated each question and ground truth answer pair. The 'context' paragraph MUST contain the specific information that supports the ground truth answer.
4. If the provided text does not contain sufficient information to generate a question-answer pair, do not attempt to create one.
5. Your responses should be in the following format:
   Question: [Generated question]
   Answer: [Ground truth answer]
   Context: [Relevant paragraph from the text that supports the answer]

Remember, your primary objective is to create accurate, grounded, and contextually relevant question-answer pairs while strictly avoiding any fabrication or speculation.
"""

class QAGenerator:
    def __init__(self, model: str):
        self._model = model
        if "gpt-" in model:
            self._client = patch(OpenAI())
        elif "claude-" in model:
            # claude-3-opus-20240229
            from anthropic import Anthropic
            self._client = instructor.from_anthropic(Anthropic())
        elif "command-r-" in model:
            import cohere
            self._client = instructor.from_cohere(cohere.Client())
        else:
            self._client = instructor.from_litellm(completion)

    def generate_from_texts(
        self,
        texts: List[Tuple[int, str]],
        max_questions=10,
    ) -> QADataset:
        """
        Generate questions from a list of texts.

        :param texts: List of texts to generate questions from.
        :param max_questions: Maximum number of questions to generate.
        :return: Dataset containing the generated questions.
        """
        items: List[QADatasetItem] = []
        ref_ids: List[int] = []
        num_texts = len(texts)
        questions_per_text = max_questions // num_texts
        remaining_questions = max_questions % num_texts

        progress_bar = tqdm(total=max_questions, desc="Generating questions", colour='green')

        for index, (ref_id, text) in enumerate(texts):
            # print(f"{len(text)=}, {text=}")
            try:
                # Determine the number of questions to generate for the current text
                current_max_questions = questions_per_text
                if index < remaining_questions:
                    current_max_questions += 1

                # Generate questions
                response = self._client.chat.completions.create(
                    model=self._model,
                    response_model=QADataset,
                    messages=[
                        {"role": "system", "content": default_prompt},
                        {"role": "user", "content": f"Generate {current_max_questions} questions for the following block of text: {text}"}
                    ],
                )
                print(f"{len(items)=}, {response=}")

                # Add the generated items to our total list of questions
                items.extend(response.items)
                ref_ids.extend([ref_id] * len(response.items))

                # Update the progress bar by the number of questions generated
                progress_bar.update(len(response.items))

                # Stop generating questions if we have reached the maximum number of questions
                if len(items) >= max_questions:
                    break

            except Exception as e:
                print(f"Failed to generate questions for batch {index + 1}: {e}")
                continue

            # Sleep for 1 second to avoid overloading the LLM
            # time.sleep(1)

        # Ensure the progress bar is closed
        progress_bar.close()

        return QADataset(
            items=items,
        ), ref_ids

    def generate_from_pdf(
        self,
        url: str,
        max_questions=10,
        **kwargs,
    ) -> QADataset:
        """
        Generate questions from a PDF file.

        :param url: The URL of the PDF file.
        :param max_questions: Maximum number of questions to generate.
        :param kwargs: Additional arguments like chunk_size, chunk_overlap, etc.
        :return: Dataset containing the generated questions.
        """
        # Download the PDF file
        response = requests.get(url)
        pdf_file = BytesIO(response.content)

        # Extract text from the PDF file
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

        # Remove any newline characters
        text = text.replace("\n", " ")

        # Chunk the text to prevent exceeding the context window of models at the question generation step.
        chunk_size = kwargs.get("chunk_size", 1024)
        chunk_overlap = kwargs.get("chunk_overlap", 128)

        # Split by tokens
        token_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Chunk the text
        texts = token_splitter.split_text(text)

        return self.generate_from_texts(texts=texts, max_questions=max_questions)

def get_gdc_data_texts(gdc_data_file):
    lines = open(gdc_data_file).readlines()
    texts = [(idx, json.loads(line)['prompt']) for idx, line in enumerate(tqdm(lines, ncols=100, desc="Loading"))]
    return texts
    
def do_generate_qa_pairs(args):
    texts = get_gdc_data_texts(args.gdc_data_file)
    texts = [t for t in texts if len(t[1]) > 256]
    texts = random.sample(texts, min(args.max_questions * 10, len(texts)))

    qa_generator = QAGenerator(model=args.model)

    qa_dataset, ref_ids = qa_generator.generate_from_texts(texts=texts, max_questions=args.max_questions)

    json_data = json.loads(qa_dataset.model_dump_json())['items']
    assert len(json_data) == len(ref_ids)
    with open(args.output_file, 'w') as fd:
        for item, ref_id in zip(json_data, ref_ids):
            item['ref_id'] = ref_id
            fd.write(f"{json.dumps(item, ensure_ascii=False)}\n")
    print(f"Saved QA pairs to {args.output_file}")


def get_args():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-1106", help="Model name")
    parser.add_argument("--max_questions", type=int, default=10, help="Maximum number of questions to generate")
    parser.add_argument("--gdc_data_file", type=str, default="/opt/local/datasets/speechless_data/sggdc_data_pt.jsonl", help="Path to the GDC data file")
    parser.add_argument("--output_file", type=str, default="output.jsonl", help="Path to the output file")
    args = parser.parse_args()
    return args
def main():
    args = get_args()
    do_generate_qa_pairs(args)

if __name__ == '__main__':
    main()