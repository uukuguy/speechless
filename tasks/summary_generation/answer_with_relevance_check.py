#!/usr/bin/env python

import os
import re
from collections import OrderedDict
from glob import glob
from llm_utils import LLMClient
from loguru import logger
from tqdm import tqdm
from multiprocess_utils import initialize_multiprocessing, run_func_in_multiprocessing

llm_client = LLMClient()

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
def deduplicate_text(text):
    lines = text.split('\n')
    unique_lines = OrderedDict()
    number_pattern = re.compile(r'^\s*(?:[\[\(\{]?\d+[\.\)]?[\]\)\}]?\.?\s*)?(?:[\"\'])?')
    for line in lines:
        line = line.strip()
        if line:
            cleaned_line = number_pattern.sub('', line)
            cleaned_line = cleaned_line.rstrip('"\'')
            unique_lines[cleaned_line] = None
    return '\n'.join(unique_lines.keys())
def ShortenInitialAnswer(Response):
    Quotes = deduplicate_text(Response.split('</quotes>', 1)[0].strip())
    Response = Quotes + '\n</quotes>\n' + Response.split('</quotes>', 1)[1]
    return Response
def wrap_specific_tags_with_cdata(tag_name, content):
    tag_pattern = f'<{tag_name}>(.*?)</{tag_name}>'
    replacement = f'<{tag_name}><![CDATA[\\1]]></{tag_name}>'
    return re.sub(tag_pattern, replacement, content, flags=re.DOTALL)
tags_to_wrap = ["Quotes", "English", "Chinese"]

HEAD = '''Read the questions and answers provided below. First, critically assess the overall relevance of the answers provided to the set of questions asked.
If, upon your assessment, you find that the answers do not contain information that is relevant to the questions asked, stop your review process immediately and respond with a single sentence: "※※※※※※※The provided answers are not relevant to the questions.※※※※※※※". Do not provide any additional explanation or background information, only this sentence should be given as a response in case of irrelevant answers.
If, however, the answers are relevant to the questions asked, proceed to compile answers for each question according to the instructions below. Ensure to aggregate all the relevant answers from the multiple answer results provided in the document, and organize them sequentially by their order number, compiling the corresponding quotes, English answers, and Chinese answers for each question.
If the provided answers' quotes are not differentiated by question, ensure to break them down and assign the quotes to each respective question, outputting them separately within each question’s section.
To provide a comprehensive review, differentiate the responses into quotes, English answers, and Chinese answers for each question based on the details given in the 'Answer' XML tags. Structure your review using the XML format showcased below if the answers are relevant to the questions asked:
<?xml version="1.0" encoding="UTF-8"?>
<Questions>
   <Question number="1"> 
      <Quotes>
          Quotes for question 1 from all the answer results
      </Quotes>
      <English>
          Aggregated English answer for question 1 from all the answer results
      </English>
      <Chinese>
          所有答案结果中的汇总中文答案 1
      </Chinese>
   </Question>
   <Question number="2">
      <Quotes>
          Quotes for question 2 from all the answer results
      </Quotes>
      <English>
          Aggregated English answer for question 2 from all the answer results
      </English>
      <Chinese>
          所有答案结果中的汇总中文答案 2
      </Chinese>
  </Question>
</Questions>
Here are the question lists, in <questions></questions>XML tags:
<questions>
'''
MIDDLE = '''
</questions>
Here are the answer lists, in <Answer></Answer>XML tags:
<Answer>
'''
NotRelevant = [i.replace(' ', '') for i in '''Iamunabletoprovide
Iamnotabletoprovide
Icannotprovide
Unfortunatelytheliteratureprovideddoesnot
Unfortunatelytheprovidedliteraturedoesnot
UnfortunatelyIdonothaveenoughrelevantinformation
theliteraturedoesnotcontain
theliteratureprovideddoesnotcontain
itdoesnotseemtocontain
Iamunabletoconclusivelyanswer
Idonotbelievetheycontain
Iwillbeunabletocompile
.However,Icansummarize
Unfortunatelytheprovidedanswersdonotcontain
※※※Theprovidedanswersarenotrelevanttothequestions.※※※'''.split('\n')]

from knowledge_extraction import get_detailed_questions, generate_part_questions_list

def generate_prompts(args):
    query = args.query

    root_dir = f"outputs/{query}"
    os.makedirs(root_dir, exist_ok=True)

    relevance_check_dir = f"{root_dir}/relevance_check"
    os.makedirs(relevance_check_dir, exist_ok=True)

    relevance_check_prompts_dir = f"{root_dir}/relevance_check/cached_prompts"
    os.makedirs(relevance_check_prompts_dir, exist_ok=True)

    knowledge_extraction_cached_prompts_dir = f"{root_dir}/knowledge_extraction/cached_prompts"
    question_prompt_files = glob(f"{knowledge_extraction_cached_prompts_dir}/Prompt_*.txt")

    # question_prompt_files and answer_files 
    answer_files = [ question_prompt_file.replace("cached_prompts", "generated_answers").replace("Prompt_", "Answer_") for question_prompt_file in question_prompt_files ]
    logger.info(f"Found {len(question_prompt_files)} question prompt files and {len(answer_files)} answer files")

    # Answer file name fromat: Answer_<paper_id>.PART<part_number>.CHUNK<chunk_number>txt
    # eg: Answer_65499d88939a5f4082be9b60.PART7.CHUNK8.txt 
    # A paper is consisted of multiple parts, and each part is consisted of multiple chunks.
    # Construct the paper answer dictionary, where the key is the paper_id and the value is a object with the following attributes:
    # - paper_id
    # - parts: a list of part objects, each with the following attributes:
    #   - part_number
    #   - chunks: a list of chunk objects, each with the following attributes:
    #     - chunk_number
    #     - content
    paper_answer_dict = {}
    for answer_file in tqdm(answer_files, desc="Constructing paper answers"):
        answer_file_basename = os.path.basename(answer_file)
        paper_id = answer_file_basename.split("_")[1].split(".")[0]
        part_number = answer_file_basename.split("PART")[1].split(".")[0]
        chunk_number = answer_file_basename.split("CHUNK")[1].split(".")[0]
        with open(answer_file, "r", encoding="utf-8") as f:
            content = f.read()
        if paper_id not in paper_answer_dict:
            paper_answer_dict[paper_id] = {"paper_id": paper_id, "parts": []}
        part = next((part for part in paper_answer_dict[paper_id]["parts"] if part["part_number"] == part_number), None)
        if part is None:
            part = {"part_number": part_number, "chunks": []}
            paper_answer_dict[paper_id]["parts"].append(part)
        chunk = {"chunk_number": chunk_number, "content": content}
        part["chunks"].append(chunk)
    logger.info(f"Constructed {len(paper_answer_dict)} paper answers")

    chunk_size = 7
    detailed_questions = get_detailed_questions(root_dir)
    AllPrompt = generate_part_questions_list(detailed_questions, chunk_size=chunk_size)

    model_name = os.getenv("OPENAI_DEFAULT_MODEL")
    llm_client = LLMClient(model_name=model_name)

    for paper_id, paper_answer in tqdm(paper_answer_dict.items(), desc="Processing paper answers"):
        relevance_check_prompts = {}
        parts = paper_answer["parts"]
        num_parts = len(parts)

        AnswerDict = {}
        for part in parts:
            part_index = int(part["part_number"])
            chunks = part["chunks"]
            # AnswerDict[part_index] = "\n".join([ c["content"] for c in chunks])
            AnswerDict[part_index] = [c["content"] for c in chunks]

        for part_index, Answer in tqdm(AnswerDict.items(), desc="Generating relevance check prompts"):
            prompt_index = part_index // chunk_size
            if prompt_index < len(AllPrompt):
                prompt = HEAD + AllPrompt[prompt_index] + MIDDLE + '\n</Answer>\n<Answer>\n'.join(Answer).replace(
                    'Human: ', '') + '\n</Answer>\n'
                relevance_check_prompts[part_index] = prompt
                relevance_check_prompt_file = f"{relevance_check_prompts_dir}/Prompt_{paper_id}.PART{part_index}.txt"
                if not os.path.exists(relevance_check_prompt_file):
                    with open(relevance_check_prompt_file, "w", encoding="utf-8") as f:
                        f.write(prompt)
            else:
                print(f"Warning: No corresponding prompt for {paper_id}-PART{part}")
    logger.info(f"Generated {len(paper_answer_dict)} papers for relevance check prompts")


def generate_answers_for_prompt(params: dict):
    prompt_file = params['prompt_file']
    max_tokens = params['max_tokens']
    # llm_client = params['llm_client']

    num_generated = 0
    output_filename = prompt_file.replace("cached_prompts", "generated_answers").replace("Prompt_", "Answer_")
    if not os.path.exists(output_filename):
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
        generated_text = llm_client.generate(
            prompt, system_prompt="你是一个帮助进行学术综述撰写的专家。", generate_params={
                "max_tokens": max_tokens
            }
        )
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(generated_text)
        num_generated += 1
    return {
        'num_generated': num_generated
    }

def answer_questions_for_literature(args):
    query, num_processes, pool_map_chunk_size = args.query, args.num_processes, args.pool_map_chunk_size

    root_dir = f"outputs/{query}"
    os.makedirs(root_dir, exist_ok=True)


    relevance_check_dir = f"{root_dir}/relevance_check"
    os.makedirs(relevance_check_dir, exist_ok=True)

    relevance_check_prompts_dir = f"{relevance_check_dir}/cached_prompts"
    os.makedirs(relevance_check_prompts_dir, exist_ok=True)

    relevance_check_answers_dir = f"{relevance_check_dir}/generated_answers"
    os.makedirs(relevance_check_answers_dir, exist_ok=True)

    num_generated = 0
    all_prompt_files = glob(f"{relevance_check_prompts_dir}{os.sep}Prompt_*.txt")
    if num_processes > 1:
        params_list = [{
            'prompt_file': prompt_file,
            'max_tokens': 128000,
            # 'llm_client': llm_client
        } for prompt_file in all_prompt_files]

        results = []
        for result in run_func_in_multiprocessing(
            generate_answers_for_prompt,
            params_list,
            num_processes=num_processes,
            chunk_size=pool_map_chunk_size,
            unordered=True,
            use_progress_bar=True,
            progress_bar_desc="Answering questions"
        ):
            if result.is_success():
                results.append(result.result)
                num_generated +=1 

    else:
        for prompt_file in tqdm(all_prompt_files, desc="Knowledge extraction"):
            params = {
                'prompt_file': prompt_file,
                'max_tokens': 128000,
                # 'llm_client': llm_client
            }
            result = generate_answers_for_prompt(params)
            num_generated += result['num_generated']
        logger.info(f"Knowledge extraction: Generated {num_generated} answers.")



def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True, help="Query")
    parser.add_argument("--generate_prompts", action="store_true", help="Generate prompts for literature")
    parser.add_argument("--answer_questions", action="store_true", help="Answer questions for literature")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use")
    parser.add_argument("--pool_map_chunk_size", type=int, default=1, help="Pool map chunk size")
    return parser.parse_args()

def main():
    args = get_args()
    if args.generate_prompts:
        generate_prompts(args)
    if args.answer_questions:
        answer_questions_for_literature(args)

if __name__ == "__main__":
    initialize_multiprocessing()
    main()