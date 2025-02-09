#!/usr/bin/env python
import os
import re
import json
import pickle
import tiktoken
import math
from typing import List
from tqdm import tqdm
from loguru import logger
from glob import glob
from llm_utils import LLMClient
from multiprocess_utils import initialize_multiprocessing, run_func_in_multiprocessing

llm_client = LLMClient()

FormattedPromptHead = '''I'm going to give you a scientific literature. Then I'm going to ask you some questions about it. I'd like you to first write down exact quotes of parts of the document word by word that would help answer the question, and then I'd like you to answer the question using facts from the quoted content. Do not omit any relevant information from the text, and avoid introducing any falsehoods or assumptions that aren't directly supported by the literature. Here is the literature, in <literature></literature> XML tags:
<literature>
'''
FormattedPromptMiddle = '''
</literature>
Here are the question lists, in <questions></questions>XML tags:
<questions>
'''
FormattedPromptEnd = '''
</questions>
First, you need to sequentially extract any quotes in the literature that are most relevant to each question, and print them in numbered order, separated by newlines. Quotes should be relatively brief. Do not attempt to summarize or answer questions at this stage, but simply repeat exactly what the corresponding part of the literature says.
Please enclose the full list of quotes in <quotes></quotes> XML tags. If there are no relevant quotes, write "No relevant quotes" instead.
Then, answer each question, starting with "Answer:".  Do not include or reference quoted content verbatim in the answer. Don't say "According to Quote [1]" when answering. Do not write reference number of quotes after answer. Put your answer to the user inside <answer></answer> XML tags. Output formatted text, with line breaks for each question.Separate quotes and answers with a blank line. Provide the answers to all questions in English. After completing the English answers, translate all those answers into Chinese and provide the Chinese version.
Thus, the format of your overall response should look like what's shown between the <example></example> tags.  Make sure to follow the formatting and spacing exactly.
<example>
<quotes>
[1] "Company X reported revenue of $12 million in 2021."
</quotes>
<English version answer>
1.Company X earned $12 million in 2021.
</English version answer>
<Chinese version answer>
1.X公司在2021年赚了1200万美元。
</Chinese version answer>
<quotes>
[1] "Almost 90% of revenue came from widget sales, with gadget sales making up the remaining 10%."
</quotes>
<English version answer>
2.Almost 90% of it came from widget sales.
</English version answer>
<Chinese version answer>
2.几乎90%的收入来自小部件销售。
</Chinese version answer>
</example>
If the question cannot be answered by the document, say so. If deemed necessary, the answer to the question can be extended entirely from the content of the document.
Answer all of the questions immediately without preamble. '''
TitlePattern = r'^[^a-zA-Z0-9\u4e00-\u9FFF\s]*\s*(' + '|'.join([
    'ACKNOWLEDGMENT',
    'ACKNOWLEDGEMENT',
    'SUPPLEMENTARY MATERIAL',
    'REFERENCE',
    'References',
    'DATA AVAILABILITY',
    'Declaration of competing interest',
    'ABBREVIATIONS',
    'ASSOCIATED CONTENT',
    'Conflicts of interest',
    'Supporting Information',
]) + ')'
InvaildSymbolPattern = r"[^a-zA-Z0-9\u4e00-\u9fa5\u0370-\u03FF ,.!?\-_:;'\"(){}\[\]&<>%\$@\*/=#·Å+•×\\]"


def MergeParagraphs(text):
    ending_punctuation = '.!?。！？'
    paired_punctuation = {
        '(': ')',
        '[': ']',
        '{': '}',
        '<': '>',
        '（': '）',
        '【': '】',
        '《': '》',
        '「': '」',
        '『': '』',
        '"': '"',
        "'": "'",
        '`': '`'
    }
    all_opening = ''.join(paired_punctuation.keys())
    all_closing = ''.join(paired_punctuation.values())
    lines = text.split('\n')
    processed_lines = []
    current_paragraph = ""
    in_special_structure = False
    special_structure_lines = []
    punctuation_stack = []

    def is_sentence_end(s):
        return (
            s and s[-1] in ending_punctuation
            and (len(punctuation_stack) == 0 or punctuation_stack[-1] in ['"', '"', ''', '''])
        )

    def is_heading(line):
        return (
            re.match(r'^#{1,6}\s', line) or re.match(r'^[A-Z0-9\u4e00-\u9fa5]{1,20}[.、:：]', line)
            or (len(line) <= 40 and line.isupper())
        )

    for line in lines:
        line = line.strip()
        if re.match(r'^[-+|]{3,}$', line) or re.match(r'^```', line):
            in_special_structure = not in_special_structure
            if in_special_structure:
                if current_paragraph:
                    processed_lines.append(current_paragraph)
                    current_paragraph = ""
                special_structure_lines = [line]
            else:
                processed_lines.extend(special_structure_lines)
                special_structure_lines = []
            continue
        if in_special_structure:
            special_structure_lines.append(line)
            continue
        if line:
            for char in line:
                if char in all_opening:
                    punctuation_stack.append(char)
                elif char in all_closing:
                    if punctuation_stack and paired_punctuation[punctuation_stack[-1]] == char:
                        punctuation_stack.pop()
            if is_heading(line):
                if current_paragraph:
                    processed_lines.append(current_paragraph)
                processed_lines.append(line)
                current_paragraph = ""
                punctuation_stack.clear()
            else:
                if (
                    current_paragraph and is_sentence_end(current_paragraph)
                    and (line[0].isupper() or line[0] in all_opening)
                ):
                    processed_lines.append(current_paragraph)
                    current_paragraph = line
                else:
                    if current_paragraph:
                        if (current_paragraph[-1] in ending_punctuation and line[0].isupper()) or \
                                (current_paragraph[-1] in ',:;，：；' and not line[0].isupper()):
                            current_paragraph += "\n" + line
                        else:
                            current_paragraph += " " + line
                    else:
                        current_paragraph = line
        else:
            if current_paragraph:
                processed_lines.append(current_paragraph)
                current_paragraph = ""
                punctuation_stack.clear()
    if current_paragraph:
        processed_lines.append(current_paragraph)
    return '\n\n'.join([
        i.replace('\n', '').replace('. . . ', '').replace('. . . ', '').replace('. . . ', '') for i in processed_lines
    ]).replace('. . . ', '').replace('. . . ', '').replace('. . . ', '').replace('. . . ', '').replace('. . . ', '')


def GetRefineContents(Contents):
    for i in range(10):
        Contents = Contents.replace('. . . ', '')
    Contents = Contents.replace('ﬀ', 'ff').replace('', 'fi').replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace(
        'ﬃ', 'ffi'
    ).replace('ﬄ', 'ffl').replace('ﬅ', 'ft').replace('ﬆ', 'st').split('\n')
    Contents = [re.sub(InvaildSymbolPattern, '', Content) for Content in Contents]
    Final = []
    threshold = 320
    for x in Contents:
        if len(x) < threshold and re.match(TitlePattern, x.strip(), re.IGNORECASE) and Final:
            break
        else:
            Final.append(x)
    return MergeParagraphs('\n'.join(Final))


def split_text(text, max_tokens):
    """Split text into chunks that don't exceed max_tokens while preserving paragraphs."""
    total_tokens = num_tokens_from_string(text)
    if total_tokens <= max_tokens:
        return [text]
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    current_tokens = 0
    for paragraph in paragraphs:
        paragraph_tokens = num_tokens_from_string(paragraph)
        if paragraph_tokens > max_tokens:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_tokens = num_tokens_from_string(sentence)
                if current_tokens + sentence_tokens > max_tokens and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sentence
                    current_tokens = sentence_tokens
                else:
                    current_chunk += sentence
                    current_tokens += sentence_tokens
        else:
            if current_tokens + paragraph_tokens > max_tokens and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
                current_tokens = paragraph_tokens
            else:
                if current_chunk:
                    current_chunk += "\n\n"
                current_chunk += paragraph
                current_tokens += paragraph_tokens
    if current_chunk:
        chunks.append(current_chunk.strip())
    if num_tokens_from_string(chunks[-2] + chunks[-1]) < max_tokens * 1.2:
        chunks = chunks[:-2] + [chunks[-2] + chunks[-1]]
    return chunks


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def get_detailed_questions(root_dir: str):
    detailed_questions_file = f"{root_dir}/review_outlines/detailed_questions.txt"
    detailed_questions = open(detailed_questions_file, "r").readlines()
    detailed_questions = [line.strip() for line in detailed_questions if line.strip()]
    detailed_questions = [re.sub(r"^\d+\.\s+", "", line) for line in detailed_questions]

    return detailed_questions

def generate_part_questions_list(detailed_questions: List[str], chunk_size: int = 7):
    chunk_size = 7
    AllPrompt = []
    if len(detailed_questions) > chunk_size:
        for i in range(0, len(detailed_questions), chunk_size):
            AllPrompt.append('\n'.join(detailed_questions[i:i + chunk_size]))
    else:
        AllPrompt.extend(detailed_questions)

    return AllPrompt


def generate_prompts(query: str, max_tokens: int = 4096):

    root_dir = f"outputs/{query}"
    os.makedirs(root_dir, exist_ok=True)

    knowledge_extraction_dir = f"{root_dir}/knowledge_extraction"
    os.makedirs(knowledge_extraction_dir, exist_ok=True)

    knowledge_extraction_cached_prompts_dir = f"{root_dir}/knowledge_extraction/cached_prompts"
    os.makedirs(knowledge_extraction_cached_prompts_dir, exist_ok=True)

    prompt_framework = FormattedPromptHead + FormattedPromptMiddle + FormattedPromptEnd
    framework_tokens = num_tokens_from_string(prompt_framework)
    max_doc_tokens = max_tokens - framework_tokens

    papers_content_file = f"{root_dir}/retrieve_papers/papers_content.pkl"
    papers = pickle.load(open(papers_content_file, "rb"))

    # detailed_questions_file = f"{root_dir}/detailed_questions.txt"
    # detailed_questions = open(detailed_questions_file, "r").readlines()
    # detailed_questions = [line.strip() for line in detailed_questions if line.strip()]
    # detailed_questions = [re.sub(r"^\d+\.\s+", "", line) for line in detailed_questions]
    detailed_questions = get_detailed_questions(root_dir)

    # chunk_size = 7
    # AllPrompt = []
    # if len(detailed_questions) > chunk_size:
    #     for i in range(0, len(detailed_questions), chunk_size):
    #         AllPrompt.append('\n'.join(detailed_questions[i:i + chunk_size]))
    # else:
    #     AllPrompt.extend(detailed_questions)
    AllPrompt = generate_part_questions_list(detailed_questions, chunk_size=7)

    for paper_id, paper in tqdm(papers.items(), desc="Generate prompts"):

        # doc = paper.content
        # doc = GetRefineContents(doc)
        # doc_tokens = num_tokens_from_string(doc)
        # if doc_tokens > max_doc_tokens:
        #     num_chunks = math.ceil(doc_tokens / max_doc_tokens)
        #     tokens_per_chunk = math.floor(doc_tokens / num_chunks)
        #     doc_chunks = split_text(doc, tokens_per_chunk)
        # else:
        #     doc_chunks = [doc]
        paper_chunks = list(paper.chunks.values())

        for chunk_index, paper_chunk in enumerate(paper_chunks):
            paper_id = paper_chunk.paper_id
            chunk_id = paper_chunk.chunk_id
            chunk_content = f"<chunk id={paper_id}-{chunk_id}>" + paper_chunk.content + "</chunk>"
            for i, p in enumerate(AllPrompt):
                full_prompt = (
                    FormattedPromptHead + '\n' + chunk_content.strip().replace('\n\n', '\n') + '\n' +
                    FormattedPromptMiddle + '\n' + p.strip().replace('\n\n', '\n') + '\n' + FormattedPromptEnd
                )
                output_filename = f'{knowledge_extraction_cached_prompts_dir}{os.sep}Prompt_{paper_id}.PART{i}.CHUNK{chunk_index}.txt'
                with open(output_filename, 'w', encoding='utf-8') as f:
                    f.write(full_prompt)

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

def answer_questions_for_literature(query: str, num_processes: int = 1, pool_map_chunk_size: int = 1):

    root_dir = f"outputs/{query}"
    os.makedirs(root_dir, exist_ok=True)

    knowledge_extraction_dir = f"{root_dir}/knowledge_extraction"
    os.makedirs(knowledge_extraction_dir, exist_ok=True)

    knowledge_extraction_cached_prompts_dir = f"{root_dir}/knowledge_extraction/cached_prompts"
    os.makedirs(knowledge_extraction_cached_prompts_dir, exist_ok=True)

    llm_client = LLMClient()
    # generated_text = llm_client.generate(prompt, system_prompt="你是一个帮助进行学术综述撰写的专家。")

    knowledge_extraction_generated_answers_dir = f"{root_dir}/knowledge_extraction/generated_answers"
    os.makedirs(knowledge_extraction_generated_answers_dir, exist_ok=True)
    logger.info(
        f"Knowledge extraction: Generating answers for all prompts to {knowledge_extraction_generated_answers_dir}..."
    )


    num_generated = 0
    all_prompt_files = glob(f"{knowledge_extraction_cached_prompts_dir}{os.sep}Prompt_*.txt")
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
            # output_filename = prompt_file.replace("cached_prompts", "generated_answers").replace("Prompt_", "Answer_")
            # if not os.path.exists(output_filename):
            #     with open(prompt_file, 'r', encoding='utf-8') as f:
            #         prompt = f.read()
            #     generated_text = llm_client.generate(
            #         prompt, system_prompt="你是一个帮助进行学术综述撰写的专家。", generate_params={
            #             "max_tokens": 128000
            #         }
            #     )
            #     with open(output_filename, 'w', encoding='utf-8') as f:
            #         f.write(generated_text)
            #     num_generated += 1
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
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--generate_prompts", action="store_true", help="Generate prompts for literature")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max tokens for each prompt")
    parser.add_argument("--answer_questions", action="store_true", help="Answer questions for literature")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of processes to use")
    parser.add_argument("--pool_map_chunk_size", type=int, default=1, help="Pool map chunk size")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    if args.generate_prompts:
        generate_prompts(args.query, max_tokens=args.max_tokens)
    if args.answer_questions:
        answer_questions_for_literature(
            args.query, num_processes=args.num_processes, pool_map_chunk_size=args.pool_map_chunk_size
        )


if __name__ == "__main__":
    initialize_multiprocessing()
    main()
