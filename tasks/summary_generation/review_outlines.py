import os, sys, re
import tqdm
import time
import shutil
from queue import Queue, Empty
from threading import Thread, Semaphore
from math import inf
import xml.etree.ElementTree as ET
import func_timeout
from io import StringIO
from llm_utils import LLMClient
from rich.console import Console
from rich.markdown import Markdown
from loguru import logger

console = Console()

overtime = inf


def construct_prompt(HEAD, END, HEADFromReview, ENDFromReview, FromReview, REPEAT, File):
    if not os.path.exists(f"{File.replace('.txt', '')}{os.sep}QuestionsFromReview_{REPEAT - 1}.txt"):
        if FromReview:
            Content = open(File, 'r', encoding='UTF8').read()
            Prompt = HEADFromReview + File + "'>\n" + Content + '\n' + ENDFromReview
        else:
            Prompt = HEAD + END
        os.makedirs(File.replace('.txt', ''), exist_ok=True)
        open(File.replace('.txt', '') + os.sep + 'PromptQuestionsFromReview', 'w', encoding='UTF8').write(Prompt)
        return File.replace('.txt', ''), Prompt


def GetResponse(prompts, REPEAT, Threads, FunctionClaudeAPI, FunctionOpenAIAPI, STDOUT):
    results = [None] * len(prompts)
    prompt_queue = Queue()
    for i, (folder, prompt) in enumerate(prompts):
        prompt_queue.put((i, folder, prompt))
    ClaudeAPISemaphore = []
    OpenAIAPISemaphore = []
    FunctionClaudeAPI = {
        k: v
        for k, v in FunctionClaudeAPI.items()
        if v
    }
    FunctionOpenAIAPI = {
        k: v
        for k, v in FunctionOpenAIAPI.items()
        if v
    }
    for i in FunctionClaudeAPI:
        ClaudeAPISemaphore.append(Semaphore(value=1))
    for i in FunctionOpenAIAPI:
        OpenAIAPISemaphore.append(Semaphore(value=3))
    progress_bar = tqdm.tqdm(total=len(prompts) * REPEAT, position=0, desc='GetQuestionsFromReview', file=STDOUT)

    def worker():
        start_time = time.time()
        while True:
            if time.time() - start_time > overtime:
                print(f"Worker timed out for prompt {i}")
                break
            try:
                i, folder, prompt = prompt_queue.get_nowait()
            except Empty:
                break
            for Time in range(REPEAT):
                TRY = 0
                start_time2 = time.time()
                while True:
                    if time.time() - start_time2 > overtime:
                        print(f"Worker timed out for prompt {i}")
                        break
                    semaphore_acquired = None
                    response = None
                    try:
                        for semaphore, GetResponseFunction in zip(ClaudeAPISemaphore, FunctionClaudeAPI.values()):
                            if semaphore.acquire(blocking=False):
                                semaphore_acquired = semaphore
                                response = GetResponseFunction(prompt)
                                semaphore.release()
                                break
                        if not response:
                            for semaphore, GetResponseFunction in zip(OpenAIAPISemaphore, FunctionOpenAIAPI.values()):
                                if semaphore.acquire(blocking=False):
                                    semaphore_acquired = semaphore
                                    response = GetResponseFunction(prompt)
                                    semaphore.release()
                                    break
                        if not response:
                            time.sleep(0.5)
                            continue
                        if len(response) < 200 or response.strip().startswith('Unfortunately'):
                            with open(
                                os.path.join(folder, f'QuestionsFromReview_{Time}-{TRY}.txt'), 'w', encoding='UTF8'
                            ) as file:
                                file.write(response + '\n')
                                TRY += 1
                                continue
                        if 'Outlines' in response:
                            content = '<?xml version="1.0" encoding="UTF-8"?>\n<Outlines>' + response.split(
                                '<Outlines>', 1
                            )[1].rsplit('</Outlines>', 1)[0] + '</Outlines>'
                            try:
                                root = ET.fromstring(content)
                            except Exception as e:
                                with open(
                                    os.path.join(folder, f'QuestionsFromReview_{Time}-{TRY}.txt'), 'w', encoding='UTF8'
                                ) as file:
                                    file.write(response + '\n')
                                    TRY += 1
                                    continue
                        with open(
                            os.path.join(folder, f'QuestionsFromReview_{Time}.txt'), 'w', encoding='UTF8'
                        ) as file:
                            file.write(response + '\n')
                        results[i] = response
                        break
                    except (Exception, func_timeout.exceptions.FunctionTimedOut) as e:
                        pattern = r"Function\s+(\S+).*?timed out after\s+(\d+\.\d+)\s+seconds."
                        replacement = r"\1 timed out after \2 s"
                        if semaphore_acquired:
                            semaphore_acquired.release()
                        open('Exceptionlog', 'a').write(
                            '\t'.join([
                                str(i), folder,
                                time.strftime('%Y-%m-%d_%H.%M.%S', time.localtime()),
                                re.sub(pattern, replacement, str(e), flags=re.DOTALL), '\n'
                            ])
                        )
                        TRY += 1
                        continue
                progress_bar.update(1)

    threads = [Thread(target=worker) for _ in range(Threads)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    progress_bar.close()
    return results


def ReshapeOutlines(text):

    def get_indent_level(line):
        original = line
        line = line.expandtabs(4)
        return len(line) - len(line.lstrip())

    def extract_content(line):
        patterns = [
            r'^\d+\.\d+\.*\s+(.+)$', r'^\d+\.\s+(.+)$', r'^[A-Z]\.\s+(.+)$', r'^[a-z]\)\s+(.+)$', r'^\d+\)\s+(.+)$',
            r'^[IVX]+\.\s+(.+)$', r'^[-•]\s*(.+)$', r'^(.+)$'
        ]
        for pattern in patterns:
            match = re.match(pattern, line.strip())
            if match:
                return match.group(1).strip()
        return line.strip()

    try:
        text = re.sub(r'<\?xml.*?\?>\s*', '', text)
        if '<Outlines>' not in text:
            return text
        content = '<?xml version="1.0" encoding="UTF-8"?>\n<Outlines>' + text.split('<Outlines>',
                                                                                    1)[1].rsplit('</Outlines>',
                                                                                                 1)[0] + '</Outlines>'
        root = ET.fromstring(content)
        formatted_lines = []
        global outline
        for outline in root.findall('.//Outline'):
            outline_text = outline.text.strip() if outline.text.strip() else ''.join([i for i in outline.itertext()])
            if not outline_text:
                continue
            lines = [line.strip() for line in outline_text.split('\n') if line.strip()]
            if not lines:
                continue
            LineIndex = 0
            for line in lines:
                LineIndex += 1
                if not line.strip().startswith('<'):
                    current_section = extract_content(line)
                    break
            for line in lines[LineIndex:]:
                content = extract_content(line)
                if content and content != current_section:
                    formatted_lines.append(f"{current_section} - {content}")
            if len(lines) == 1:
                formatted_lines.append(current_section)
        if not formatted_lines:
            return '\n'.join(
                line.strip() for line in text.split('\n') if line.strip() and not line.strip().startswith(('<', '</'))
            )
        return '\n'.join(formatted_lines)
    except (ET.ParseError, AttributeError):
        lines = [line for line in text.split('\n') if line.strip()]
        if not lines:
            return text
        formatted_lines = []
        current_section = extract_content(lines[0])
        for line in lines[1:]:
            content = extract_content(line)
            if content and content != current_section:
                formatted_lines.append(f"{current_section} - {content}")
        if len(lines) == 1:
            formatted_lines.append(current_section)
        if not formatted_lines:
            return '\n'.join(lines)
        return '\n'.join(formatted_lines)


def GetQuestions(text):
    lines = text.split('\n')
    formatted_lines = []
    current_section = ""
    for line in lines:
        if line.strip().startswith(
            tuple([f"{i}. " for i in range(100)])
        ) and (len(re.findall(r'[a-zA-Z]', line.strip())) / len(line.strip()) > 0.6 if line.strip() else False):
            current_section = line.split('. ')[1].strip()
            formatted_lines.append(current_section)
    return '\n'.join([f'{i + 1}. {j}' for i, j in enumerate(formatted_lines)])


def generate_overall_outline(Folder, TOPIC, Threads, FunctionClaudeAPI, FunctionOpenAIAPI, STDOUT, FromReview=False, MaxToken=25000):
    old_stdout = sys.stdout
    sys.stdout = STDOUT
    REPEAT = 5
    os.chdir(Folder)
    Folders = [i for i in os.listdir('.') if i.startswith('10.') and i.endswith('.txt')]
    Folders.sort()
    HEAD = f"""Your task is to craft a detailed and logical question-based outline for a comprehensive review article about {TOPIC}. The outline should first be presented entirely in English, followed by its translation in Chinese after a line break. The detailed instructions are as follows:
"""
    HEADFromReview = f"""Based on the review about {TOPIC} presented in the attached file content, your task is to craft a detailed and logical question-based outline for a comprehensive review article. The outline should first be presented entirely in English, followed by its translation in Chinese after a line break. The detailed instructions are as follows:
<file-attachment-contents filename='"""
    END = f'''
Ensure that your outline adheres to the following criteria:
1. Develop at least 17 primary themes that are intricately linked to {TOPIC}, paving the way for a structured review article.
2. Under each primary theme, identify at least 7 detailed questions that probe deep into the respective themes, fostering detailed and insightful paragraphs in the final article.
3. Integrate both experimental and computational methodologies, including aspects from DFT and other theoretical approaches, evenly and coherently in the questions to offer a rounded perspective on {TOPIC}.
4. The progression of the primary themes should mirror a traditional review trajectory: starting with the foundational concepts, transitioning to advanced nuances, discussing prevailing challenges, and culminating with future prospects and visions.
5. Formulate the questions such that when answered, they collectively offer a wide-ranging view of the {TOPIC} landscape, without any significant overlap or omission of vital aspects.
Remember that this outline should function as a structured pathway for researchers, guiding them to draft a holistic review article on {TOPIC} without overlooking any critical details.
Start with the complete English version of the outline. After finishing it, provide the Chinese translation, maintaining the depth and essence of the English version, separated by a line break.
'''
    ENDFromReview = '''</file-attachment-contents>
''' + END
    IntegrationQuestionsForReview = f"""Your task is to construct a structured, question-based outline for a comprehensive review article on the topic detailed in the provided attachment, with a significant emphasis on {TOPIC}. The focus should be on presenting a thorough exploration across various dimensions of {TOPIC}, smoothly transitioning from a foundational understanding to an in-depth analysis of specific instances related to {TOPIC}. The outline should facilitate a logical flow between a broad range of themes, consistently maintaining a focal emphasis on the core elements of {TOPIC}.
Instructions:
1. **Introduction**
- Begin with a section that lays the groundwork for the {TOPIC}, detailing its background, historical developments, and the evolution of key elements or breakthroughs within this field.
2. **Foundational Theories and Principles**
- Include a section on the fundamental theories and principles that form the basis of {TOPIC}, providing a foundation for a deeper exploration into specific areas or instances in later sections.
3. **Detailed Examination of Specific Instances**
- Significantly expand the section dedicated to the specific instances or case studies of {TOPIC}, covering a broad spectrum of applications, variations, or case studies that illustrate the depth and breadth of the field.
4. **Methodologies and Techniques**
- Delve into the diverse methodologies and techniques that are vital to researching and understanding {TOPIC}, maintaining an equitable focus on both theoretical and practical approaches.
5. **Integrative Discussion Section**
- Include a single section that integrates discussions on related sub-topics, such as technological, economic, and sustainability aspects, highlighting their interconnections and impacts on the main {TOPIC}.
6. **Conclusion and Future Directions**
- Conclude with a synthesis of insights from all sections, providing a comprehensive overview and identifying potential future research avenues in the realm of {TOPIC}.
7. **Output Format**:
- Present your outline in English as per the XML structure provided below. The outline should encompass all the aforementioned sections, each clearly identified and structured in a logical manner.
The expected XML structure is:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<Outlines>
	<Outline id="1">
		English outlines 1
	</Outline>
	<Outline id="2">
		English outlines 2
	</Outline>
	...
	<Outline id="n">
		English outlines n
	</Outline>
</Outlines>
```
Ensure that the outline serves as a rich source of information, seamlessly transitioning from theoretical underpinnings to a thorough exploration of {TOPIC}, and culminating in a cohesive conclusion that offers a vision for the future. The structure should be traditional yet detailed, emphasizing extensively on the key facets of {TOPIC}.
<file-attachment-contents filename='"""
    all_prompts = []
    if FromReview:
        for folder in Folders:
            all_prompts.append(construct_prompt(HEAD, END, HEADFromReview, ENDFromReview, FromReview, REPEAT, folder))
    else:
        all_prompts.append(
            construct_prompt(HEAD, END, HEADFromReview, ENDFromReview, FromReview, REPEAT, 'DirectGeneration')
        )
    all_prompts = [i for i in all_prompts if i]
    responses = GetResponse(all_prompts, REPEAT, Threads, FunctionClaudeAPI, FunctionOpenAIAPI, STDOUT)
    print('IntegrationQuestionsForReview')
    if not os.path.exists(f"AllQuestionsFromReview{os.sep}QuestionsFromReview_0.txt"):
        Folders = [
            i for i in os.listdir('.') if (i.startswith('10.') and not i.endswith('.txt')) or i == 'DirectGeneration'
        ]
        Folders.sort()
        os.makedirs('AllQuestionsFromReview', exist_ok=True)
        AllQuestionsFromReview = ''
        for folder in Folders:
            AllQuestionsFromReview += '\n\n\n'.join([
                open(f'{folder}{os.sep}QuestionsFromReview_{Time}.txt', 'r', encoding='UTF8').read()
                for Time in range(REPEAT)
            ])
        Prompt = IntegrationQuestionsForReview + 'AllQuestionsFromReview.txt>\n' + AllQuestionsFromReview + '\n</file-attachment-contents>'
        open(f'AllQuestionsFromReview{os.sep}PromptAllQuestionsFromReview.txt', 'w', encoding='UTF8').write(Prompt)
        REPEAT = 1
        responses = GetResponse([['AllQuestionsFromReview', Prompt]], REPEAT, Threads, FunctionClaudeAPI,
                                FunctionOpenAIAPI, STDOUT)
        with open(f'AllQuestionsFromReview{os.sep}QuestionsFromReview_0.txt', 'r', encoding='UTF8') as File:
            Outline = ReshapeOutlines(File.read())
        with open(f'AllQuestionsFromReview{os.sep}QuestionsFromReviewManual.txt', 'w', encoding='UTF8') as File:
            File.write(Outline)
    os.chdir('..')
    sys.stdout = old_stdout


def generate_detailed_outline(Folder, TOPIC, Threads, FunctionClaudeAPI, FunctionOpenAIAPI, STDOUT, MaxToken=25000):
    old_stdout = sys.stdout
    sys.stdout = STDOUT
    os.chdir(Folder)
    PaperQuestionsFromParagraphQuestionsForReview = f"""Utilizing the document provided, create a set of structured questions that correlate with the specific segments related to {TOPIC} as outlined in the literature. These questions are intended to extract precise information directly from the document to construct a comprehensive review article. It's essential to tailor each question to uncover the details specified in the segments of the document, ensuring an in-depth accumulation of data that aligns with the broader queries in the review's outline.
Construct each question by combining the main topic {TOPIC} with the pertinent questions from the document's individual subsections, then reorder them to present a logical sequence. For instance, if the document presents the following segment:
```
1. Background and Significance
   - What is spin and catalyst?
   - What is its industrial significance?
```
The corresponding questions tailored for information extraction should be formulated and ordered as:
```
1. Background and Significance - What are the core principles of spin and catalyst as described in the document?
2. Background and Significance - How has spin and catalyst impacted the industry according to the document?
```
Repeat this process for each segment, ensuring the questions are distinct and provide a clear framework for extracting detailed responses based on the document's specifics.
Your ultimate aim is to create a sequence of questions that will enable the gathering of rich, specific information from the document to develop a well-rounded perspective on each section related to {TOPIC}.
After drafting the detailed set of questions in English, translate them into Chinese, preserving the original depth and intention to support a bilingual information extraction process.
Output the complete English questions first, each clearly numbered. Follow this with the Chinese version, keeping each question separate and ensuring the two language sets are distinctly partitioned.
<file-attachment-contents filename='"""
    print('PaperQuestionsFromParagraphQuestionsForReview')
    shutil.copy(
        f'AllQuestionsFromReview{os.sep}QuestionsFromReviewManual.txt', f'..{os.sep}ParagraphQuestionsForReview.txt'
    )
    os.makedirs('PaperQuestionsFromParagraphQuestionsForReview', exist_ok=True)
    if not os.path.exists(f"PaperQuestionsFromParagraphQuestionsForReview{os.sep}QuestionsFromReview_0.txt"):
        ReviewOutline = open(f"AllQuestionsFromReview{os.sep}QuestionsFromReviewManual.txt", 'r',
                             encoding='UTF8').read()
        Prompt = PaperQuestionsFromParagraphQuestionsForReview + 'ReviewOutline.txt>\n' + ReviewOutline + '\n</file-attachment-contents>'
        open(
            f'PaperQuestionsFromParagraphQuestionsForReview{os.sep}PromptPaperQuestionsFromParagraphQuestionsForReview.txt',
            'w',
            encoding='UTF8'
        ).write(Prompt)
        REPEAT = 1
        Questions = ''
        while len([i for i in Questions.splitlines()
                   if i.strip()]) != len([i for i in ReviewOutline.splitlines() if i.strip()]):
            responses = GetResponse([['PaperQuestionsFromParagraphQuestionsForReview', Prompt]], REPEAT, Threads,
                                    FunctionClaudeAPI, FunctionOpenAIAPI, STDOUT)
            with open(
                f'PaperQuestionsFromParagraphQuestionsForReview{os.sep}QuestionsFromReview_0.txt', 'r', encoding='UTF8'
            ) as File:
                Questions = GetQuestions(File.read())
        with open(f'..{os.sep}QuestionsForReview.txt', 'w', encoding='UTF8') as File:
            File.write(Questions)
    os.chdir('..')
    sys.stdout = old_stdout

from common_utils import ReviewType, review_type_descriptions

def do_generate_overall_outline(query, TOPIC):
    output_file = f"outputs/{query}/overall_outline.txt"
    if os.path.exists(output_file):
        logger.info(f"Overall outline already exists for query {query}: {output_file}")
        return

    HEAD = f"""Your task is to craft a detailed and logical question-based outline for a comprehensive review article about {TOPIC}. The outline should first be presented entirely in English, followed by its translation in Chinese after a line break. The detailed instructions are as follows:
"""
    END = f'''
Ensure that your outline adheres to the following criteria:
1. Develop at least 17 primary themes that are intricately linked to {TOPIC}, paving the way for a structured review article.
2. Under each primary theme, identify at least 7 detailed questions that probe deep into the respective themes, fostering detailed and insightful paragraphs in the final article.
3. Integrate both experimental and computational methodologies, including aspects from DFT and other theoretical approaches, evenly and coherently in the questions to offer a rounded perspective on {TOPIC}.
4. The progression of the primary themes should mirror a traditional review trajectory: starting with the foundational concepts, transitioning to advanced nuances, discussing prevailing challenges, and culminating with future prospects and visions.
5. Formulate the questions such that when answered, they collectively offer a wide-ranging view of the {TOPIC} landscape, without any significant overlap or omission of vital aspects.
Remember that this outline should function as a structured pathway for researchers, guiding them to draft a holistic review article on {TOPIC} without overlooking any critical details.
Start with the complete English version of the outline. After finishing it, provide the Chinese translation, maintaining the depth and essence of the English version, separated by a line break.
'''
    prompt = HEAD + END

    llm_client = LLMClient()
    generated_text = llm_client.generate(prompt, system_prompt="你是一个帮助进行学术综述撰写的专家。")

    # print(generated_text)
    console.print(Markdown(generated_text))

    with open(output_file, "w", encoding='utf-8') as f:
        f.write(generated_text)

    logger.info(f"Overall outline saved to {output_file}")


def do_generate_detail_outline(query, TOPIC):
    detail_outlines_file = f"outputs/{query}/detailed_outline.txt"
    if os.path.exists(detail_outlines_file):
        logger.info(f"Detail outline already exists for query {query}: {detail_outlines_file}")
        return

    output_file = f"outputs/{query}/overall_outline.txt"
    with open(output_file, "r", encoding='utf-8') as f:
        global_generated_text = f.read()

    PaperQuestionsFromParagraphQuestionsForReview = f"""Utilizing the document provided, create a set of structured questions that correlate with the specific segments related to {TOPIC} as outlined in the literature. These questions are intended to extract precise information directly from the document to construct a comprehensive review article. It's essential to tailor each question to uncover the details specified in the segments of the document, ensuring an in-depth accumulation of data that aligns with the broader queries in the review's outline.
Construct each question by combining the main topic {TOPIC} with the pertinent questions from the document's individual subsections, then reorder them to present a logical sequence. For instance, if the document presents the following segment:
```
1. Background and Significance
   - What is spin and catalyst?
   - What is its industrial significance?
```
The corresponding questions tailored for information extraction should be formulated and ordered as:
```
1. Background and Significance - What are the core principles of spin and catalyst as described in the document?
2. Background and Significance - How has spin and catalyst impacted the industry according to the document?
```
Repeat this process for each segment, ensuring the questions are distinct and provide a clear framework for extracting detailed responses based on the document's specifics.
Your ultimate aim is to create a sequence of questions that will enable the gathering of rich, specific information from the document to develop a well-rounded perspective on each section related to {TOPIC}.
After drafting the detailed set of questions in English, translate them into Chinese, preserving the original depth and intention to support a bilingual information extraction process.
Output the complete English questions first, each clearly numbered. Follow this with the Chinese version, keeping each question separate and ensuring the two language sets are distinctly partitioned.
<file-attachment-contents filename='"""


    ReviewOutline = global_generated_text
    prompt = PaperQuestionsFromParagraphQuestionsForReview + 'ReviewOutline.txt>\n' + ReviewOutline + '\n</file-attachment-contents>'
    detail_outlines_prompt_file = f"outputs/{query}/detailed_outline_prompt.txt"
    with open(detail_outlines_prompt_file, "w", encoding='utf-8') as f:
        f.write(prompt)
    # os.makedirs('PaperQuestionsFromParagraphQuestionsForReview', exist_ok=True)
    # open(
    #     f'PaperQuestionsFromParagraphQuestionsForReview{os.sep}PromptPaperQuestionsFromParagraphQuestionsForReview.txt',
    #     'w', encoding='utf-8').write(prompt)

    llm_client = LLMClient()
    generated_text = llm_client.generate(prompt, system_prompt="你是一个帮助进行学术综述撰写的专家。")
    # print(f"Detail generated text: {generated_text}")
    console.print(Markdown(generated_text))
    # content = ""
    # if 'Outlines' in generated_text:
    #     content = '<?xml version="1.0" encoding="UTF-8"?>\n<Outlines>' + generated_text.split('<Outlines>',1)[1].rsplit('</Outlines>',1)[0] + '</Outlines>'
    #     try:
    #         root = ET.fromstring(content)
    #     except Exception as e:
    #         logger.warning(f"Error parsing XML content: {e}")
    # print(f"content: {content}")
    with open(detail_outlines_file, "w", encoding='utf-8') as f:
        f.write(generated_text)
    logger.info(f"Detail outline saved to {detail_outlines_file}")

def do_generate_detail_questions(query, TOPIC):
    detail_outlines_file = f"outputs/{query}/detailed_outline.txt"
    with open(detail_outlines_file, "r", encoding='utf-8') as f:
        detail_generated_text = f.read()

    questions_text = GetQuestions(detail_generated_text)

    detail_questions_file = f"outputs/{query}/detailed_questions.txt"
    with open(detail_questions_file, "w", encoding='utf-8') as f:
        f.write(questions_text)
    logger.info(f"Detail questions saved to {detail_questions_file}")

def get_query_and_topic():
    query = "损失函数"
    review_type = "concept"

    # query = "Text2SQL研究现状如何，面临哪些挑战？"
    # review_type = "status"

    # query = "有哪些方法可以提升大模型的规划能力，各自优劣是什么？"
    # review_type = "comparison"

    # query = "多模态大模型的技术发展路线是什么样的？"
    # review_type = "timeline"

    review_type = ReviewType(review_type)
    review_desc = review_type_descriptions[review_type]


    if review_type == ReviewType.CONCEPT:
        TOPIC = f"关于“{query}“的{review_desc}"
    elif review_type == ReviewType.STATUS:
        TOPIC = f"关于“{query}“的{review_desc}"
    elif review_type == ReviewType.COMPARISON:
        TOPIC = f"关于“{query}“的{review_desc}"
    elif review_type == ReviewType.TIMELINE:
        TOPIC = f"关于“{query}“的{review_desc}"
    else:
        raise ValueError(f"Invalid review type {review_type}")

    return query, TOPIC
    
if __name__ == '__main__':
    query, TOPIC = get_query_and_topic()
    do_generate_overall_outline(query, TOPIC)
    do_generate_detail_outline(query, TOPIC)
    do_generate_detail_questions(query, TOPIC)