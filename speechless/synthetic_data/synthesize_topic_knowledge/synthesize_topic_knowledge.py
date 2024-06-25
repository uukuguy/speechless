#!/usr/bin/env python
import os, sys, json, re
from tqdm import tqdm
SPEECHLESS_ROOT = os.getenv("SPEECHLESS_ROOT")
if SPEECHLESS_ROOT is not None:
    sys.path.insert(0, SPEECHLESS_ROOT)
from speechless.generate.llm_api import get_llm_api

system_prompt = "你是一名电网调度控制领域的专家，你需要用你的专业知识回答用户的问题。要求回答专业，准确，条理清楚。"

api_keys = {
    'w1': "3a4b3b8f9e2c548710adfcfc472d4602.d2xaIekJuPs6Yzmz",
    'w2': "624e9ce65bdb2f447c0b5e8af7783e9a.C4ZQX44QViOFdNSN",
    'w3': "a2d3af415bda175906422c10c195348e.1Jc0EIKefwEZbnlQ",
    'w4': "75263bb470384e13a1ffa46705aaeeb4.L1C0skaLuZGpA12C",
    'w5': "d5909b06d181bb1939efaf89a5f4285b.sjZR1Qw6RoMk8BN6",
    'w6': "ea2d92d1e3ccb2ebf99850e59d41e5ae.ePgTUaSQfIA5IpVG",
}

def prepare_api_key(args):
    assert args.key_id in api_keys, f"Invalid key_id: {args.key_id}"
    os.setenv("ZHIPUAI_API_KEY", api_keys[args.key_id])
    
def do_sub_topics(args):
    main_topics = [ json.loads(line.strip())['topic'] for line in open(args.main_topics_file, "r").readlines()]

    llm_api = get_llm_api(LLM_API=args.llm_api, model=args.model)

    generate_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # "stop": ["\n"],
    }
    sub_topics_prompt_template = "生成10个全面的子主题，涵盖“{}”主题下的各个方面，尽可能使主题多样化。输出JSON格式的子主题列表，不要输出其它解释说明："

    with open(args.output_file, "w") as fd:
        for m_topic in tqdm(main_topics, ncols=100, desc="Main topics"):
            print(f"----- {m_topic} -----")
            instruction = sub_topics_prompt_template.format(m_topic)

            
            generated_text = llm_api.generate(instruction, 
                                            generate_args=generate_args, 
                                            system_prompt=system_prompt, 
                                            verbose=args.verbose)
            print(generated_text)

            line = json.dumps({"main_topic": m_topic, "sub_topics": generated_text}, ensure_ascii=False)
            fd.write(line + "\n")
            fd.flush()

def do_clean_sub_topics(args):
    cleaned_sub_topics_list = []
    lines = open(args.input_file, "r").readlines()
    for line in tqdm(lines, ncols=100):
        data = json.loads(line.strip())
        main_topic = data['main_topic']
        sub_topics_str = data['sub_topics']
        sub_topics_str = sub_topics_str.replace("```json\n", "").replace("```", "")

        new_sub_topics = []

        candidate_keys = ['title', 'name', 'subtopic', 'topic']
        sub_topics_lines = sub_topics_str.split("\n")
        for s_line in sub_topics_lines:
            s_line = s_line.strip()
            key = None
            for ck in candidate_keys:
                if f"\"{ck}\": " in s_line:
                    key = ck
                    break
            if key:
                sub_topic = s_line.split(f"\"{key}\": ")[-1]
                sub_topic_list = re.findall(r'\"(.*?)\"', sub_topic)
                if len(sub_topic_list) > 0:
                    sub_topic = sub_topic_list[0]
                    sub_topic = re.sub(r'^\d+\. ', "", sub_topic)
                    new_sub_topics.append({'sub_topic': sub_topic})

        cleaned_sub_topics_list.append({'main_topic': main_topic, 'sub_topics': new_sub_topics})

    with open(args.output_file, "w") as fd:
        for cleaned_sub_topics in cleaned_sub_topics_list:
            line = json.dumps(cleaned_sub_topics, ensure_ascii=False)
            fd.write(line + "\n")
            fd.flush()                

# 生成基于主题的开放式问答问题
def do_topic_questions(args):
    llm_api = get_llm_api(args.llm_api, args.model)

    generate_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # "stop": ["\n"],
    }

    sub_topics_list = [ json.loads(line.strip()) for line in open(args.input_file).readlines()]


    with open(args.output_file, "w") as fd:
        for i, d in enumerate(tqdm(sub_topics_list, ncols=100)):
            main_topic = d['main_topic']
            sub_topics = d['sub_topics']

            for j, sub_topic in enumerate(sub_topics):
                sub_topic = sub_topic['sub_topic']
                print(f"----- {i}. {main_topic} / {j}. {sub_topic} -----")
                # instruction = f"生成10个关于“{sub_topic}”的问题，问题要求专业，准确，条理清楚。输出JSON格式的问题列表，不要输出其它解释说明："
                instruction = f"生成10个与{sub_topic}主题相关的问题或请求，这些问题和请求应尽可能多样化。只返回JSON格式的列表，生成的问题和请求列表如下："

                generated_text = llm_api.generate(instruction, 
                                                generate_args=generate_args, 
                                                system_prompt=system_prompt, 
                                                verbose=args.verbose)
                # print(generated_text)

                line = json.dumps({"main_topic": main_topic, "sub_topic": sub_topic, "questions": generated_text}, ensure_ascii=False)
                fd.write(line + "\n")
                fd.flush()

def do_clean_topic_questions(args):
    cleaned_topic_questions_list = []
    lines = open(args.input_file, "r").readlines()
    for line in tqdm(lines, ncols=100):
        data = json.loads(line.strip())
        main_topic = data['main_topic']
        sub_topic = data['sub_topic']
        
        topic_questions_str = data['questions']
        topic_questions_str = topic_questions_str.replace("```json\n", "").replace("```", "")

        new_topic_questions = []

        candidate_keys = ['question', 'request']
        topic_questions_lines = topic_questions_str.split("\n")
        for s_line in topic_questions_lines:
            s_line = s_line.strip()
            key = None
            for ck in candidate_keys:
                if f"\"{ck}\": " in s_line:
                    key = ck
                    break
            if key:
                question = s_line.split(f"\"{key}\": ")[-1]
                question_list = re.findall(r'\"(.*?)\"', question)
                if len(question_list) > 0:
                    question = question_list[0]
                    question = re.sub(r'^\d+\. ', "", question)
                    new_topic_questions.append({ck: question})

        cleaned_topic_questions_list.append({'main_topic': main_topic, 'sub_topic': sub_topic, 'questions': new_topic_questions})

    with open(args.output_file, "w") as fd:
        for cleaned_topic_questions in cleaned_topic_questions_list:
            line = json.dumps(cleaned_topic_questions, ensure_ascii=False)
            fd.write(line + "\n")
            fd.flush()                

def do_revise_questions(args):

    llm_api = get_llm_api(args.llm_api, args.model)

    generate_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # "stop": ["\n"],
    }

    prompt_template = """已知{sub_topic}主题下的问题：{question}

您能否修改上述问题，以包含更多的背景或细节？修订后的问题可以是以下任何一种：

1. 在原始问题中添加一些背景信息。这些背景可能说明了该问题的重要性、解释了相关知识或添加其他合理的信息。

2. 将问题改写成不同的格式或风格，例如命令语句、回答长度要求等。

3. 需要详细阐述特定主题或讨论某个观点的延长型问题。

4. 任何其他相关的问题或陈述。

修订后的问题应包含两到四个句子。请在JSON列表中生成{n_questions}个尽可能多样化的修订后问句或陈述。
"""
    n_questions = 10
    sub_topics_list = [ json.loads(line.strip()) for line in open(args.input_file).readlines()]
    with open(args.output_file, "w") as fd:
        for i, d in enumerate(tqdm(sub_topics_list, ncols=100)):
            main_topic = d['main_topic']
            sub_topics = d['sub_topics']

            for j, s_topic in enumerate(sub_topics):
                sub_topic = s_topic['sub_topic']
                print(f"----- {i}. {main_topic} / {j}. {sub_topic} -----")
                questions = s_topic['questions']
                for k, q in enumerate(questions):
                    question = q['question']
                    print(f"{question}")
                    instruction = prompt_template.format(sub_topic=sub_topic, question=question, n_questions=n_questions)

                    generated_text = llm_api.generate(instruction, 
                                                    generate_args=generate_args, 
                                                    system_prompt=system_prompt, 
                                                    verbose=args.verbose)
                    # print(generated_text)
                    q['revised_questions'] = generated_text

                line = json.dumps({"main_topic": main_topic, "sub_topic": sub_topic, "questions": questions}, ensure_ascii=False)
                fd.write(line + "\n")
                fd.flush()

def do_answer_questions(args):
    prepare_api_key(args)
    llm_api = get_llm_api(args.llm_api, args.model)

    generate_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        # "stop": ["\n"],
    }

    prompt_template = """用户的问题属于{sub_topic}主题下的问题，请给出分章节的教科书级别的回答。

- 提供正确回答问题所必需的专业背景知识。
- 分步骤解释回答的逻辑推理过程。
- 用专业术语和术语解释回答。
- 需要计算的问题，请给出详细的计算公式解释和计算过程。
- 可能的情况下，举例说明回答的适用性和实际应用。
- 问题业务领域相关的其他高度相关的信息。
- 回答长度控制在500至2000字。

问题：{question}\n回答："""

    sub_topics_list = [ json.loads(line.strip()) for line in open(args.input_file).readlines()]
    with open(args.output_file, "w") as fd:
        print(f"Answer questions from {args.start_idx} to {args.end_idx}")
        for x, d in enumerate(tqdm(sub_topics_list[args.start_idx:args.end_idx], ncols=100)):
            i = x + args.start_idx
            if i < args.start_idx:
                continue
            if i >= args.end_idx:
                break
            main_topic = d['main_topic']
            sub_topic = d['sub_topic']
            questions = d['questions']

            for j, question in enumerate(questions):
                print(f"----- {i}. {main_topic} / {sub_topic}: {j}. {question} -----")
                instruction = prompt_template.format(sub_topic, question=question)

                generated_text = llm_api.generate(instruction, 
                                                generate_args=generate_args, 
                                                system_prompt=system_prompt, 
                                                verbose=args.verbose)
                # print(generated_text)
                q['answer'] = generated_text

            line = json.dumps({"main_topic": main_topic, "sub_topic": sub_topic, "questions": questions}, ensure_ascii=False)
            fd.write(line + "\n")
            fd.flush()

            
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_sub_topics", action="store_true", help="Do sub topics")
    parser.add_argument("--do_clean_sub_topics", action="store_true", help="Do clean sub topics")
    parser.add_argument("--do_topic_questions", action="store_true", help="Do topic questions" )
    parser.add_argument("--do_clean_topic_questions", action="store_true", help="Do clean topic questions" )
    parser.add_argument("--do_revise_questions", action="store_true", help="Do revise questions" )
    parser.add_argument("--do_answer_questions", action="store_true", help="Do answer questions" )
    parser.add_argument("--start_idx", type=int, default=0, help="Start index")
    parser.add_argument("--end_idx", type=int, default=500, help="End index")
    parser.add_argument("--main_topics_file", type=str, default="gdc-most-common-topics.jsonl", help="Main topics file")
    parser.add_argument("--input_file", type=str, help="Input file")
    parser.add_argument("--output_file", type=str, help="Output file")
    parser.add_argument("--key_id", type=str)

    parser.add_argument("--question", type=str, help="Question")
    parser.add_argument("--llm_api", type=str, default="ZhipuAI", choices=['ZhipuAI', 'OpenAI', 'DashScope'], help="LLM API")
    parser.add_argument("--model", type=str, default=None, help="Model")
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument("--max_tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.95, help="Temperature")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top p")
    return parser.parse_args()

            
def main(args):
    llm_api = get_llm_api(args.llm_api, args.model)

    if args.do_sub_topics:
        do_sub_topics(args)        
    elif args.do_clean_sub_topics:
        do_clean_sub_topics(args)
    elif args.do_topic_questions:
        do_topic_questions(args)
    elif args.do_clean_topic_questions:
        do_clean_topic_questions(args)
    elif args.do_revise_questions:
        do_revise_questions(args)
    elif args.do_answer_questions:
        do_answer_questions(args)


if __name__ == "__main__":
    args = get_args()
    main(args)
