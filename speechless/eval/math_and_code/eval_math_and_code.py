# Derived from yule-BUAA/MergeLM/blob/main/inference_llms_instruct_math_code.py
import os, sys, json, re
import jsonlines
import glob 
import torch
import logging
import shutil

# for gsm8k and hendrycks_math
def get_math_task_prompt():
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    return problem_prompt

# for human_eval and mbpp
def generate_code_task_prompt(input_text):
    INSTRUCTION = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.


### Instruction:
Create a Python script for this problem:
{input_text}

### Response:"""
    return INSTRUCTION

# -------------------- test_gsm8k --------------------

def gsm8k_extract_answer_number(completion):
    text = completion.split("The answer is: ")
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r"[\-+]?\d*[\.,/]?\d+", extract_ans)
        if match:
            if "/" in match.group():
                denominator = match.group().split("/")[1]
                numerator = match.group().split("/")[0]
                if is_number(denominator) == True and is_number(numerator) == True:
                    if denominator == "0":
                        return round(float(numerator.replace(",", "")))
                    else:
                        frac = Fraction(match.group().replace(",", ""))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(",", "")) == float("inf"):
                    return None
                return round(float(match.group().replace(",", "")))
        else:
            return None
    else:
        return None


def test_gsm8k(
    llm,
    test_data_path,
    args,
    logger: logging.Logger,
    start_index=0,
    end_index=sys.maxsize,
):
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = get_math_task_prompt()
    logger.info(f"gsm8k test prompt is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item["answer"].split("#### ")[1]
            temp_ans = int(temp_ans.replace(",", ""))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start_index:end_index]
    gsm8k_answers = gsm8k_answers[start_index:end_index]
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=60)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1, max_tokens=1024, stop=stop_tokens
    )
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_gsm8k_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(
        zip(gsm8k_ins, res_completions, gsm8k_answers)
    ):
        y_pred = gsm8k_extract_answer_number(completion)
        if y_pred != None:
            results.append(float(y_pred) == float(prompt_answer))
        else:
            results.append(False)
            temp = {"question": prompt, "output": completion, "answer": prompt_answer}
            invalid_outputs.append(temp)
    accuracy = sum(results) / len(results)
    logger.info(
        f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}"
    )
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"gsm8k test data length is {len(results)}, accuracy is {accuracy}")
    logger.info(args)


# -------------------- test_hendrycks_math --------------------

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval

def math_process_results(doc, completion, answer, invalid_outputs):
    split_ans = completion.split('The answer is: ')
    if len(split_ans) > 1:
        ans = split_ans[-1]
        extract_ans_temp = ans.split('.\n')[0]
        extract_ans_temp = extract_ans_temp.strip()
        if len(extract_ans_temp) > 0 and extract_ans_temp[-1] == '.':
            extract_ans = extract_ans_temp[0:-1]
        else:
            extract_ans = extract_ans_temp
        extract_ans = extract_ans.strip()
        if is_equiv(extract_ans, answer):
            return True
        else:
            return False
    else:
        temp = {'question': doc, 'output': completion, 'answer': answer}
        invalid_outputs.append(temp)
        return False

def test_hendrycks_math(
    llm,
    test_data_path,
    args,
    logger: logging.Logger,
    start_index=0,
    end_index=sys.maxsize,
):
    hendrycks_math_ins = []
    hendrycks_math_answers = []
    problem_prompt = get_math_task_prompt()
    logger.info(f"MATH test prompt is {problem_prompt}")
    with open(test_data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["instruction"])
            hendrycks_math_ins.append(temp_instr)
            solution = item["output"]
            temp_ans = remove_boxed(last_boxed_only_string(solution))
            hendrycks_math_answers.append(temp_ans)

    hendrycks_math_ins = hendrycks_math_ins[start_index:end_index]
    hendrycks_math_answers = hendrycks_math_answers[start_index:end_index]
    batch_hendrycks_math_ins = batch_data(hendrycks_math_ins, batch_size=50)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(
        temperature=0.0, top_p=1, max_tokens=2048, stop=stop_tokens
    )
    logger.info(f"sampling params is {sampling_params}")

    res_completions = []
    for idx, prompt in enumerate(batch_hendrycks_math_ins):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]
        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    results = []
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(
        zip(hendrycks_math_ins, res_completions, hendrycks_math_answers)
    ):
        res = math_process_results(prompt, completion, prompt_answer, invalid_outputs)
        results.append(res)
    accuracy = sum(results) / len(results)
    logger.info(
        f"invalid outputs length is {len(invalid_outputs)}, invalid_outputs are {invalid_outputs}"
    )
    logger.info(f"data index starts from {start_index}, ends at {end_index}")
    logger.info(f"MATH test data length is {len(results)}, accuracy is {accuracy}")
    logger.info(args)


# -------------------- test_human_eval --------------------

def test_human_eval(
    llm,
    args,
    logger: logging.Logger,
    start_index=0,
    end_index=sys.maxsize,
    save_gen_results_folder=None,
):
    problems = read_problems()
    task_ids = sorted(problems.keys())[start_index:end_index]
    prompts = [problems[task_id]["prompt"] for task_id in task_ids]
    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

        prompt = prompts[i].replace("    ", "\t")
        prompt_batch = [generate_code_task_prompt(prompt)]

        ids_batch = [task_ids[i]]
        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace("\t", "    ")
                    all_code = gen_seq.replace("\t", "    ")

                    completion_seqs.append(
                        {
                            "task_id": task_id,
                            "completion": completion_seq,
                            "all_code": all_code,
                        }
                    )

        write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    outputs = []
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            completion = code["completion"]
            completion = completion.replace("\r", "")
            completion = completion.strip()
            if "```python" in completion:
                logger.info("completion matches ```python")
                def_line = completion.index("```python")
                completion = completion[def_line:].strip()
                completion = completion.replace("```python", "")
                try:
                    next_line = completion.index("```")
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if '__name__ == "__main__"' in completion:
                logger.info('completion matches __name__ == "__main__"')
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "# Example usage" in completion:
                logger.info("completion matches # Example usage")
                next_line = completion.index("# Example usage")
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace("The solution is:", "")
                try:
                    next_line = completion.index("\n\nThe answer is:")
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            if "The answer is:" in completion:
                logger.info("completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace("The answer is:", "")
                try:
                    next_line = completion.index("\n\nThe answer is:")
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            code["completion"] = completion
        outputs += codes

    logger.info(f"save to {save_gen_results_folder}.jsonl")
    write_jsonl(f"{save_gen_results_folder}.jsonl", outputs)


# -------------------- test_mbpp --------------------
def read_mbpp(path):
    mbpp_problems = {}
    with jsonlines.open(path, "r") as fin:
        for obj in fin:
            mbpp_problems[obj["task_id"]] = obj
    return mbpp_problems


def test_mbpp(
    llm,
    test_data_path,
    args,
    logger: logging.Logger,
    start_index=0,
    end_index=sys.maxsize,
    save_gen_results_folder=None,
):
    problems = read_mbpp(test_data_path)
    task_ids = sorted(problems.keys())[start_index:end_index]
    prompts = []
    for task_id in task_ids:
        prompt = f"\n{problems[task_id]['text']}\nTest examples:"
        if task_id == 493:
            # The test examples are too long, we choose to only include the function name.
            test_example = problems[task_id]["test_list"][0]
            prompt += f"\ncalculate_polygons(startx, starty, endx, endy, radius)"
        else:
            for test_example in problems[task_id]["test_list"]:
                prompt += f"\n{test_example}"
        prompts.append(prompt)

    num_samples = len(prompts)
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=2048)

    shutil.rmtree(save_gen_results_folder, ignore_errors=True)
    os.makedirs(save_gen_results_folder, exist_ok=True)

    for i in tqdm(range(num_samples), ncols=0, total=num_samples):
        output_file = f"{save_gen_results_folder}/{args.start_index + i}.jsonl"

        prompt = prompts[i].replace("    ", "\t")
        prompt_batch = [generate_code_task_prompt(prompt)]

        ids_batch = [task_ids[i]]
        completion_seqs = []

        loops = 1

        for _ in tqdm(range(loops), total=loops, leave=False, ncols=0):
            with torch.no_grad():
                completions = llm.generate(prompt_batch, sampling_params)
            gen_seqs = [completions[0].outputs[0].text]

            if gen_seqs is not None:
                assert len(ids_batch) == 1
                task_id = ids_batch[0]

                for seq_idx, gen_seq in enumerate(gen_seqs):
                    completion_seq = gen_seq.split("### Response:")[-1]
                    completion_seq = completion_seq.replace("\t", "    ")
                    all_code = gen_seq.replace("\t", "    ")

                    completion_seqs.append(
                        {
                            "task_id": task_id,
                            "completion": completion_seq,
                            "all_code": all_code,
                        }
                    )

        write_jsonl(output_file, completion_seqs)

    files = sorted(glob.glob(f"{save_gen_results_folder}/*.jsonl"))
    logger.info(f"find {len(files)} files in {save_gen_results_folder}")

    problems = read_mbpp(test_data_path)
    outputs = [[] for _ in range(len(problems))]
    for code_file in tqdm(files, total=len(files)):
        codes = [c for c in stream_jsonl(code_file)]
        for code in codes:
            task_id = code["task_id"]
            completion = code["completion"]
            completion = completion.strip()
            if "```python" in completion:
                logger.info("completion matches ```python")
                def_line = completion.index("```python")
                completion = completion[def_line:].strip()
                completion = completion.replace("```python", "")
                try:
                    next_line = completion.index("\n```")
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if '__name__ == "__main__"' in completion:
                logger.info('completion matches __name__ == "__main__"')
                try:
                    next_line = completion.index('if __name__ == "__main__":')
                    completion = completion[:next_line].strip()
                except:
                    logger.info("wrong completion")
            if "# Example usage" in completion:
                logger.info("completion matches # Example usage")
                next_line = completion.index("# Example usage")
                completion = completion[:next_line].strip()
            if "# Test examples" in completion:
                logger.info("completion matches # Test examples")
                next_line = completion.index("# Test examples")
                completion = completion[:next_line].strip()
            # the following codes are used to deal with the outputs of code-alpaca
            if "The solution is:" in completion:
                logger.info("completion matches The solution is:")
                def_line = completion.index("The solution is:")
                completion = completion[def_line:].strip()
                completion = completion.replace("The solution is:", "")
                try:
                    next_line = completion.index("\n\nThe answer is:")
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            if "The answer is:" in completion:
                logger.info("completion matches The answer is:")
                def_line = completion.index("The answer is:")
                completion = completion[def_line:].strip()
                completion = completion.replace("The answer is:", "")
                try:
                    next_line = completion.index("\n\nThe answer is:")
                    completion = completion[:next_line].strip()
                except:
                    completion = completion.strip()
                    logger.info("maybe wrong completion")
            outputs[task_id - 11].append(completion)

    logger.info(f"save to {save_gen_results_folder}.jsonl")
    with open(f"{save_gen_results_folder}.jsonl", "w", encoding="utf-8") as fout:
        json.dump(outputs, fout)
