import os, sys, json, argparse
import collections, itertools, random
from tqdm import tqdm
import numpy as np
import transformers
from datetime import datetime
from loguru import logger

import gc, ctypes
import torch

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, f"{script_path}/lm-evaluation-harness")
from lm_eval import tasks
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.tasks import get_task_dict
from lm_eval.utils import positional_deprecated, run_task_tests
from lm_eval.models.gpt2 import HFLM

import fnmatch


def _is_json_task(task_name):
    return task_name == "json" or task_name.startswith("json=")


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        if _is_json_task(pattern):
            task_names.add(pattern)

        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return sorted(list(task_names))


# https://www.kaggle.com/code/simjeg/platypus2-70b-without-wikipedia-rag
# Function to clean RAM & vRAM
def clean_memory():
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    torch.cuda.empty_cache()


decontaminate_suffix = "_decontaminate"


def resp_to_results(
    process_res_queue, task_dict, docs, decontaminate, overlaps, bootstrap_iters=100000, write_out_info=None
):
    if write_out_info is None:
        write_out = False
    else:
        write_out = True

    results = collections.defaultdict(dict)
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, requests)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

            if write_out:
                write_out_info[task_name][doc_id][metric] = str(value)

            # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
            if decontaminate and task_name in overlaps:
                if doc_id not in overlaps[task_name]:
                    vals[(task_name, metric + decontaminate_suffix)].append(value)

    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        real_metric = metric  # key when looking up the metric with task.aggregation
        if metric.endswith(decontaminate_suffix):
            real_metric = metric.replace(decontaminate_suffix, "")  # decontaminated still uses the same metric
        results[task_name][metric] = task.aggregation()[real_metric](items)

        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this

        stderr = lm_eval.metrics.stderr_for_metric(
            metric=task.aggregation()[real_metric],
            bootstrap_iters=min(bootstrap_iters, 1000) if metric in ["bleu", "chrf", "ter"] else bootstrap_iters,
        )

        if stderr is not None:
            results[task_name][metric + "_stderr"] = stderr(items)

    return results


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    provide_description=None,
    num_fewshot=0,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
    write_out=False,
    output_path="eval_results/lm_eval",
    output_base_path=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param write_out: bool
        If True, write all prompts, logits and metrics to json for offline analysis
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system

    os.makedirs(f"{output_path}/details", exist_ok=True)
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print(
            "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
        )

    exist_tasks = []

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [(name, task)
                       for name, task in task_dict.items()
                       if (task.has_validation_docs() or task.has_test_docs())]

    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}
    write_out_info = {}

    docs_for_decontamination = collections.defaultdict(list)

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    print(f"evaluating {len(task_dict_items)} tasks")
    # get lists of each type of request
    for task_name, task in tqdm(task_dict_items, ncols=100, desc="Task"):
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        task_docs = list(task_doc_func())
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        print(f"Task: {task_name}; number of docs: {len(task_docs)}")

        # FIXME
        task_write_out_file = f"{output_path}/details/{task_name}_write_out_info.json"
        if os.path.exists(task_write_out_file):
            exist_tasks.append(task_name)
            task_json_data = json.load(open(task_write_out_file))
            task_json_data = { x['doc_id']: x for x in task_json_data}
            write_out_info[task_name] = []
            for doc_id, doc in tqdm(enumerate(itertools.islice(task_docs, 0, limit)), ncols=100, desc=f"Doc({task_name})"):
                docs[(task_name, doc_id)] = doc
                if write_out:
                    write_out_info[task_name].append({"doc_id": doc_id})

                if decontaminate and task.should_decontaminate():
                    docs_for_decontamination[(task_name, task_set)].append(task.doc_to_decontamination_query(doc))
                doc_json_data = task_json_data[doc_id]
                for key in doc_json_data:
                    if key.startswith('logit_'):
                        i = int(key[len('logit_'):])
                        resp = doc_json_data[key]
                        if task_name == "drop":
                            resp = "".join(resp.split("\n")[:1])
                        process_res_queue[(task_name, doc_id)].append((i, resp))

                        if write_out:

                            write_out_info[task_name][doc_id][f"logit_{i}"] = resp
                            task = task_dict[task_name]
                            if isinstance(task, lm_eval.base.MultipleChoiceTask):
                                write_out_info[task_name][doc_id]["truth"] = doc["gold"]
                            elif isinstance(task, lm_eval.tasks.winogrande.Winogrande):
                                write_out_info[task_name][doc_id]["truth"] = task.answer_to_num[doc["answer"]]
                            else:
                                write_out_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)
                    elif key.startswith('prompt_'):
                        i = int(key[len('prompt_'):])
                        write_out_info[task_name][doc_id][f"prompt_{i}"] = doc_json_data[key]

                # if write_out:
                #     write_out_info[task_name] = prompt_details
        else:

            if write_out:
                prompt_details = []

            description = (description_dict[task_name] if description_dict and task_name in description_dict else "")
            if limit is not None:
                limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

            for doc_id, doc in tqdm(enumerate(itertools.islice(task_docs, 0, limit)), ncols=100, desc=f"Doc({task_name})"):
                if decontaminate and task.should_decontaminate():
                    docs_for_decontamination[(task_name, task_set)].append(task.doc_to_decontamination_query(doc))

                docs[(task_name, doc_id)] = doc
                ctx = task.fewshot_context(doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description)
                reqs = task.construct_requests(doc, ctx)

                if write_out:
                    prompt_details.append({"doc_id": doc_id})

                # print the prompt for the first few documents
                # if doc_id < 1:
                #     print(
                #         f"Task: {task_name}; document {doc_id}; context prompt (starting on next line):\n{ctx}\n(end of prompt on previous line)"
                #     )
                #     print("Requests:", reqs)

                if not isinstance(reqs, (list, tuple)):
                    reqs = [reqs]
                for i, req in enumerate(reqs):
                    requests[req.request_type].append(req)
                    # i: index in requests for a single task instance
                    # doc_id: unique id that we can get back to a doc using `docs`
                    requests_origin[req.request_type].append((i, task_name, doc, doc_id))

                    if write_out:
                        prompt_details[-1][f"prompt_{i}"] = "".join((map(lambda x: "".join(x), req.args)))

            if write_out:
                write_out_info[task_name] = prompt_details

    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(docs_for_decontamination, decontamination_ngrams_path, limit)

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        resps = [x if req.index is None else x[req.index] for x, req in zip(resps, reqs)]

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            if task_name == "drop":
                resp = "".join(resp.split("\n")[:1])
            process_res_queue[(task_name, doc_id)].append((i, resp))

            if write_out:
                write_out_info[task_name][doc_id][f"logit_{i}"] = resp
                task = task_dict[task_name]
                if isinstance(task, lm_eval.base.MultipleChoiceTask):
                    write_out_info[task_name][doc_id]["truth"] = doc["gold"]
                elif isinstance(task, lm_eval.tasks.winogrande.Winogrande):
                    write_out_info[task_name][doc_id]["truth"] = task.answer_to_num[doc["answer"]]
                else:
                    write_out_info[task_name][doc_id]["truth"] = task.doc_to_target(doc)
        clean_memory()

    # vals = collections.defaultdict(list)

    # # unpack results and sort back in order and return control to Task
    # for (task_name, doc_id), requests in process_res_queue.items():
    #     requests.sort(key=lambda x: x[0])
    #     requests = [x[1] for x in requests]

    #     task = task_dict[task_name]
    #     doc = docs[(task_name, doc_id)]

    #     metrics = task.process_results(doc, requests)
    #     for metric, value in metrics.items():
    #         vals[(task_name, metric)].append(value)

    #         if write_out:
    #             write_out_info[task_name][doc_id][metric] = str(value)

    #         # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
    #         if decontaminate and task_name in overlaps:
    #             if doc_id not in overlaps[task_name]:
    #                 vals[(task_name, metric + decontaminate_suffix)].append(value)

    # # aggregate results
    # for (task_name, metric), items in vals.items():
    #     task = task_dict[task_name]
    #     real_metric = metric  # key when looking up the metric with task.aggregation
    #     if metric.endswith(decontaminate_suffix):
    #         real_metric = metric.replace(
    #             decontaminate_suffix, ""
    #         )  # decontaminated still uses the same metric
    #     results[task_name][metric] = task.aggregation()[real_metric](items)

    #     # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
    #     # so we run them less iterations. still looking for a cleaner way to do this

    #     stderr = lm_eval.metrics.stderr_for_metric(
    #         metric=task.aggregation()[real_metric],
    #         bootstrap_iters=min(bootstrap_iters, 1000)
    #         if metric in ["bleu", "chrf", "ter"]
    #         else bootstrap_iters,
    #     )

    #     if stderr is not None:
    #         results[task_name][metric + "_stderr"] = stderr(items)
    results = resp_to_results(
        process_res_queue=process_res_queue,
        task_dict=task_dict,
        docs=docs,
        decontaminate=decontaminate,
        overlaps=overlaps,
        bootstrap_iters=bootstrap_iters,
        write_out_info=write_out_info
    )

    if write_out:

        # output_base_path = (
        #     pathlib.Path(output_base_path)
        #     if output_base_path is not None
        #     else pathlib.Path(".")
        # )
        # try:
        #     output_base_path.mkdir(parents=True, exist_ok=False)
        # except FileExistsError:
        #     pass

        os.makedirs(f"{output_path}/details", exist_ok=True)
        for task_name, _ in task_dict_items:
            if task_name not in exist_tasks:
                with open(
                    # output_base_path.joinpath(f"{output_path}/details/{task_name}_write_out_info.json"),
                    f"{output_path}/details/{task_name}_write_out_info.json",
                    "w",
                    encoding="utf8",
                ) as fp:
                    json.dump(write_out_info[task_name], fp, indent=4, ensure_ascii=False)

    return {
        "results": dict(results),
        "versions": dict(versions)
    }


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


"""
acc = 0.0
n = 0
for d in json_data:
if 'results' in d:
    results = d['results']
    for k, v in results.items():
        if k.startswith('hendrycksTest'):
        acc += results[k]['acc']
        n += 1

number of docs:
arc_challenge: 1172
hellaswag: 10042
mmlu: hendrycksTest-* 57 tasks, 100 - 1534
gsm8k: 1319
"""


def summarize_results(json_data):
    arc_acc_norm = 0.0
    hellaswag_acc_norm = 0.0
    mmlu_acc = 0.0
    mmlu_n = 0
    truthfullqa_mc2 = 0.0
    gsm8k_acc = 0.0
    winoground_acc = 0.0
    drop_f1 = 0.0
    for d in json_data:
        if 'results' in d:
            results = d['results']
            for k, v in results.items():
                if k.startswith('hendrycksTest'):
                    mmlu_acc += results[k]['acc']
                    mmlu_n += 1
                elif k == 'arc_challenge':
                    arc_acc_norm = results[k]['acc_norm']
                elif k == 'hellaswag':
                    hellaswag_acc_norm = results[k]['acc_norm']
                elif k == 'truthfulqa_mc':
                    truthfullqa_mc2 = results[k]['mc2']
                elif k == 'winogrande':
                    winoground_acc = results[k]['acc']
                elif k == 'gsm8k':
                    gsm8k_acc = results[k]['acc']
                elif k == 'drop':
                    drop_f1 = results[k]['f1']

    mmlu_acc /= (mmlu_n + 1e-12)

    open_llm_score = (arc_acc_norm + hellaswag_acc_norm + mmlu_acc + truthfullqa_mc2 + winoground_acc + gsm8k_acc + drop_f1) / 7

    summary = {
        'ARC (acc_norm)': arc_acc_norm,
        'HellaSwag (acc_norm)': hellaswag_acc_norm,
        'MMLU (acc)': mmlu_acc,
        'TruthfulQA (mc2)': truthfullqa_mc2,
        'Winoground (acc)': winoground_acc,
        'GSM8K (acc)': gsm8k_acc,
        'DROP (f1)': drop_f1,
        'Open LLM Score': open_llm_score,
    }

    return summary


def do_lmeval(
    model,
    model_args=None,
    tasks=[],
    batch_size=None,
    max_batch_size=None,
    device=None,
    no_cache=False,
    bootstrap_iters=1000,  # Use the default 1000, #100000,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    write_out=False,
    output_path="eval_results/lm_eval",
    output_base_path=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model, transformers.PreTrainedModel object, or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write details about prompts and logits to json for all tasks
    :param output_base_path: str, optional
        Directory to which detailed eval info will be written. Defaults to present working dir.
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    logger.info(f"Loading model {model_args} ...")
    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
                "trust_remote_code": True,
            },
        )
    elif isinstance(model, transformers.PreTrainedModel):
        lm = lm_eval.models.get_model("hf-causal")(
            pretrained=model,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )
        no_cache = True
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    if not no_cache:
        lm = lm_eval.base.CachingLM(
            lm,
            "lm_cache/" + (model if isinstance(model, str) else model.model.config._name_or_path) + "_" +
            model_args.replace("=", "-").replace(",", "_").replace("/", "-") + ".db",
        )

    logger.info(f"Loaded model {model_args}")

    all_results = []
    for task_names_list, num_fewshot, limit in tasks:
        logger.info(f"---------- {len(task_names_list)} tasks: {task_names_list[:5]}..., {num_fewshot=}, {limit=}")
        task_dict = get_task_dict(task_names_list)
        print(f"Got {len(task_dict)} tasks: {task_dict.keys()}")

        if check_integrity:
            run_task_tests(task_list=task_names_list)

        results = evaluate(
            lm=lm,
            task_dict=task_dict,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            description_dict=description_dict,
            decontamination_ngrams_path=decontamination_ngrams_path,
            write_out=write_out,
            output_path=output_path,
            output_base_path=output_base_path,
        )

        clean_memory()

        # add info about the model and few shot config
        model_name = None
        if isinstance(model, str):
            model_name = model
        elif isinstance(model, transformers.PreTrainedModel):
            model_name = "pretrained=" + model.config._name_or_path
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "batch_sizes": list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else [],
            "device": device,
            "no_cache": no_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "description_dict": description_dict,
        }
        print(f"{results=}")
        all_results.append(results)

    logger.info(f"LMEval Done. {model_args}")

    return all_results


class MultiChoice:

    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0 and not _is_json_task(value):
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


openllm_tasks = [
    "arc_challenge|25|0",
    "hellaswag|10|0",
    "hendrycksTest-*|5|0",
    "truthfulqa_mc|0|0",
    "winogrande|5|0",
    "gsm8k|5|0",
    "drop|3|0",
]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--show_results", action="store_true", help="")
    parser.add_argument("--do_gen", action="store_true", help="")

    parser.add_argument("--model", type=str, default="hf-causal-experimental")
    parser.add_argument("--model_args", default="")
    # parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    # multiple task argument
    parser.add_argument("--task", type=str, nargs='+', default=openllm_tasks, help="<task1 name>,<task2 name>...|<num_fewshot>|<limit>")
    parser.add_argument("--batch_size", type=str, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_path", default="eval_results/lm_eval")
    # parser.add_argument("--limit", type=float, default=None,
    #                     help="Limit the number of examples per task. "
    #                          "If <1, limit is a percentage of the total number of examples.")
    parser.add_argument("--data_sampling", type=float, default=None)
    parser.add_argument("--no_cache", action="store_true", default=True)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")
    parser.add_argument("--write_out", action="store_true", default=True)
    parser.add_argument("--output_base_path", type=str, default=None)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--results_file", type=str)

    return parser.parse_args()


def do_gen(args):
    start_time = datetime.now().strftime("%Y%m%d%H%M%S")


    # assert not args.provide_description  # not implemented

    # if args.limit:
    #     print(
    #         "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
    #     )

    # if args.tasks is None:
    #     task_names = tasks.ALL_TASKS
    # else:
    #     task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    # print(f"Selected Tasks: {task_names}")

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    eval_tasks = []
    for t in args.task:
        tokens = t.split("|")
        print(f"{tokens=}")
        limit = None
        if len(tokens) == 2:
            str_task_names, num_fewshot = tokens
            num_fewshot = int(num_fewshot)
            task_names = pattern_match(str_task_names.split(","), tasks.ALL_TASKS)
        elif len(tokens) == 3:
            str_task_names, num_fewshot, limit = tokens
            num_fewshot = int(num_fewshot)
            limit = int(limit)
            if limit <= 0:
                limit = None
            task_names = pattern_match(str_task_names.split(","), tasks.ALL_TASKS)
        else:
            raise ValueError(f"Invalid task format: {t}, must be <task1 name>,<task2 name>...|<num_fewshot>|<limit>")
        if args.limit is not None:
            limit = args.limit
            if limit <= 0:
                limit = None
        eval_tasks.append((task_names, num_fewshot, limit))

    print(f"Selected Tasks: {eval_tasks}")

    # results = simple_evaluate(
    #     model=args.model,
    #     model_args=args.model_args,
    #     tasks=task_names,
    #     num_fewshot=args.num_fewshot,
    #     batch_size=args.batch_size,
    #     max_batch_size=args.max_batch_size,
    #     device=args.device,
    #     no_cache=args.no_cache,
    #     limit=args.limit,
    #     description_dict=description_dict,
    #     decontamination_ngrams_path=args.decontamination_ngrams_path,
    #     check_integrity=args.check_integrity,
    #     write_out=args.write_out,
    #     output_base_path=args.output_base_path,
    # )
    all_results = do_lmeval(
        model=args.model,
        model_args=args.model_args,
        tasks=eval_tasks,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        output_path=args.output_path,
        output_base_path=args.output_base_path,
    )

    summary = summarize_results(all_results)
    print(f"{summary=}")
    all_results.append(summary)

    dumped = json.dumps(all_results, ensure_ascii=False, indent=2)
    print(dumped)

    end_time = datetime.now().strftime("%Y%m%d%H%M%S")
    if args.output_path:
        # os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        os.makedirs(args.output_path, exist_ok=True)
        results_file = f"{args.output_path}/lmeval_results_{start_time}_{end_time}.json"
        with open(results_file, "w") as f:
            f.write(dumped)
        latest_results_file = f"{args.output_path}/lmeval_results_latest.json"
        if os.path.islink(latest_results_file):
            os.remove(latest_results_file)
        os.symlink(os.path.basename(results_file), latest_results_file)


    # batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))
    # print(
    #     f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
    #     f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
    # )
    print(f"{args.model_args}")
    for results in all_results:
        if "results" in results:
            print(make_table(results))
        else:
            print(results)

    print(f"Done.\n")

def show_results(args):
    results_file = args.results_file
    if results_file is None:
        latest_results_file = f"{args.output_path}/lmeval_results_latest.json"
    all_results = json.load(open(latest_results_file))
    summary = summarize_results(all_results)
    print(summary)
    for results in all_results:
        if "results" in results:
            print(make_table(results))
        else:
            print(results)

def main():
    args = parse_args()
    print(f"{args=}")

    if args.do_gen:
        do_gen(args)
    if args.show_results:
        show_results(args)

if __name__ == "__main__":
    main()
