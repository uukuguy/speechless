from math_verify import verify, parse
import os
import signal
from contextlib import redirect_stdout, redirect_stderr

    
def math_verify_api_compute_score(sequences: str, ground_truth: str) -> float:
    with open(os.devnull, "w") as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                def timeout_handler(signum, frame):
                        pass
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(5)
                answer = parse(sequences)
                if ground_truth.find('\\boxed')==-1:
                    groud_trush = '\\boxed{'+ground_truth+'}'
                else:
                    groud_trush = ground_truth
                groud_trush=parse(groud_trush)
                if verify(answer, groud_trush):
                    return {"score":1,'acc':1}
                signal.alarm(0)
            except (TimeoutError, Exception):
                signal.alarm(0)
                pass
    return {"score":0,'acc':0}


def compute_score(data_source, solution_str, ground_truth, extra_info=None, sandbox_fusion_url=None, concurrent_semaphore=None, memory_limit_mb=None):
    if data_source in ['math','omni_math','amc','aime']:
        # E3-RL4LLMs
        res = math_verify_api_compute_score(solution_str, ground_truth)
    else:
        from verl.utils.reward_score import default_compute_score
        res = default_compute_score(data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb)

    if isinstance(res, dict):
        return res
    elif isinstance(res, (int, float, bool)):
        return float(res)
    else:
        return float(res[0])
