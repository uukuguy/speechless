from math_verify import verify, parse
import os
import signal
from contextlib import redirect_stdout, redirect_stderr

    
def compute_score(sequences: str, ground_truth: str) -> float:
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
