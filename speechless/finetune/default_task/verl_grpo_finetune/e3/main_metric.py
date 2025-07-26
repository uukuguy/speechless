import json
import numpy as np

def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

def get_pass_at_n(res,
                  n_list=None): 
    if n_list is None:
        n_list=[1,2,4,8,16]
    dataset_passn={}
    for key in res['acc']:
        pass_at_n={}
        for n in n_list:
            pass_at_n[n]=[]
            pass_rate=[]
            for item in res['acc'][key]:
                pass_rate.append(pass_at_k(len(item),sum(item),n))
            pass_at_n[n].append(sum(pass_rate)/len(pass_rate)*100)
        dataset_passn[key]=pass_at_n
    return dataset_passn

def compute_metric(metric_files,n_list=None):
    all_res = {}
    all_res_length = {}
    for file in metric_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
        pass_at_n = get_pass_at_n(data,n_list)
        for dataset in pass_at_n.keys():
            if dataset not in all_res:
                all_res[dataset] = {}
            for n in pass_at_n[dataset].keys():
                if n not in all_res[dataset]:
                    all_res[dataset][n] = []
                all_res[dataset][n].extend(pass_at_n[dataset][n])
        for dataset in data["length"].keys():
            if dataset not in all_res_length:
                all_res_length[dataset] = []
            all_res_length[dataset].append(data["length"][dataset])

    res_at_n = {}
    res_l = {}
    # average pass@k over different checkpoints
    for dataset in all_res.keys():
        res_at_n[dataset] = {}
        for n in all_res[dataset].keys():
            res_at_n[dataset][n] = sum(all_res[dataset][n]) / len(all_res[dataset][n])
    # average length over different checkpoints
    for dataset in all_res_length.keys():
        res_l[dataset] = sum(all_res_length[dataset]) / len(all_res_length[dataset])
    return res_at_n, res_l

if __name__ == "__main__":
    all_res = compute_metric(
        [
            "./.metrics/grpo_0.json",
            "./.metrics/grpo_1.json",
            "./.metrics/grpo_2.json",
        ]
        ,
        n_list=[1,2,3,4,6,8,12,16]
    )
    print('--- pass@k ---')
    print(all_res[0])
    print('--- length ---')
    print(all_res[1])
    
