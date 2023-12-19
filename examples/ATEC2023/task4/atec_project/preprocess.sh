#!/bin/bash

DIR1="`dirname $BASH_SOURCE`"
MYDIR=`readlink -f "$DIR1"`
cd ${MYDIR}

source env_s/bin/activate

python utils/preprocess_dataset.py --input /home/admin/workspace/job/input/training/vpn --dataset_name vpn --output_path utils/build_datasets/vpn --output_name vpn # 流量数据预处理
python utils/preprocess_dataset.py --input /home/admin/workspace/job/input/training/botnet --dataset_name botnet --output_path utils/build_datasets/botnet --output_name botnet
python utils/preprocess_dataset.py --input /home/admin/workspace/job/input/training/malware --dataset_name malware --output_path utils/build_datasets/malware --output_name malware

python utils/merge_data.py