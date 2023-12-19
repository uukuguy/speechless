#!/bin/bash

DIR1="`dirname $BASH_SOURCE`"
MYDIR=`readlink -f "$DIR1"`
cd ${MYDIR}

source env_s/bin/activate

# bash ptuning/ds_train_finnetune.sh  # 全参微调
bash ptuning/train.sh  # ptuning微调

