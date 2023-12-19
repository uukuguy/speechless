#!/bin/bash

DIR1="`dirname $BASH_SOURCE`"
MYDIR=`readlink -f "$DIR1"`
cd ${MYDIR}

source env_s/bin/activate

# 激活环境
echo "get feature"
python get_feature.py
echo "get feature done!"
echo "training"
# 训练
bash train.sh
echo "training done!"
