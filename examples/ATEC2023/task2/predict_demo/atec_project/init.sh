#!/bin/bash
set -x
DIR1="`dirname $BASH_SOURCE`"
MYDIR=`readlink -f "$DIR1"`
cd ${MYDIR}
source env/bin/activate
export PYTHONHOME=${MYDIR}/env

#lscpu
#free -m

if type nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
fi
