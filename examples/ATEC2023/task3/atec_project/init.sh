#!/bin/bash

DIR1="`dirname $BASH_SOURCE`"
MYDIR=`readlink -f "$DIR1"`
cd ${MYDIR}

source env_s/bin/activate
nvidia-smi
