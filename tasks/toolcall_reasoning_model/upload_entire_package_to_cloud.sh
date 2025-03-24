#!/bin/bash

set -uxo pipefail

current_time=$(date "+%Y.%m.%d-%H.%M.%S")

PARENT_DIR=$(dirname $(pwd))
PACKAGE_NAME=$(basename $(pwd))

CLOUD_DIR=/RFT/${PACKAGE_NAME}
PACKAGE_DIR=$PARENT_DIR/$PACKAGE_NAME

ZIP_FILE=$PACKAGE_NAME-${current_time}.tar.gz

cd $PARENT_DIR
tar -zcvf ${ZIP_FILE} $@ $PACKAGE_NAME

bypy upload ${ZIP_FILE} $CLOUD_DIR/

cd $PACKAGE_DIR
