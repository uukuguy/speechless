#!/bin/bash

set -uxo pipefail


PARENT_DIR=$(dirname $(pwd))
PACKAGE_NAME=$(basename $(pwd))

CLOUD_DIR=/RFT/${PACKAGE_NAME}

cd $PARENT_DIR
zip -ry $PACKAGE_NAME.zip $PACKAGE_NAME

bypy upload $PACKAGE_NAME.zip $CLOUD_DIR/

cd $PACKAGE_DIR
