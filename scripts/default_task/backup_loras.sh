#!/bin/bash

SCRIPT_PATH=$(cd $(dirname ${BASH_SOURCE[0]}); pwd)
PARENT_PATH=$(cd "${SCRIPT_PATH}/.." ; pwd)
TASK_NAME=$(basename ${SCRIPT_PATH})

echo "Script Path: ${SCRIPT_PATH}"
echo "Parent Path: ${PARENT_PATH}"
echo "Task Name: ${TASK_NAME}"

TASK_ZIP_FILE=${PARENT_PATH}/${TASK_NAME}-loras.$(date +%Y%m%d-%H%M%S).zip
cd .. && zip -0 -ry ${TASK_ZIP_FILE} ${TASK_NAME}
