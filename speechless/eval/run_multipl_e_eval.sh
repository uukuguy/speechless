#!/bin/bash
MULTIPL_E_RESULTS_DIR=$(basename ${PWD})
docker run -it --rm \
    --network none \
    -v ${PWD}:/${MULTIPL_E_RESULTS_DIR}:rw \
    multipl-e-eval --dir /${MULTIPL_E_RESULTS_DIR} --output-dir /${MULTIPL_E_RESULTS_DIR} --recursive