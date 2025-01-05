#!/usr/bin/env bash
INPUT_DIR=/nas/public/exchange/JPEG-AI/data/SPLICED/dso-1_dsi-1_jpeg_ai/DSO-1/qf_65/splicing-89.jpg
OUTPUT_DIR=/nas/public/exchange/JPEG-AI/test_results/splicing/TruFor/qf_65

echo "Running script on ${INPUT_DIR}"

mkdir -p ${OUTPUT_DIR}
docker run --runtime=nvidia --gpus all -v $(realpath ${INPUT_DIR}):/data -v $(realpath ${OUTPUT_DIR}):/data_out trufor -gpu 1 -in "${INPUT_DIR}" -out "${OUTPUT_DIR}"
