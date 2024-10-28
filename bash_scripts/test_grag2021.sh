#!/usr/bin/env bash

# USER PARAMETERS (put your device configuration params here)
DEVICE=3  # PUT THERE YOUR GPU ID
COMPRESSED_DIR='/nas/public/exchange/JPEG-AI/data/TEST'
SYN_DIR='/nas/public/exchange/JPEG-AI/data/TEST_SYN'
RESULTS_DIR='/nas/public/exchange/JPEG-AI/test_results'

echo ""
echo "-------------------------------------------------"
echo "| Testing Grag2021 for all tasks |"
echo "-------------------------------------------------"

python ../utils/slack.py -u edo.cannas -m "Testing Grag2021 for all tasks started..."

echo ""
echo "-------------------------------------------------"
echo "| Real VS JPEG-AI |"
echo "-------------------------------------------------"
python ../test_detector_jpeg-ai.py --input_dir=${COMPRESSED_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Grag2021_progan --weights_path /nas/home/ecannas/third_party_code/DMimageDetection/models/weights --batch_size 96

echo ""
echo "-------------------------------------------------"
echo "| Real VS JPEG |"
echo "-------------------------------------------------"
python ../test_detector_jpeg-standard.py --input_dir=${COMPRESSED_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Grag2021_progan --weights_path /nas/home/ecannas/third_party_code/DMimageDetection/models/weights --batch_size 96

echo ""
echo "-------------------------------------------------"
echo "| Real VS Synthetic |"
echo "-------------------------------------------------"
python ../test_detector_syn.py --input_dir=${SYN_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector Grag2021_progan --weights_path /nas/home/ecannas/third_party_code/DMimageDetection/models/weights --batch_size 96

python ../utils/slack.py -u edo.cannas -m "Testing Grag2021 for all tasks finished."