#!/usr/bin/env bash

# USER PARAMETERS (put your device configuration params here)
DEVICE=4  # PUT THERE YOUR GPU ID
COMPRESSED_DIR='/nas/public/exchange/JPEG-AI/data/TEST'
SYN_DIR='/nas/public/exchange/JPEG-AI/data/TEST_SYN'
RESULTS_DIR='/nas/public/exchange/JPEG-AI/test_results'

echo ""
echo "-------------------------------------------------"
echo "| Testing CLIP2024 for all tasks |"
echo "-------------------------------------------------"

python ../utils/slack.py -u edo.cannas -m "Testing CLIP2024 for all tasks started..."

echo ""
echo "-------------------------------------------------"
echo "| Real VS JPEG-AI |"
echo "-------------------------------------------------"
python ../test_detector_jpeg-ai.py --input_dir=${COMPRESSED_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector CLIP2024 --weights_path /nas/public/exchange/JPEG-AI/code/utils/third_party/ClipBased_SyntheticImageDetection_main/weights --batch_size 1024

echo ""
echo "-------------------------------------------------"
echo "| Real VS JPEG |"
echo "-------------------------------------------------"
python ../test_detector_jpeg-standard.py --input_dir=${COMPRESSED_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector CLIP2024 --weights_path /nas/public/exchange/JPEG-AI/code/utils/third_party/ClipBased_SyntheticImageDetection_main/weights --batch_size 1024

echo ""
echo "-------------------------------------------------"
echo "| Real VS Synthetic |"
echo "-------------------------------------------------"
python ../test_detector_syn.py --input_dir=${SYN_DIR} --output_dir=${RESULTS_DIR} --gpu=${DEVICE} --detector CLIP2024 --weights_path /nas/public/exchange/JPEG-AI/code/utils/third_party/ClipBased_SyntheticImageDetection_main/weights --batch_size 1024

python ../utils/slack.py -u edo.cannas -m "Testing CLIP2024 for all tasks finished."