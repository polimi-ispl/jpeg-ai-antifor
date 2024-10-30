#!/bin/bash

# Define the set_target_bpp values
bpp_values=(12 25 75 100 200)
qfs=(65 75 85 95)

# Define the GPU to be used
gpu=4

# Define the model directory
models_dir="/nas/home/ecannas/third_party_code/jpeg-ai-reference-software/models"

# Define the input and output directories
input_path="/nas/public/dataset/columbia_uncompressed/4cam_splc"
bin_path="/nas/public/exchange/JPEG-AI/data/SPLICED/Columbia_uncompressed/4cam_splc"

## Send a message to the personal Slack channel
#python ../utils/slack.py -u edo.cannas -m "Compression of Columbia uncompressed spliced started..."
#
## Loop through each bpp value and run the Python script
#for bpp in "${bpp_values[@]}"; do
#    echo "Running compress_dataset.py with --set_target_bpp=${bpp}"
#    python compress_dataset.py ${input_path} ${bin_path} --set_target_bpp=${bpp} --gpu ${gpu} --models_dir_name ${models_dir}
#done
#
## Send a message to the personal Slack channel
#python ../utils/slack.py -u edo.cannas -m "Compression of Columbia uncompressed spliced finished!"

# Send a message to the personal Slack channel
python ../utils/slack.py -u edo.cannas -m "Standard JPEG compression of Columbia uncompressed spliced started..."

# Loop through each bpp value and run the Python script
for qf in "${qfs[@]}"; do
    echo "Running compress_dataset.py with --set_target_bpp=${qf}"
    python jpeg_compress_dataset.py --input_path=${input_path} --output_path=${bin_path} --set_target_qf=${qf}
done

# Send a message to the personal Slack channel
python ../utils/slack.py -u edo.cannas -m "Standad JPEG compression of Columbia uncompressed spliced finished!"