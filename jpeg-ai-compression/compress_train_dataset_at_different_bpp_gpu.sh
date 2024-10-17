#!/bin/bash

# Define the set_target_bpp values
bpp_values=(12 25 75 100 200)

# --- Compress the LSUN testset at various bpp values --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TRAIN"
bin_path="/nas/public/exchange/JPEG-AI/data/TRAIN"

# Define the GPU to be used
gpu=4

# Send a message to the personal Slack channel
python ../utils/slack.py -u edo.cannas -m "Compression train dataset started..."

# Loop through each bpp value and run the Python script
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the train set of Mandelli2024, --set_target_bpp=${bpp}"
    python compress_train_dataset.py ${input_path} ${bin_path} --gpu ${gpu} --set_target_bpp=${bpp} --models_dir_name /nas/home/ecannas/third_party_code/jpeg-ai-reference-software/models
done

# Send a message to the personal Slack channel
python ../utils/slack.py -u edo.cannas -m "Compression train dataset finished!"