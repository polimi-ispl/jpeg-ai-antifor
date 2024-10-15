#!/bin/bash

# Define the set_target_bpp values
bpp_values=(12 25 75 100 200)

# --- Compress the LSUN testset at various bpp values --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/lsun/original"
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/lsun/compressed"

# Define the GPU to be used
gpu=4

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Compression LSUN started..."

# Loop through each bpp value and run the Python script
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 LSUN test set, --set_target_bpp=${bpp}"
    python compress_lsun.py ${input_path} ${bin_path} --gpu ${gpu} --set_target_bpp=${bpp} --models_dir_name ../models
done

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Compression LSUN finished!"