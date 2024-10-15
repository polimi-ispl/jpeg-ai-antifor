#!/bin/bash

# Define the set_target_bpp values
bpp_values=(12 25 75 100 200)

# Define the input and output directories
input_path="/nas/home/ecannas/third_party_code/jpeg-ai-reference-software/data/test"
bin_path="/nas/home/ecannas/third_party_code/jpeg-ai-reference-software/data/test_compressed_cpu"

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Compression tasks started..."

# Loop through each bpp value and run the Python script
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py with --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --set_target_bpp=${bpp} --models_dir_name ../models
done

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Compression tasks finished!"