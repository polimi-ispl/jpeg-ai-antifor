#!/bin/bash

# Define the set_target_bpp values
bpp_values=(12 25 75 100 200)

# Define the GPU to be used
gpu=4

# Define the number of samples
num_samples=100

# --- Compress the Imagenet testset at various bpp values with the CPU and GPU--- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/imagenet/original"
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/CPU_vs_GPU/imagenet/compressed_CPU"

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Comparing CPU vs GPU compression ImageNet started..."

# Loop through each bpp value and run the Python script
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 ImageNet test set with CPU, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --set_target_bpp=${bpp} --models_dir_name ../models --num_samples=${num_samples}
done

# Change the output directory
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/CPU_vs_GPU/imagenet/compressed_GPU"
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 ImageNet test set with GPU, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --gpu=${gpu} --set_target_bpp=${bpp} --models_dir_name ../models --num_samples=${num_samples}
done

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Comparing CPU vs GPU compressing ImageNet finished!"

# --- Compress the CelebA testset at various bpp values with the CPU and GPU--- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/celeba/original"
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/CPU_vs_GPU/celeba/compressed_CPU"

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Comparing CPU vs GPU compressing CelebA started..."

# Loop through each bpp value and run the Python script
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 CelebA test set with CPU, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --set_target_bpp=${bpp} --models_dir_name ../models --num_samples=${num_samples}
done

# Change the output directory
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/CPU_vs_GPU/celeba/compressed_GPU"
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 CelebA test set with GPU, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --gpu=${gpu} --set_target_bpp=${bpp} --models_dir_name ../models --num_samples=${num_samples}
done

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Comparing CPU vs GPU compressing CelebA finished!"

# --- Compress the COCO testset at various bpp values with the CPU and GPU--- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/coco/original"
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/CPU_vs_GPU/coco/compressed_CPU"

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Comparing CPU vs GPU compressing COCO started..."

# Loop through each bpp value and run the Python script
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 COCO test set with CPU, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --set_target_bpp=${bpp} --models_dir_name ../models --num_samples=${num_samples}
done

# Change the output directory
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/CPU_vs_GPU/coco/compressed_GPU"
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 COCO test set with GPU, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --gpu=${gpu} --set_target_bpp=${bpp} --models_dir_name ../models --num_samples=${num_samples}
done

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Comparing CPU vs GPU compressing COCO finished!"

# --- Compress the RAISE testset at various bpp values with the CPU and GPU--- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/raise/original"
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/CPU_vs_GPU/raise/compressed_CPU"

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Comparing CPU vs GPU compressing COCO started..."

# Loop through each bpp value and run the Python script
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 RAISE test set with CPU, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --set_target_bpp=${bpp} --models_dir_name ../models --num_samples=${num_samples}
done

# Change the output directory
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/CPU_vs_GPU/raise/compressed_GPU"
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the Wang2020 RAISE test set with GPU, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --gpu=${gpu} --set_target_bpp=${bpp} --models_dir_name ../models --num_samples=${num_samples}
done

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Comparing CPU vs GPU compressing RAISE finished!"