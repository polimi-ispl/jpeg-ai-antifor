#!/bin/bash

# Define the set_target_bpp values
bpp_values=(12 25 75 100 200)

# Define the GPU to be used
gpu=4

# --- Compress the ImageNet testset at various bpp values --- #

## Define the input and output directories
#input_path="/nas/public/exchange/JPEG-AI/data/TEST/imagenet/original"
#bin_path="/nas/public/exchange/JPEG-AI/data/TEST/imagenet/compressed"
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression Imagenet started..."
#
## Loop through each bpp value and run the Python script
#for bpp in "${bpp_values[@]}"; do
#    echo "Running compress_dataset.py for the Wang2020 ImageNet test set, --set_target_bpp=${bpp}"
#    python compress_dataset.py ${input_path} ${bin_path} --gpu ${gpu} --set_target_bpp=${bpp} --models_dir_name ../models
#done
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression Imagenet finished!"
#
## --- Compress the CelebA testset at various bpp values --- #
#
## Define the input and output directories
#input_path="/nas/public/exchange/JPEG-AI/data/TEST/celeba/original"
#bin_path="/nas/public/exchange/JPEG-AI/data/TEST/celeba/compressed"
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression CelebA started..."
#
## Loop through each bpp value and run the Python script
#for bpp in "${bpp_values[@]}"; do
#    echo "Running compress_dataset.py for the Wang2020 CelebA test set, --set_target_bpp=${bpp}"
#    python compress_dataset.py ${input_path} ${bin_path} --gpu ${gpu} --set_target_bpp=${bpp} --models_dir_name ../models
#done
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression CelebA finished!"
#
## --- Compress the COCO testset at various bpp values --- #
#
## Define the input and output directories
#input_path="/nas/public/exchange/JPEG-AI/data/TEST/coco/original"
#bin_path="/nas/public/exchange/JPEG-AI/data/TEST/coco/compressed"
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression COCO started..."
#
## Loop through each bpp value and run the Python script
#for bpp in "${bpp_values[@]}"; do
#    echo "Running compress_dataset.py for the Wang2020 COCO test set, --set_target_bpp=${bpp}"
#    python compress_dataset.py ${input_path} ${bin_path} --gpu ${gpu} --set_target_bpp=${bpp} --models_dir_name ../models
#done
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression COCO finished!"
#
## --- Compress the RAISE testset at various bpp values --- #
#
## Define the input and output directories
#input_path="/nas/public/exchange/JPEG-AI/data/TEST/raise/original"
#bin_path="/nas/public/exchange/JPEG-AI/data/TEST/raise/compressed"
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression RAISE started..."
#
## Loop through each bpp value and run the Python script
#for bpp in "${bpp_values[@]}"; do
#    echo "Running compress_dataset.py for the Wang2020 RAISE test set, --set_target_bpp=${bpp}"
#    python compress_dataset.py ${input_path} ${bin_path} --gpu ${gpu} --set_target_bpp=${bpp} --models_dir_name ../models
#done
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression RAISE finished!"

## --- Compress the LAION testset at various bpp values --- #
#
## Define the input and output directories
#input_path="/nas/public/exchange/JPEG-AI/data/TEST/laion/original"
#bin_path="/nas/public/exchange/JPEG-AI/data/TEST/laion/compressed"
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression LAION started..."
#
## Loop through each bpp value and run the Python script
#for bpp in "${bpp_values[@]}"; do
#    echo "Running compress_dataset.py for the Ohja2023 LAION test set, --set_target_bpp=${bpp}"
#    python compress_dataset.py ${input_path} ${bin_path} --gpu ${gpu} --set_target_bpp=${bpp} --models_dir_name ../models
#done
#
## Send a message to the personal Slack channel
#python slack.py -u edo.cannas -m "Compression LAION finished!"

# --- Compress the FFHQ testset at various bpp values --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/ffhq/original"
bin_path="/nas/public/exchange/JPEG-AI/data/TEST/ffhq/compressed"

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Compression FFHQ started..."

# Loop through each bpp value and run the Python script
for bpp in "${bpp_values[@]}"; do
    echo "Running compress_dataset.py for the FFHQ test set, --set_target_bpp=${bpp}"
    python compress_dataset.py ${input_path} ${bin_path} --gpu ${gpu} --set_target_bpp=${bpp} --models_dir_name ../models
done

# Send a message to the personal Slack channel
python slack.py -u edo.cannas -m "Compression FFHQ finished!"