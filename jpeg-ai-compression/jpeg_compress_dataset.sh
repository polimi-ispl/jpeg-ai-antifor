#!/bin/bash

# Define the set_target_bpp values
qf_values=(65 75 85 95)

# --- Compress the ImageNet --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/imagenet/original"
out_path="/nas/public/exchange/JPEG-AI/data/TEST/imagenet/compressed"


# Loop through qf_values and run the Python script
for qf in "${qf_values[@]}"; do
   echo "Running compress_dataset.py for the Wang2020 ImageNet test set, --set_target_qf=${qf}"
   python jpeg_compress_dataset.py --input_path=${input_path} --output_path=${out_path} --set_target_qf=${qf}
done

# --- Compress the CelebA testset at various bpp values --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/celeba/original"
out_path="/nas/public/exchange/JPEG-AI/data/TEST/celeba/compressed"


# Loop through qf_values and run the Python script
for qf in "${qf_values[@]}"; do
   echo "Running compress_dataset.py for the Celeba test set, --set_target_qf=${qf}"
   python jpeg_compress_dataset.py --input_path=${input_path} --output_path=${out_path} --set_target_qf=${qf}
done


# --- Compress the COCO testset at various bpp values --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/coco/original"
out_path="/nas/public/exchange/JPEG-AI/data/TEST/coco/compressed"

# Loop through qf_values and run the Python script
for qf in "${qf_values[@]}"; do
   echo "Running compress_dataset.py for the COCO test set, --set_target_qf=${qf}"
   python jpeg_compress_dataset.py --input_path=${input_path} --output_path=${out_path} --set_target_qf=${qf}
done


# --- Compress the RAISE testset at various bpp values --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/raise/original"
out_path="/nas/public/exchange/JPEG-AI/data/TEST/raise/compressed"

# Loop through qf_values and run the Python script
for qf in "${qf_values[@]}"; do
   echo "Running compress_dataset.py for the Raise test set, --set_target_qf=${qf}"
   python jpeg_compress_dataset.py --input_path=${input_path} --output_path=${out_path} --set_target_qf=${qf}
done


# --- Compress the LAION testset at various bpp values --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/laion/original"
out_path="/nas/public/exchange/JPEG-AI/data/TEST/laion/compressed"

# Loop through qf_values and run the Python script
for qf in "${qf_values[@]}"; do
   echo "Running compress_dataset.py for the LAION test set, --set_target_qf=${qf}"
   python jpeg_compress_dataset.py --input_path=${input_path} --output_path=${out_path} --set_target_qf=${qf}
done


# --- Compress the FFHQ testset at various bpp values --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/ffhq/original"
out_path="/nas/public/exchange/JPEG-AI/data/TEST/ffhq/compressed"

# Loop through qf_values and run the Python script
for qf in "${qf_values[@]}"; do
   echo "Running compress_dataset.py for the FFHQ test set, --set_target_qf=${qf}"
   python jpeg_compress_dataset.py --input_path=${input_path} --output_path=${out_path} --set_target_qf=${qf}
done


# --- Compress the LSUN --- #

# Define the input and output directories
input_path="/nas/public/exchange/JPEG-AI/data/TEST/lsun/original"
out_path="/nas/public/exchange/JPEG-AI/data/TEST/lsun/compressed"


# Loop through qf_values and run the Python script
for qf in "${qf_values[@]}"; do
   echo "Running compress_dataset.py for the lsun test set, --set_target_qf=${qf}"
   python jpeg_compress_dataset.py --input_path=${input_path} --output_path=${out_path} --set_target_qf=${qf}
done

