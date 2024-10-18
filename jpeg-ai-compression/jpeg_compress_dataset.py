
# --- Libraries --- #
import os
import glob
from PIL import Image
import sys
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count


def list_images(directory):
    # Define the image formats to look for
    image_formats = ["*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.tiff", "*.TIFF", "*.jpg", "*.JPG"]

    # List to store the image paths
    image_files = []

    # Iterate over each format and collect the image files
    for format in image_formats:
        image_files.extend(glob.glob(os.path.join(directory, format)))

    return image_files


def process_dir_with_jpeg(input_dir: str, qf: int, save_dir: str):

    # --- Process the directory --- #

    # Create the directories if they don't exist
    save_dir = os.path.join(save_dir, f"qf_{qf}")
    os.makedirs(save_dir, exist_ok=True)

    # Process all the files in the input directory
    # and save the decoded images in the save directory

    # List of images in the input directory with various formats (e.g., PNG, JPEG, TIFF, etc.)
    image_files = list_images(input_dir)

    if not image_files: # lsun case

        # to deal with lsun dataset which has subfolders:
        for root, dirs, files in os.walk(input_dir):

            for sub_r in dirs:
                # working folder, where we have to find the image files
                img_dir = os.path.join(root, sub_r)

                image_files = list_images(img_dir)
                sub_save_dir = os.path.join(save_dir, sub_r)
                os.makedirs(sub_save_dir, exist_ok=True)

                for image_path in tqdm(image_files):
                    # Get the filename and the extension
                    file = os.path.basename(image_path)
                    extension = file.split(".")[-1]
                    out_path = os.path.join(sub_save_dir, file.replace(f'.{extension}', ".jpg"))
                    if os.path.exists(out_path):
                        print('Skipping (already decoded)', out_path)
                        continue
                    else:
                        # JPEG COMPRESSION
                        img = Image.open(image_path)
                        img.save(out_path, format='JPEG', quality=qf)

    else: # standard case

        for image_path in tqdm(image_files):
            # Get the filename and the extension
            file = os.path.basename(image_path)
            extension = file.split(".")[-1]
            out_path = os.path.join(save_dir, file.replace(f'.{extension}', ".jpg"))
            if os.path.exists(out_path):
                print('Skipping (already decoded)', out_path)
                continue
            else:
                # JPEG COMPRESSION
                img = Image.open(image_path)
                img.save(out_path, format='JPEG', quality=qf)


# --- Main --- #
if __name__ == "__main__":

    # --- Setup an argument parser --- #
    parser = argparse.ArgumentParser(description='Compress a directory of images using standard JPEG compression')
    parser.add_argument('--input_path', type=str, default='samples', help='Input directory')
    parser.add_argument('--output_path', type=str, default='decoded_samples', help='Save directory')
    parser.add_argument('--set_target_qf', type=int, default=75, help='Set the target quality factor')
    args = parser.parse_args()

    process_dir_with_jpeg(args.input_path, args.set_target_qf, args.output_path)
