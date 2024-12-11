"""
Small script using the src.reco.coders.encoder module to encode and decode a list of samples inside a directory
and save the decoded image as a .png file.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
import os
import torch
import glob
from PIL import Image
import sys
import argparse
from tqdm import tqdm
from multiprocessing import cpu_count
# --- Setup the JPEG-AI software suite --- #
sys.path.append('../')
from utils.params import *
sys.path.append(JPEG_AI_PATH)  # Add the jpeg-ai-reference-software to the path
from src.codec import get_downloader
from src.codec.common import Image
from src.codec.coders import CodecEncoder
from src.codec.coders import (def_encoder_base_parser, def_encoder_parser_decorator)


# --- Helpers functions and classes --- #
class RecoEncoder(CodecEncoder):
    def __init__(self, base_parser, parser_decorator, name='reco'):
        super(RecoEncoder, self).__init__(name, base_parser, parser_decorator)

    def encode_stream(self, params):
        raw_image = Image.read_file(params['input_path'])

        if self.ce.target_device == 'cpu':
            #torch.set_num_threads(1)
            torch.set_num_threads(cpu_count()//2)

        self.rec_image, decisions = self.ce.compress(raw_image)

        self.create_bs(params['bin_path'])
        self.init_ec_module()

        self.ce.encode(self.ec_module, decisions)

        self.close_bs()

        return decisions

    def encode_and_decode(self, input_path: str, bin_path: str, dec_save_path: str):
        # Open the image
        try:
            # Read the image file
            raw_image = Image.read_file(input_path)

            # Set target device
            if self.ce.target_device == 'cpu':
                #torch.set_num_threads(1)
                torch.set_num_threads(cpu_count()//2)

            # Encode and decode the image
            self.rec_image, decisions = self.ce.compress(raw_image)

            # Save the bitstream
            self.create_bs(bin_path)
            self.init_ec_module()

            self.ce.encode(self.ec_module, decisions)

            self.close_bs()

            # Save decoded image
            self.rec_image.write_png(dec_save_path)

            return decisions
        except Exception as e:
            print(f"Error while processing the image: {e}")
            return None


def create_custom_parser(args: argparse.Namespace):
    parser = def_encoder_base_parser('Reconstruction')

    # Manually add the arguments from the first parser to the second parser
    for key, value in vars(args).items():
        parser.add_argument(f'--{key}', default=value, type=type(value))

    return parser


def list_images(directory):
    # Define the image formats to look for
    image_formats = ["*.png", "*.PNG", "*.jpeg", "*.JPEG", "*.tiff", "*.TIFF", "*.jpg", "*.JPG", '*.tif', '*.TIF']

    # List to store the image paths
    image_files = []

    # Iterate over each format and collect the image files
    for format in image_formats:
        image_files.extend(glob.glob(os.path.join(directory, format)))

    return image_files


def process_dir_with_encoder(coder: RecoEncoder, input_dir: str, save_dir: str):

    # --- Setup the coding engine --- #
    coder.print_coder_info()

    kwargs, params, _ = coder.init_common_codec(build_model=True, ce=None, cmd_args = None,
                                                overload_ce=True, cmd_args_add=False)
    profiler_path = kwargs.get('profiler_path', None)
    # print(params)

    # Load the models
    coder.load_models(get_downloader(kwargs.get('models_dir_name', 'models'), critical_for_file_absence=not kwargs.get('skip_loading_error', False)))
    coder.set_target_bpp_idx(kwargs['bpp_idx'])

    # --- Process the directory --- #

    # Create the directories if they don't exist
    save_dir = os.path.join(save_dir, f"target_bpp_{kwargs['set_target_bpp']}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Process all the files in the input directory
    # and save the decoded images in the save directory
    for root, _, files in os.walk(input_dir):
        # Find all the images in root with various formats (e.g., PNG, JPEG, TIFF, etc.)
        image_files = list_images(root)
        # if there are no images in the directory, skip
        if not image_files:
            continue
        # Create the subdirectory in the save directory for the particular folder we are saving
        sub_save_dir = os.path.join(save_dir, os.path.relpath(root, input_dir))
        os.makedirs(sub_save_dir, exist_ok=True)
        # Process the images
        for image_path in tqdm(image_files):
            file = os.path.basename(image_path)
            extension = file.split(".")[-1]
            bin_path = os.path.join(sub_save_dir, file.replace(f'.{extension}', ""))
            dec_path = os.path.join(sub_save_dir, file.replace(f'.{extension}', ".png"))
            if os.path.exists(dec_path):
                print('Skipping (already decoded)', dec_path)
                continue
            coder.encode_and_decode(image_path, bin_path, dec_path)

# --- Main --- #
if __name__ == "__main__":

    # --- Setup an argument parser --- #
    parser = argparse.ArgumentParser(description='Compress a directory of images using the RecoEncoder')
    parser.add_argument('--gpu', type=int, default=None, help='GPU index')
    parser.add_argument('input_path', type=str, default='samples', help='Input directory')
    parser.add_argument('bin_path', type=str, default='decoded_samples', help='Save directory')
    parser.add_argument('--set_target_bpp', type=int, default=1, help='Set the target bpp '
                                                                      '(multiplied by 100)')
    parser.add_argument('--models_dir_name', type=str, default='../models', help='Directory name for the '
                                                                                 'models used in the encoder-decoder'
                                                                                 'pipeline')
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to process')
    args = parser.parse_args()


    # --- Setup the device --- #
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU {args.gpu}")
        args.target_device = 'gpu'

    # --- Setup the coder --- #
    encoder_parser = create_custom_parser(args)

    # --- Setup the encoder --- #
    coder = RecoEncoder(encoder_parser, def_encoder_parser_decorator(encoder_parser))

    # --- Process the directory --- #
    process_dir_with_encoder(coder, args.input_path, args.bin_path)