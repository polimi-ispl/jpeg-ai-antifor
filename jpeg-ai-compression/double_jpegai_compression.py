"""
Small script to double JPEG AI compress spliced images.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
import os
import torch
import glob
import PIL.Image as PILImage
import sys
import argparse
from tqdm import tqdm
import pandas as pd
from multiprocessing import cpu_count
# --- Setup the JPEG-AI software suite --- #
sys.path.append('../')
from utils.params import *
sys.path.append(JPEG_AI_PATH)  # Add the jpeg-ai-reference-software to the path
from src.codec import get_downloader
from src.codec.common import Image
from src.codec.coders import CodecEncoder
from src.codec.coders import (def_encoder_base_parser, def_encoder_parser_decorator)
import numpy as np


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

    def encode_decode_raw_image(self, raw_image: Image, save_path):
        try:
            # Set target device
            if self.ce.target_device == 'cpu':
                # torch.set_num_threads(1)
                torch.set_num_threads(cpu_count() // 2)

            # Encode and decode the image
            self.rec_image, decisions = self.ce.compress(raw_image)

            # Save the bitstream
            self.create_bs(save_path.replace('.png', ''))
            self.init_ec_module()

            self.ce.encode(self.ec_module, decisions)

            self.close_bs()

            # Save decoded image
            self.rec_image.write_png(save_path)

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


def double_jpegai_compression(filename, data_info, coder, target_bpp, save_dir):

    # Get the images with specific filename
    images = data_info[data_info['filename'] == filename]

    # Cycle over all the BPP values
    rows = []
    for idx, row in images.iterrows():

        # --- Load the data

        # Get the target image, i.e., the image with specific target bpp
        target_image = Image.read_file(images[images['target_bpp'] == target_bpp/100].index.get_level_values(1).item())
        # Get the source image
        source_image = Image.read_file(idx[1])
        # Load the tampering mask
        mask = PILImage.open(row['gt']).convert('L')
        # GT "has to be 0 for pristine pixels and 1 for forged pixels."
        if idx[0] == 'DSO-1':
            mask = np.array(mask) < 0.1  # DSO-1 is inverted
        elif idx[0] == 'Columbia':
            mask = np.array(mask) > 127
        else:
            mask = np.array(mask).astype(bool)

        # --- Splice the images
        spliced_image = target_image.get_tensor()
        spliced_image[:, :, mask] = source_image.get_tensor()[:, :, mask]

        # --- Encode and decode the image

        # Save the spliced image using the target BPP, specifying in the filename the source BPP
        save_path = os.path.join(save_dir, f"{filename}_source_bpp-{int(row['target_bpp']*100)}.png")

        # Re-create the Image file for the spliced image using the target image as reference
        spliced_image = Image.create_from_tensor(spliced_image, data_range=target_image.data_range,
                                                 bit_depth=target_image.bit_depth, color_space=target_image.color_space)
        spliced_image.input_file = save_path

        # Encode and decode the image
        if os.path.exists(save_path):
            print(f"Skipping {save_path} as it already exists")
        else:
            coder.encode_decode_raw_image(spliced_image, save_path)

        # --- Save the info
        rows.append(pd.DataFrame(index=[save_path], data={'filename': filename, 'source_bpp': row['target_bpp'],
                                                          'target': images[images['target_bpp'] == target_bpp/100].index.get_level_values(1).item(),
                                                          'gt': row['gt']}))

    # Concatenate all the info about the images
    all_info = pd.concat(rows)
    return all_info


def main(coder: RecoEncoder, input_dir: str, save_dir: str, dataset: str, num_samples: int = None):

    # --- Setup the coding engine --- #
    coder.print_coder_info()

    kwargs, params, _ = coder.init_common_codec(build_model=True, ce=None, cmd_args=None,
                                                overload_ce=True, cmd_args_add=False)
    profiler_path = kwargs.get('profiler_path', None)
    # print(params)

    # Load the models
    coder.load_models(get_downloader(kwargs.get('models_dir_name', 'models'),
                                     critical_for_file_absence=not kwargs.get('skip_loading_error', False)))
    coder.set_target_bpp_idx(kwargs['bpp_idx'])

    # --- Load the spliced samples --- #
    all_data_info = pd.read_csv(os.path.join(input_dir, 'spliced_data_complete.csv'), index_col=[0, 1, 2])
    # Get only JPEG AI compressed images
    all_data_info = all_data_info.loc['JPEG-AI']
    # Get only the images from dataset
    all_data_info = all_data_info[all_data_info['dataset'] == dataset]

    # --- Process the samples --- #

    # Create the directories if they don't exist
    save_dir = os.path.join(save_dir, dataset, 'double_jpegai_compressed', f"target_bpp_{kwargs['set_target_bpp']}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Find all the unique filenames
    unique_filenames = all_data_info['filename'].unique()
    if num_samples:  # get only num_samples
        unique_filenames = unique_filenames[:num_samples]
    results_df = []
    for filename in tqdm(unique_filenames, total=len(unique_filenames)):
        results_df.append(double_jpegai_compression(filename, all_data_info, coder, kwargs['set_target_bpp'], save_dir))

    # --- Save the results
    results_df = pd.concat(results_df)
    results_df.to_csv(os.path.join(save_dir, 'double_jpegai_compressed.csv'))


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
    parser.add_argument('--dataset', type=str, default='DSO-1', help='Dataset to process',
                        choices=['CASIA1', 'CocoGLIDE', 'Columbia', 'Coverage', 'DSO-1'])
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
    main(coder, args.input_path, args.bin_path, args.dataset, args.num_samples)
