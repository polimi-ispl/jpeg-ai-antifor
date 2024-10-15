"""
A simple script to test state-of-the-art detectors for synthetic image detection on our dataset of JPEG-AI compressed samples.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
import os
import sys
import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from multiprocessing import cpu_count
from utils.params import *
from utils.data import JPEGAIDataset
from utils.slack import ISPLSlack
from utils.detector import Detector
import pandas as pd

# --- Helpers functions and classes --- #

def process_dataset(args: argparse.Namespace):

    # --- Parse the params we need --- #
    input_dir = args.input_dir
    output_dir = args.output_dir
    gpu = args.gpu
    detector = args.detector
    weigths_paths = args.weights_path

    # --- Prepare the dataset --- #
    data_info = pd.read_csv(os.path.join(input_dir, 'data_info.csv'))
    dataset = JPEGAIDataset(root_dir=input_dir, data_df=data_info)


# --- Main --- #

