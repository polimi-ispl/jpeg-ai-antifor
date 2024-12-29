"""
A simple script to test state-of-the-art detectors for synthetic image detection in the various cases considered in our
paper, i.e.:
1. pristine VS pristine compressed with JPEG AI;
2. pristine VS pristine compressed with JPEG;
3. pristine VS synthetic;
4. synthetic VS synthetic compressed with JPEG AI;
5. synthetic VS synthetic compressed with JPEG;
6. pristine VS pristine w/ augmentations;
6. synthetic VS synthetic w/ augmentations.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
import os
import sys
import argparse
import torch
from tqdm import tqdm
from multiprocessing import cpu_count
from utils.params import *
from utils.data import JPEGAIDataset, JPEGDataset, SynImgDataset, get_transform_list
from utils.slack import ISPLSlack
from utils.detector import Detector
import pandas as pd

# --- Helpers functions and classes --- #

def main(args: argparse.Namespace):

    # --- Parse the params we need --- #
    input_dir = args.input_dir
    output_dir = args.output_dir
    gpu = args.gpu
    detector_name = args.detector
    weigths_paths = args.weights_path
    test_all = args.test_all
    test_type = args.test_type
    debug = args.debug
    batch_size = args.batch_size

    # --- Prepare the device --- #
    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # --- Prepare the dataset --- #
    all_data_info = pd.read_csv(os.path.join(input_dir, 'detector_data_complete.csv'))
    all_data_info = all_data_info.loc[SYN_DETECTOR_MAPPING[detector_name]]  # select the test data for the specific detector
    transforms = get_transform_list(detector_name)  # get the transforms for the specific detector
    # Select the data according to the test type and instantiate the dataset
    if test_type == 'real_vs_real-jpegai':
        all_data_info = all_data_info.loc['Pristine']
        data_info = all_data_info.loc[('Uncompressed', 'JPEG-AI')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
        dataset = JPEGAIDataset(root_dir=input_dir, data_df=data_info, transform=transforms)
    elif test_type == 'real_vs_real-jpeg':
        all_data_info = all_data_info.loc['Pristine']
        data_info = all_data_info.loc[('Uncompressed', 'JPEG')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
        dataset = JPEGDataset(root_dir=input_dir, data_df=data_info, transform=transforms)
    elif test_type == 'synthetic_vs_real':
        data_info = all_data_info.loc[pd.IndexSlice[:, 'Uncompressed']]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
        dataset = SynImgDataset(root_dir=input_dir, data_df=data_info, transform=transforms)
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count()//2)

    # --- Prepare the detector --- #
    detector = Detector(detector_name, weigths_paths, device=device)

    # --- Prepare the output dataframe --- #
    results = data_info.copy()
    results['logits'] = 0.0

    # --- PROCESS THE DATASET --- #
    count = 0
    logits_idx = results.columns.get_loc('logits')
    for image, label in tqdm(dataloader):
        # Send the data to device
        image = image.to(device)
        # Process the batch
        logits = detector.process_sample(image)
        # Save the results
        results.iloc[count:count+batch_size, logits_idx] = logits
        # Update the count
        count += batch_size

    # --- Save the results --- #
    output_dir = os.path.join(output_dir, detector_name)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, test_type)
    if debug:
        results.to_csv(save_path + '_debug.csv')
    elif test_all:
        results.to_csv(save_path+'_all_dataset.csv')
    else:
        results.to_csv(save_path+'.csv')


    return


# --- Main --- #
if __name__ == '__main__':

    # --- Parse the arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing the dataset",
                        default="/nas/public/exchange/JPEG-AI/data")
    parser.add_argument("--output_dir", type=str, help="Path to the output directory where to save the results",
                        default="./results")
    parser.add_argument('--batch_size', type=int, help="The batch size to use", default=1)
    parser.add_argument("--gpu", type=int, help="The GPU to use", default=0)
    parser.add_argument("--detector", type=str, help="The detector to use", default='Grag2021_progan',
                        choices=DETECTORS)
    parser.add_argument("--weights_path", type=str, help="The path to the weights of the detector",
                        default='./weights')
    parser.add_argument("--test_all", action='store_true',
                        help="Whether to test all datasets or only the ones used in the corresponding detector paper")
    parser.add_argument('--test_type', type=str, help="The type of test to perform", default='jpeg-ai_vs_real',
                        choices=['real_vs_real-jpegai', 'real_vs_real-jpeg', 'synthetic_vs_synthetic-jpegai',
                                 'synthetic_vs_synthetic-jpeg', 'synthetic_vs_real', 'real_vs_real-augmented',
                                 'synthetic_vs_synthetic-augmented'])
    parser.add_argument("--slack", action='store_true', help="Whether to send slack notifications")
    parser.add_argument('--debug', action='store_true', help="Whether to run in debug mode")
    args = parser.parse_args()

    # --- Call main --- #
    slack_m = ISPLSlack()
    try:
        if args.slack:
            slack_m.to_user(recipient='edo.cannas', message=f'Starting test for {args.detector}...')
        main(args)
    except Exception as e:
        if args.slack:
            slack_m.to_user(recipient='edo.cannas', message=f'Test for {args.detector} crashed! Error: {e}')
        print(f"Error: {e}")
        sys.exit(1)

    # --- Exit --- #
    if args.slack:
        slack_m.to_user(recipient='edo.cannas', message=f'Test for {args.detector} completed!')
    sys.exit(0)
    # TODO: test previous experiments to see if everything runs as before
