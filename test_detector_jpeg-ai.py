"""
A simple script to test state-of-the-art detectors for synthetic image detection on our dataset of JPEG-AI compressed samples.

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
from utils.data import JPEGAIDataset, get_transform_list
from utils.slack import ISPLSlack
from utils.detector import SynImgDetector
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
    debug = args.debug
    batch_size = args.batch_size

    # --- Prepare the device --- #
    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # --- Prepare the dataset --- #
    all_data_info = pd.read_csv(os.path.join(input_dir, 'data_info.csv'))
    # The dataframe has two columns, 'jpeg-ai_compressed' == True if the image is compressed with JPEG-AI
    # and 'jpeg_compressed' == True if the image is compressed with JPEG
    # We need to select the rows where 'jpeg-ai_compressed' == True and ('jpeg-ai_compressed'==False and 'jpeg_compressed'==False)
    # for selecting the uncompressed samples
    jpeg_ai_info = all_data_info.loc[all_data_info['jpeg-ai_compressed']]
    uncompressed_info = all_data_info.loc[(~all_data_info['jpeg-ai_compressed']) & (~all_data_info['jpeg_compressed'])]
    data_info = pd.concat([jpeg_ai_info, uncompressed_info])
    if not test_all:
        data_info = data_info.loc[data_info['dataset'].isin(COMPRESSED_TEST_DATA[detector_name])]
    if debug:
        data_info = data_info.loc[data_info['dataset']=='imagenet']
        data_info = data_info.iloc[:10]
    transforms = get_transform_list(detector_name)
    dataset = JPEGAIDataset(root_dir=input_dir, data_df=data_info, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=cpu_count()//2)

    # --- Prepare the detector --- #
    detector = SynImgDetector(detector_name, weigths_paths, device=device)

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
    save_path = os.path.join(output_dir, 'results_jpeg-ai_vs_real')
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
                        default="/nas/public/exchange/JPEG-AI/data/TEST")
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
