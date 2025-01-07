"""
Script for testing the splicing detector on a dataset.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries import
import os
import sys
import argparse
import torch
from tqdm import tqdm
from multiprocessing import cpu_count
from utils.data import get_transform_list, ImgSplicingDataset
from utils.slack import ISPLSlack
from utils.detector import ImgSplicingDetector
import pandas as pd
from PIL import Image
import numpy as np

# --- Helpers functions and classes --- #

def run_splicing_test(test_type: str, input_dir: str, save_path: str, detector: str, device: torch.device,
                     all_data_info: pd.DataFrame, transforms: torch.nn.Module, num_workers: int, debug: bool):

    # --- Select the data according to the test type and instantiate the dataset
    if test_type == 'uncompressed':
        data_info = all_data_info.loc['Uncompressed']
        if debug:
            data_info = data_info.loc['CASIA1']
            data_info = data_info.iloc[:10]
    elif test_type == 'jpegai':
        data_info = all_data_info.loc['JPEG-AI']
        if debug:
            data_info = data_info.loc['CASIA1']
            data_info = data_info.iloc[:10]
    elif test_type == 'jpeg':
        if 'JPEG' not in all_data_info.index:
            raise ValueError('No JPEG dataset available')
        else:
            data_info = all_data_info.loc['JPEG']
            if debug:
                data_info = data_info.loc['CASIA1']
                data_info = data_info.iloc[:10]
    else:
        raise ValueError('Unknown test type')
    # Create the dataloader
    dataset = ImgSplicingDataset(root_dir=input_dir, data_df=data_info, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # --- Prepare the results
    results = data_info.copy()
    results['mask_path'] = ''

    # --- PROCESS THE DATASET
    mask_path_idx = results.columns.get_loc('mask_path')
    for batch_idx, (image, _) in enumerate(tqdm(dataloader)):
        image = image.to(device)
        mask = detector.process_sample(image)
        # Save the mask as .npy file
        mask_save_path = os.path.join(save_path, results.iloc[batch_idx]['dataset'],
                                        f"{results.iloc[batch_idx]['filename']}_mask.npy")
        os.makedirs(os.path.dirname(mask_save_path), exist_ok=True)
        np.save(mask_save_path, mask)
        results.iloc[batch_idx, mask_path_idx] = mask_save_path

    return results

def main(args: argparse.Namespace):

    # --- Parse the parameters we need --- #
    input_dir = args.input_dir
    output_dir = args.output_dir
    gpu = args.gpu
    detector_name = args.detector
    weights_path = args.weights_path
    test_all = args.test_all
    test_type = args.test_type
    debug = args.debug
    num_workers = args.num_workers

    # --- Prepare the device --- #
    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # --- Prepare the detector --- #
    detector = ImgSplicingDetector(detector_name, weights_path, device=device)

    # --- Prepare the dataset --- #
    all_data_info = pd.read_csv(os.path.join(input_dir, 'spliced_data_complete.csv'), index_col=[0, 1, 2])
    transforms = get_transform_list(detector_name)

    # --- Run the test --- #
    output_dir = os.path.join(output_dir, detector_name)
    os.makedirs(output_dir, exist_ok=True)
    if test_all:
        tests = ['uncompressed', 'jpegai', 'jpeg']
        for test_case in tests:
            # --- Prepare the save path --- #
            save_path = os.path.join(output_dir, test_case)
            os.makedirs(save_path, exist_ok=True)
            # --- Check that the results directory is not empty --- #
            if os.path.exists(os.path.join(save_path, 'results.csv')):
                print(f"Test case {test_case} already done, skipping...")
                continue
            # --- Run the test --- #
            try:
                results = run_splicing_test(test_case, input_dir, save_path, detector, device, all_data_info, transforms,
                                            num_workers, debug)
                # --- Save the results --- #
                if debug:
                    results.to_csv(os.path.join(save_path, 'results_debug.csv'))
                else:
                    results.to_csv(os.path.join(save_path, 'results.csv'))
            except Exception as e:
                print(f"Error in test case {test_case}: {e}")
                continue
    else:
        # --- Run the test --- #
        results = run_splicing_test(test_type, input_dir, detector, device, all_data_info, transforms, num_workers, debug)
        # --- Save the results --- #
        save_path = os.path.join(output_dir, test_type)
        os.makedirs(save_path, exist_ok=True)
        if debug:
            results.to_csv(os.path.join(save_path, 'results_debug.csv'))
        else:
            results.to_csv(os.path.join(save_path, 'results.csv'))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Path to the input directory containing the dataset",
                        default="/nas/public/exchange/JPEG-AI/data")
    parser.add_argument('--output_dir', type=str, help="Path to the output directory where to save the results",
                        default="./results")
    parser.add_argument("--gpu", type=int, help="The GPU to use", default=0)
    parser.add_argument("--num_workers", type=int, help="The number of workers to use", default=cpu_count()//2)
    parser.add_argument("--detector", type=str, help="The detector to use", default='TruFor')
    parser.add_argument("--weights_path", type=str, help="The path to the weights of the detector",
                        default='./weights')
    parser.add_argument("--test_all", action='store_true',
                        help="Whether to test all datasets or only the ones used in the corresponding detector paper")
    parser.add_argument('--test_type', type=str, help="The type of test to perform", default='real',
                        choices=['Uncompressed', 'JPEGAI', 'JPEG'])
    parser.add_argument('--debug', action='store_true', help="Whether to run in debug mode")
    args = parser.parse_args()

    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print('Test completed successfully!')
    sys.exit(0)