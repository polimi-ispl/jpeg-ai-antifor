"""
A simple script to test state-of-the-art detectors for synthetic image detection in the various cases considered in our
paper, i.e.:
1. analyzing pristine images;
2. analyzing pristine images compressed with JPEG AI;
3. analyzing pristine images compressed with JPEG;
4. analyzing pristine images with augmentations;
5. analyzing synthetic images;
6. analyzing synthetic images compressed with JPEG AI;
7. analyzing synthetic images compressed with JPEG;
8. analyzing synthetic images with augmentations.

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
from utils.data import get_transform_list, ImgDataset
from utils.detector import SynImgDetector
import pandas as pd

# --- Helpers functions and classes --- #

def run_test_case(test_type: str, input_dir: str, detector: SynImgDetector, device: torch.device,
                  all_data_info: pd.DataFrame, transforms: torch.nn.Module, batch_size: int, num_workers: int, debug: bool):

    # --- Select the data according to the test type and instantiate the dataset
    if test_type == 'real':
        data_info = all_data_info.loc[('Pristine', 'Uncompressed')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
    elif test_type == 'real_JPEGAI':
        data_info = all_data_info.loc[('Pristine', 'JPEG-AI')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
    elif test_type == 'real_JPEG':
        data_info = all_data_info.loc[('Pristine', 'JPEG')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
    elif test_type == 'real_doubleJPEGAI':
        data_info = all_data_info.loc[('Pristine', 'DoubleJPEG-AI')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
    elif test_type == 'real_aug':
        raise NotImplementedError('This case is not implemented yet!')
    elif test_type == 'synthetic':
        data_info = all_data_info.loc[('Synthetic', 'Uncompressed')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
    elif test_type == 'synthetic_JPEGAI':
        data_info = all_data_info.loc[('Synthetic', 'JPEG-AI')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
    elif test_type == 'synthetic_JPEG':
        data_info = all_data_info.loc[('Synthetic', 'JPEG')]
        if debug:
            data_info = data_info.loc['imagenet']
            data_info = data_info.iloc[:10]
    elif test_type == 'synthetic_aug':
        raise NotImplementedError('This case is not implemented yet!')
    # TODO: add the case for real and synthetic augmented images
    # Create the dataloader
    dataset = ImgDataset(root_dir=input_dir, data_df=data_info, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
        results.iloc[count:count + batch_size, logits_idx] = logits
        # Update the count
        count += batch_size

    return results

def main(args: argparse.Namespace):

    # --- Parse the params we need --- #
    input_dir = args.input_dir
    output_dir = args.output_dir
    gpu = args.gpu
    detector_name = args.detector
    weigths_paths = args.weights_path
    test_all = args.test_all
    test_case = args.test_case
    debug = args.debug
    batch_size = args.batch_size
    num_workers = args.num_workers

    # --- Prepare the device --- #
    device = torch.device(f'cuda:{gpu}') if torch.cuda.is_available() else torch.device('cpu')

    # --- Prepare the detector --- #
    detector = SynImgDetector(detector_name, weigths_paths, device=device)

    # --- Prepare the dataset --- #
    all_data_info = pd.read_csv(os.path.join(input_dir, 'detector_data_complete.csv'), index_col=[0, 1, 2, 3, 4])
    all_data_info = all_data_info.loc[SYN_DETECTOR_DATASET_MAPPING[detector_name]]  # select the test data for the specific detector
    transforms = get_transform_list(detector_name)  # get the transforms for the specific detector

    # --- Run the test --- #
    output_dir = os.path.join(output_dir, detector_name)
    os.makedirs(output_dir, exist_ok=True)
    if test_all:
        tests = ['real', 'real_JPEGAI', 'real_JPEG', 'real_aug', 'synthetic', 'synthetic_JPEGAI', 'synthetic_JPEG',
                 'synthetic_aug']
        for test_case in tests:
            # --- Prepare the save path --- #
            save_path = os.path.join(output_dir, test_case)
            if os.path.exists(save_path+'.csv'):
                print(f"Test case {test_case} already done, skipping...")
                continue
            try:
                results = run_test_case(test_type=test_case, input_dir=input_dir, detector=detector, device=device,
                              all_data_info=all_data_info, transforms=transforms, batch_size=batch_size,
                              num_workers=num_workers, debug=debug)
                # --- Save the results --- #
                if debug:
                    results.to_csv(save_path + '_debug.csv')
                else:
                    results.to_csv(save_path + '.csv')
            except Exception as e:
                print(f"Error in test case {test_case}: {e}")
                continue
    else:
        results = run_test_case(test_type=test_case, input_dir=input_dir, detector=detector, device=device,
                              all_data_info=all_data_info, transforms=transforms, batch_size=batch_size,
                              num_workers=num_workers, debug=debug)

        # --- Save the results --- #
        save_path = os.path.join(output_dir, test_case)
        if debug:
            results.to_csv(save_path + '_debug.csv')
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
    parser.add_argument("--num_workers", type=int, help="The number of workers to use", default=cpu_count()//2)
    parser.add_argument("--detector", type=str, help="The detector to use", default='Grag2021_progan',
                        choices=DETECTORS)
    parser.add_argument("--weights_path", type=str, help="The path to the weights of the detector",
                        default='./weights')
    parser.add_argument("--test_all", action='store_true',
                        help="Whether to test all datasets or only the ones used in the corresponding detector paper")
    parser.add_argument('--test_case', type=str, help="The type of test to perform", default='real',
                        choices=['real', 'real_JPEGAI', 'real_JPEG', 'real_doubleJPEGAI', 'real_aug',
                                 'synthetic', 'synthetic_JPEGAI', 'synthetic_JPEG', 'synthetic_aug'])
    parser.add_argument('--debug', action='store_true', help="Whether to run in debug mode")
    args = parser.parse_args()

    # --- Call main --- #
    try:
        main(args)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    # --- Exit --- #
    sys.exit(0)
