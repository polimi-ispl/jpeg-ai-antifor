"""
Simple script to compute splicing localization metrics on the masks returned by the detectors.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
from utils.metrics import computeLocalizationMetrics, computeDetectionMetrics
import pandas as pd
import argparse
import sys
import os
import glob
from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm

# --- Functions --- #
def process_map(row, path):
    # Load everything
    gt_path = row['gt']
    map_path = row['mask_path']
    dataset = row['dataset']
    if not os.path.exists(gt_path) or not os.path.exists(map_path):
        print(f'Missing file for sample {path}, skipping this...')
        return path, None
    gt = Image.open(gt_path)
    out_map = np.load(map_path)
    if (gt.size[1], gt.size[0]) != out_map.shape:
        print(f'Mismatch at sample {path}, skipping this...')
        return path, None
    # GT "has to be 0 for pristine pixels and 1 for forged pixels."
    if dataset == 'DSO-1':
        gt = np.array(gt.convert('L')) < 0.1  # DSO-1 is inverted
    elif dataset == 'Columbia':
        gt = np.array(gt.convert('L')) > 127
    else:
        gt = np.array(gt.convert('L')).astype(bool)

    # Compute the metrics
    F1_best, F1_th, FPR_05 = computeLocalizationMetrics(out_map, gt)
    AUC, bACC, bACC_best = computeDetectionMetrics(out_map, gt)

    # Return the results
    return path, {'AUC': AUC, 'BA@0.5': bACC, 'BA best': bACC_best, 'F1@0.5': F1_th, 'F1 best': F1_best,
                      'FPR@0.5': FPR_05}

def main(args: argparse.Namespace):

    # --- Parse the arguments --- #
    results_dir = args.input_dir
    detector = args.detector
    debug = args.debug

    # --- Load the results --- #
    all_results = []
    for test_type in ['Uncompressed', 'JPEG', 'JPEGAI', 'Double JPEGAI']:
        results = pd.read_csv(os.path.join(results_dir, detector, test_type, 'results.csv'), index_col=[0, 1])
        all_results.append(pd.concat({test_type: results}, names=['Test', 'Dataset', 'Path']))
    all_results = pd.concat(all_results)
    #all_results.drop('Uncompressed', inplace=True)
    # all_results.drop('JPEG', inplace=True)

    # --- Compute the metrics --- #
    metrics = {}

    # --- Divide by test
    for test in all_results.index.get_level_values('Test').unique():
        test_results = all_results.loc[test]
        metrics[test] = {}

        # --- Divide by dataset
        for dataset in test_results.index.get_level_values('Dataset').unique():
            dataset_results = test_results.loc[dataset]
            metrics[test][dataset] = {}

            if test in ['JPEG', 'JPEGAI', 'Double JPEGAI']:
                # --- Divide by quality
                quality_col = 'qf' if test == 'JPEG' else 'target_bpp'
                for quality in dataset_results[quality_col].unique():
                    # --- Select the results for the specific quality --- #
                    quality_results = dataset_results.loc[dataset_results[quality_col]==quality]
                    paths = quality_results.index.get_level_values('Path').unique()
                    metrics_df = pd.DataFrame(index=paths, columns=['AUC', 'BA@0.5', 'BA best', 'F1@0.5', 'F1 best', 'FPR@0.5'])
                    metrics[test][dataset][quality] = {}

                    # --- Compute the metrics in parallel --- #
                    with concurrent.futures.ThreadPoolExecutor(os.cpu_count() // 2) as executor:
                        futures = {executor.submit(process_map, row, i): row
                                   for i, row in quality_results.iterrows()}
                        for future in tqdm(concurrent.futures.as_completed(futures)):
                            path, computed_metrics = future.result()
                            if computed_metrics is not None:
                                for key, value in computed_metrics.items():
                                    metrics_df.loc[path, key] = value

                    # Save the results
                    metrics[test][dataset][quality] = metrics_df
            else:
                paths = dataset_results.index.get_level_values('Path').unique()
                metrics_df = pd.DataFrame(index=paths,
                                          columns=['AUC', 'BA@0.5', 'BA best', 'F1@0.5', 'F1 best', 'FPR@0.5'])
                metrics[test][dataset]['NaN'] = {}

                # --- Compute the metrics in parallel --- #
                with concurrent.futures.ThreadPoolExecutor(os.cpu_count() // 2) as executor:
                    futures = {executor.submit(process_map, row, i): row
                               for i, row in dataset_results.iterrows()}
                    for future in tqdm(concurrent.futures.as_completed(futures)):
                        path, computed_metrics = future.result()
                        if computed_metrics is not None:
                            for key, value in computed_metrics.items():
                                metrics_df.loc[path, key] = value

                # Save the results
                metrics[test][dataset]['NaN'] = metrics_df

    # Create the final results DataFrame with 4 levels: Test, Dataset, Quality, and Path
    metrics_df = pd.concat({test: pd.concat({dataset: pd.concat({quality: df for quality, df in dataset_dict.items()})
                                          for dataset, dataset_dict in test_dict.items()})
                            for test, test_dict in metrics.items()})

    # --- Save the metrics --- #
    metrics_df.to_csv(os.path.join(results_dir, detector, 'splicing_localization_metrics.csv'))

    if debug:
        print(metrics_df)


if __name__ == '__main__':
    # --- Parse the arguments --- #
    parser = argparse.ArgumentParser(description='Compute the metrics on the scores distribution of the detectors.')
    parser.add_argument('--input_dir', type=str, help='Directory containing the results of the detectors.',
                        default='./results')
    parser.add_argument('--detector', type=str, help='Name of the detector.', required=True)
    parser.add_argument('--debug', action='store_true', help='If set, print the results.')
    args = parser.parse_args()

    # --- Call the main --- #
    main(args)

