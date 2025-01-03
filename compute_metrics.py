"""
Simple script to compute the metrics on the scores distribution of the detectors.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
from utils.metrics import compute_all_metrics
import pandas as pd
import argparse
import sys
import os
import glob

# --- Functions --- #
def main(args: argparse.Namespace):

    # --- Parse the arguments --- #
    results_dir = args.input_dir
    detector = args.detector
    debug = args.debug

    # --- Load the data --- #

    # Pristine ("real") samples
    results_pristine = pd.read_csv(os.path.join(results_dir, detector, 'real.csv'), index_col=[0, 1])
    results_pristine_jpegai = pd.read_csv(os.path.join(results_dir, detector, 'real_JPEGAI.csv'), index_col=[0, 1])
    results_pristine_jpeg = pd.read_csv(os.path.join(results_dir, detector, 'real_JPEG.csv'), index_col=[0, 1])
    # Synthetic samples
    results_synthetic = pd.read_csv(os.path.join(results_dir, detector, 'synthetic.csv'), index_col=[0, 1])
    results_synthetic_jpegai = pd.read_csv(os.path.join(results_dir, detector, 'synthetic_JPEGAI.csv'), index_col=[0, 1])
    results_synthetic_jpeg = pd.read_csv(os.path.join(results_dir, detector, 'synthetic_JPEG.csv'), index_col=[0, 1])

    # --- Compute the metrics --- #
    metrics = {}

    # --- Considering all images together
    dataset_name = 'All_images'
    metrics[dataset_name] = {}
    metrics[dataset_name]['All'] = []

    # Pristine test cases
    metrics[dataset_name]['All'].append(
        pd.DataFrame([compute_all_metrics(results_pristine, results_pristine_jpegai)],
                     index=pd.Index(['Real_vs_Real-JPEGAI']),
                     columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
    metrics[dataset_name]['All'].append(pd.DataFrame([compute_all_metrics(results_pristine, results_pristine_jpeg)],
                                                        index=pd.Index(['Real_vs_Real-JPEG']),
                                                        columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
    # Synthetic test cases
    metrics[dataset_name]['All'].append(
        pd.DataFrame([compute_all_metrics(results_synthetic, results_synthetic_jpegai)],
                     index=pd.Index(['Synth_vs_Synth-JPEGAI']),
                     columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
    metrics[dataset_name]['All'].append(
        pd.DataFrame([compute_all_metrics(results_synthetic, results_synthetic_jpeg)],
                     index=pd.Index(['Synth_vs_Synth-JPEG']),
                     columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
    # Pristine VS Synthetic test cases
    metrics[dataset_name]['All'].append(pd.DataFrame([compute_all_metrics(results_pristine, results_synthetic)],
                                                        index=pd.Index(['Real_vs_Synth']),
                                                        columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
    metrics[dataset_name]['All'].append(
        pd.DataFrame([compute_all_metrics(results_pristine_jpeg, results_synthetic_jpeg)],
                     index=pd.Index(['Real-JPEG_vs_Synth-JPEG']),
                     columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
    metrics[dataset_name]['All'].append(
        pd.DataFrame([compute_all_metrics(results_pristine_jpegai, results_synthetic_jpegai)],
                     index=pd.Index(['Real-JPEGAI_vs_Synth-JPEGAI']),
                     columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
    metrics[dataset_name]['All'] = pd.concat(metrics[dataset_name]['All'])

    # Divide by quality factor and target_bpp
    for idx, target_bpp in enumerate(results_pristine_jpegai['target_bpp'].unique()):

        part_results = []

        # Pristine test cases
        dataset = results_pristine_jpegai.loc[results_pristine_jpegai['target_bpp'] == target_bpp]
        part_results.append(pd.DataFrame([compute_all_metrics(results_pristine, dataset)],
                                         index=pd.Index(['Real_vs_Real-JPEGAI']),
                                         columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        # Synthetic test cases
        dataset = results_synthetic_jpegai.loc[results_synthetic_jpegai['target_bpp'] == target_bpp]
        part_results.append(pd.DataFrame([compute_all_metrics(results_synthetic, dataset)],
                                         index=pd.Index(['Synth_vs_Synth-JPEGAI']),
                                         columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        # Pristine VS Synthetic test cases
        dataset_pristine = results_pristine_jpegai.loc[results_pristine_jpegai['target_bpp'] == target_bpp]
        dataset_syn = results_synthetic_jpegai.loc[results_synthetic_jpegai['target_bpp'] == target_bpp]
        part_results.append(pd.DataFrame([compute_all_metrics(dataset_pristine, dataset_syn)],
                                         index=pd.Index(['Real-JPEGAI_vs_Synth-JPEGAI']),
                                         columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        metrics[dataset_name][target_bpp] = pd.concat(part_results)
    for idx, qf in enumerate(results_pristine_jpeg['qf'].dropna().unique()):

        part_results = []

        # Pristine test cases
        dataset = results_pristine_jpeg.loc[results_pristine_jpeg['qf'] == qf]
        part_results.append(pd.DataFrame([compute_all_metrics(results_pristine, dataset)],
                                         index=pd.Index(['Real_vs_Real-JPEG']),
                                         columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        # Synthetic test cases
        dataset = results_synthetic_jpeg.loc[results_synthetic_jpeg['qf'] == qf]
        part_results.append(pd.DataFrame([compute_all_metrics(results_synthetic, dataset)],
                                         index=pd.Index(['Synth_vs_Synth-JPEG']),
                                         columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        # Pristine VS Synthetic test cases
        dataset_pristine = results_pristine_jpeg.loc[results_pristine_jpeg['qf'] == qf]
        dataset_syn = results_synthetic_jpeg.loc[results_synthetic_jpeg['qf'] == qf]
        part_results.append(pd.DataFrame([compute_all_metrics(dataset_pristine, dataset_syn)],
                                         index=pd.Index(['Real-JPEG_vs_Synth-JPEG']),
                                         columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        metrics[dataset_name][qf] = pd.concat(part_results)

    # --- Considering the single datasets separately
    for dataset_name in results_pristine.index.get_level_values(0).unique():

        # Prepare the dict
        metrics[dataset_name] = {}

        # Consider all the images together
        metrics[dataset_name]['All'] = []

        # Pristine test cases
        metrics[dataset_name]['All'].append(pd.DataFrame(
            [compute_all_metrics(results_pristine.loc[dataset_name], results_pristine_jpegai.loc[dataset_name])],
            index=pd.Index(['Real_vs_Real-JPEGAI']),
            columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        metrics[dataset_name]['All'].append(pd.DataFrame(
            [compute_all_metrics(results_pristine.loc[dataset_name], results_pristine_jpeg.loc[dataset_name])],
            index=pd.Index(['Real_vs_Real-JPEG']),
            columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        # Synthetic test cases
        metrics[dataset_name]['All'].append(pd.DataFrame(
            [compute_all_metrics(results_synthetic.loc[dataset_name], results_synthetic_jpegai.loc[dataset_name])],
            index=pd.Index(['Synth_vs_Synth-JPEGAI']),
            columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        metrics[dataset_name]['All'].append(pd.DataFrame(
            [compute_all_metrics(results_synthetic.loc[dataset_name], results_synthetic_jpeg.loc[dataset_name])],
            index=pd.Index(['Synth_vs_Synth-JPEG']),
            columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        # Pristine VS Synthetic test cases
        metrics[dataset_name]['All'].append(
            pd.DataFrame([compute_all_metrics(results_pristine.loc[dataset_name], results_synthetic.loc[dataset_name])],
                         index=pd.Index(['Real_vs_Synth']),
                         columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        metrics[dataset_name]['All'].append(pd.DataFrame(
            [compute_all_metrics(results_pristine_jpeg.loc[dataset_name], results_synthetic_jpeg.loc[dataset_name])],
            index=pd.Index(['Real-JPEG_vs_Synth-JPEG']),
            columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        metrics[dataset_name]['All'].append(pd.DataFrame([compute_all_metrics(
            results_pristine_jpegai.loc[dataset_name], results_synthetic_jpegai.loc[dataset_name])],
                                                            index=pd.Index(['Real-JPEGAI_vs_Synth-JPEGAI']),
                                                            columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
        # Concatenate everything
        metrics[dataset_name]['All'] = pd.concat(metrics[dataset_name]['All'])

        # Divide by quality factor and target_bpp
        for idx, target_bpp in enumerate(results_pristine_jpegai['target_bpp'].unique()):

            part_results = []

            # Pristine test cases
            dataset = results_pristine_jpegai.loc[results_pristine_jpegai['target_bpp'] == target_bpp]
            part_results.append(
                pd.DataFrame([compute_all_metrics(results_pristine.loc[dataset_name], dataset.loc[dataset_name])],
                             index=pd.Index(['Real_vs_Real-JPEGAI']),
                             columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
            # Synthetic test cases
            dataset = results_synthetic_jpegai.loc[results_synthetic_jpegai['target_bpp'] == target_bpp]
            part_results.append(
                pd.DataFrame([compute_all_metrics(results_synthetic.loc[dataset_name], dataset.loc[dataset_name])],
                             index=pd.Index(['Synth_vs_Synth-JPEGAI']),
                             columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
            # Pristine VS Synthetic test cases
            dataset_pristine = results_pristine_jpegai.loc[results_pristine_jpegai['target_bpp'] == target_bpp]
            dataset_syn = results_synthetic_jpegai.loc[results_synthetic_jpegai['target_bpp'] == target_bpp]
            part_results.append(
                pd.DataFrame([compute_all_metrics(dataset_pristine.loc[dataset_name], dataset_syn.loc[dataset_name])],
                             index=pd.Index(['Real-JPEGAI_vs_Synth-JPEGAI']),
                             columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
            metrics[dataset_name][target_bpp] = pd.concat(part_results)
        for idx, qf in enumerate(results_pristine_jpeg['qf'].dropna().unique()):

            part_results = []
            # Pristine test cases
            dataset = results_pristine_jpeg.loc[results_pristine_jpeg['qf'] == qf]
            part_results.append(
                pd.DataFrame([compute_all_metrics(results_pristine.loc[dataset_name], dataset.loc[dataset_name])],
                             index=pd.Index(['Real_vs_Real-JPEG']),
                             columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
            # Synthetic test cases
            dataset = results_synthetic_jpeg.loc[results_synthetic_jpeg['qf'] == qf]
            part_results.append(
                pd.DataFrame([compute_all_metrics(results_synthetic.loc[dataset_name], dataset.loc[dataset_name])],
                             index=pd.Index(['Synth_vs_Synth-JPEG']),
                             columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
            # Pristine VS Synthetic test cases
            dataset_pristine = results_pristine_jpeg.loc[results_pristine_jpeg['qf'] == qf]
            dataset_syn = results_synthetic_jpeg.loc[results_synthetic_jpeg['qf'] == qf]
            part_results.append(
                pd.DataFrame([compute_all_metrics(dataset_pristine.loc[dataset_name], dataset_syn.loc[dataset_name])],
                             index=pd.Index(['Real-JPEG_vs_Synth-JPEG']),
                             columns=['WD', 'auc', 'fpr_thr0', 'tpr_thr0', 'ba_thr0']))
            metrics[dataset_name][qf] = pd.concat(part_results)

    # Create the final results Dataframe
    metrics = pd.concat(
        [pd.concat({key: pd.concat(metrics[key])}, names=['Dataset', 'Quality', 'Test']) for key in
         metrics.keys()])

    # --- Save the metrics --- #
    metrics_df = pd.concat({detector: metrics}, names=['Detector', 'Dataset', 'Quality', 'Test'])
    metrics_df.to_csv(os.path.join(results_dir, 'metrics.csv'), index=False)

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

