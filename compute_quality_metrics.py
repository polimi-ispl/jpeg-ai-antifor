"""
Small script to compute a quality report of the compressed samples using the JPEG-AI Vertification Model suite.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
import glob
import os
import sys
import pandas as pd
from tqdm import tqdm
import argparse
from utils.slack import ISPLSlack
from PIL import Image

# --- Functions --- #

def main(args: argparse.Namespace):

    # --- Parse the arguments --- #
    input_dir = args.input_dir
    output_dir = args.output_dir
    gpu = args.gpu
    jpeg_ai_path = args.jpeg_ai_path
    debug = args.debug
    test_jpegai = args.test_jpegai
    test_jpeg = args.test_jpeg

    # --- Prepare the metric processor --- #
    sys.path.append(jpeg_ai_path) # Add the jpeg-ai-reference-software to the path
    from src.codec.metrics.metrics import DataClass, MetricsProcessor, MetricsFabric  # Import the necessary classes
    metrics = MetricsProcessor()
    metrics.internal_bits = 10
    metrics.jvet_psnr = False
    metrics.metrics = MetricsFabric.metrics_list
    metrics.metrics_output = [metric for metric in MetricsFabric.metrics_list]
    metrics.color_conv = '709'
    metrics.max_samples_for_eval_on_gpu = -1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # --- Load the dataset --- #
    if os.path.exists(os.path.join(input_dir, 'data_info.csv')):
        # Load the CSV directly
        all_images = pd.read_csv(os.path.join(input_dir, 'data_info.csv'))
    else:
        # Look for all the images inside the directory (avoid binary files)
        all_images = pd.DataFrame(
            [path for path in glob.glob(os.path.join(input_dir, '**/*.*'), recursive=True)
             if (('ipynb' not in path) and ('pdf' not in path) and ('txt' not in path))],
            columns=['path'])
        # add other useful info
        all_images['dataset'] = all_images['path'].apply(lambda x: x.split('/')[7])
        all_images['jpeg-ai_compressed'] = all_images['path'].apply(lambda x: True if 'target_bpp' in x else False)
        all_images['jpeg_compressed'] = all_images['path'].apply(lambda x: True if 'qf' in x else False)
        all_images['target_bpp'] = all_images['path'].apply(
            lambda x: x.split('target_bpp_')[1].split('/')[0] if 'target_bpp' in x else None)
        all_images['target_bpp'] = all_images['target_bpp'].apply(lambda x: float(x) / 100 if x is not None else None)
        all_images['qf'] = all_images['path'].apply(lambda x: x.split('qf_')[1].split('/')[0] if 'qf_' in x else None)
        all_images['qf'] = all_images['qf'].apply(lambda x: int(x) if x is not None else None)
        all_images['filename'] = all_images['path'].apply(lambda x: os.path.basename(x))
        content_dict = {'imagenet': 'various', 'celeba': 'faces', 'ffhq': 'faces', 'coco': 'various',
                        'raise': 'various', 'laion': 'various', 'raw_camera': 'various'}
        all_images['content'] = all_images['dataset'].apply(lambda x: content_dict[x] if 'lsun' not in x else None)
        # Fix LSUN
        for i, r in all_images.loc[all_images['dataset'] == 'lsun'].iterrows():
            if r['jpeg-ai_compressed'] or r['jpeg_compressed']:
                all_images.loc[i, 'content'] = r['path'].split('/')[10]
            else:
                all_images.loc[i, 'content'] = r['path'].split('/')[9]
        # Let's add info on the single image sizes
        all_images['size'] = all_images['path'].apply(lambda x: Image.open(x).size)
        # Let's save the data into a csv
        all_images.to_csv(os.path.join(input_dir, 'data_info.csv'), index=False)


    # --- Compute the metrics --- #
    dataset_list = []
    for dataset in all_images['dataset'].unique():
        print(f'Doing dataset {dataset}...')
        # select the images from the dataset and according to the test type
        images_df = all_images.loc[all_images['dataset'] == dataset]
        if test_jpegai:
            orig_df = images_df.loc[(images_df['jpeg-ai_compressed']==False) & (images_df['jpeg_compressed']==False)]
            compr_df = images_df.loc[images_df['jpeg-ai_compressed']]
        elif test_jpeg:
            orig_df = images_df.loc[
                (images_df['jpeg-ai_compressed'] == False) & (images_df['jpeg_compressed'] == False)]
            compr_df = images_df.loc[images_df['jpeg_compressed']]
        if debug:
            orig_df = orig_df.iloc[:2]

        # Create the metrics Dataframe
        images_list = []

        # TODO: DEBUGGING
        # Cycle over the different pristine samples
        for i, r in tqdm(orig_df.iterrows()):
            # Load the original sample
            filename, dataset, content = r['filename'], r['dataset'], r['content']
            orig_sample, _ = DataClass().load_image(r['path'], color_conv='709', device='cuda')
            # Find the corresponding compressed samples
            if test_jpegai:
                comp_samples = compr_df.loc[(compr_df['filename'] == filename.replace('jpg', 'png')) \
                                            & (compr_df['dataset'] == dataset) \
                                            & (compr_df['content'] == content)]
                # Cycle over the various BPPs values
                bpps_dict = []
                for ii, rr in comp_samples.iterrows():
                    # Load the compressed samples
                    comp_sample, _ = DataClass().load_image(rr['path'], color_conv='709', device='cuda')
                    # Compute the metrics
                    metrics_vals = metrics.process_images(orig_sample, comp_sample)
                    # Save them
                    bpps_dict.append(pd.DataFrame.from_dict({rr['target_bpp']: {metric: metrics_vals[idx] for idx, metric in
                                                                                enumerate(metrics.metrics_output)}},
                                                            orient='index'))

                # Append the image path to the Dataframe and save it
                images_list.append(pd.concat({r['path']: pd.concat(bpps_dict)}, names=['path', 'bpp']))
            else:
                comp_samples = compr_df.loc[(compr_df['filename'] == filename.replace('png', 'jpg')) \
                                            & (compr_df['dataset'] == dataset) \
                                            & (compr_df['content'] == content)]
                # Cycle over the various BPPs values
                bpps_dict = []
                for ii, rr in comp_samples.iterrows():
                    # Load the compressed samples
                    comp_sample, _ = DataClass().load_image(rr['path'], color_conv='709', device='cuda')
                    # Compute the metrics
                    metrics_vals = metrics.process_images(orig_sample, comp_sample)
                    # Save them
                    bpps_dict.append(
                        pd.DataFrame.from_dict({rr['qf']: {metric: metrics_vals[idx] for idx, metric in
                                                                   enumerate(metrics.metrics_output)}},
                                               orient='index'))

                # Append the image path to the Dataframe and save it
                images_list.append(pd.concat({r['path']: pd.concat(bpps_dict)}, names=['path', 'qf']))

        # Append the images dictionary
        dataset_list.append(pd.concat({dataset: pd.concat(images_list)}, names=['dataset', 'path', 'bpp' if test_jpegai else 'qf']))

    # Save the results
    os.makedirs(output_dir, exist_ok=True)
    if debug:
        pd.concat(dataset_list).to_csv(os.path.join(output_dir, 'quality_report_debug.csv'))
    else:
        if test_jpegai:
            pd.concat(dataset_list).to_csv(os.path.join(output_dir, 'quality_report_jpegai.csv'))
        elif test_jpeg:
            pd.concat(dataset_list).to_csv(os.path.join(output_dir, 'quality_report_jpeg.csv'))
    return


if __name__ == '__main__':

    # --- Arguments --- #
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, help='Path to the directory containing the dataset',
                        default='/nas/public/exchange/JPEG-AI/data/TEST')
    parser.add_argument('--output_dir', type=str, help='Path to the directory where the results will be saved',
                        default='./quality_report')
    parser.add_argument('--gpu', type=int, help='Device to use for the computation', default=3)
    parser.add_argument('--jpeg_ai_path', type=str, help='Path to the jpeg-ai-reference-software',
                        default='/nas/home/ecannas/third_party_code/jpeg-ai-reference-software')
    parser.add_argument('--debug', action='store_true', help='Debug mode (take only 10 samples for dataset)')
    # Add the test option in a mutually exclusive group
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--test_jpegai', action='store_true', help='Test only the JPEG-AI compressed samples')
    test_group.add_argument('--test_jpeg', action='store_true', help='Test only the JPEG compressed samples')

    # --- Run main --- #
    args = parser.parse_args()
    slack_m = ISPLSlack()
    try:
        slack_m.to_user(recipient='edo.cannas', message='Quality report started...')
        main(args)
    except Exception as e:
        slack_m.to_user(recipient='edo.cannas', message=f'Quality report failed. Error {e}')
        raise e

    # --- Exit --- #
    slack_m.to_user(recipient='edo.cannas', message='Quality report completed!')
    sys.exit(0)