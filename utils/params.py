"""
Constant values used throughout the codebase.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

JPEG_AI_PATH = '/nas/home/ecannas/third_party_code/jpeg-ai-reference-software'  # put here the path to the jpeg-ai-reference-software

PRISTINE_ROOT_DIR = '/nas/public/exchange/JPEG-AI/data/TEST'

SYNTHETIC_ROOT_DIR = '/nas/public/exchange/JPEG-AI/data/TEST_SYN'

MODELS_LIST = {'Grag2021_progan': 'Grag2021_progan',
               'Grag2021_latent': 'Grag2021_latent',
               'Ojha2023': 'Ojha2023',
               'Ojha2023ResNet50': 'Ojha2023ResNet50',
               'Cozzolino2024-A': 'clipdet_latent10k',
               'Cozzolino2024-B': 'clipdet_latent10k_plus',
               'Corvi2023': 'Corvi2023',
               'Wang2020-A': 'blur_jpg_prob0.1.pth',
               'Wang2020-B': 'blur_jpg_prob0.5.pth',
               'NPR': 'NPR.pth',
               'Mandelli2024': 'synth_vs_real.pth',
               'Mandelli2024-RT': 'retrained.pth',
               'Mandelli2024-FT': 'finetuned.pth',}

COMPRESSED_TEST_DATA = {'Grag2021_progan': ['imagenet', 'coco', 'ffhq'],
             'Grag2021_latent': ['imagenet', 'coco'],
             'Ojha2023': ['imagenet', 'coco', 'lsun', 'laion', 'raw_camera', 'celeba'],
             'Ojha2023ResNet50': ['imagenet', 'coco', 'lsun', 'laion', 'raw_camera', 'celeba'],
             'Cozzolino2024-A': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'Cozzolino2024-B': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'Corvi2023': ['imagenet', 'coco', 'ffhq'],
             'Wang2020-A': ['lsun', 'imagenet', 'coco', 'celeba', 'raw_camera'],
             'Wang2020-B': ['lsun', 'imagenet', 'coco', 'celeba', 'raw_camera'],
             'NPR': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'Mandelli2024': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'celeba'],
             'Mandelli2024-RT': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'celeba'],
             'Mandelli2024-FT': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'celeba']}

SYN_TEST_DATA = {'Grag2021_progan': ['imagenet', 'coco', 'ffhq'],
             'Grag2021_latent': ['imagenet', 'coco'],
             'Ojha2023': ['imagenet', 'coco', 'lsun', 'laion', 'raw_camera', 'celeba'],
             'Ojha2023ResNet50': ['imagenet', 'coco', 'lsun', 'laion', 'raw_camera', 'celeba'],
             'Cozzolino2024-A': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'Cozzolino2024-B': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'Corvi2023': ['imagenet', 'coco', 'ffhq'],
             'Wang2020-A': ['lsun', 'imagenet', 'coco', 'celeba', 'raw_camera'],
             'Wang2020-B': ['lsun', 'imagenet', 'coco', 'celeba', 'raw_camera'],
             'NPR': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'Mandelli2024': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'celeba'],
             'Mandelli2024-RT': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'celeba'],
             'Mandelli2024-FT': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'celeba']}

SYN_DETECTOR_DATASET_MAPPING = {'Grag2021_progan': 'Corvi2023',
                             'Grag2021_latent': 'Corvi2023',
                             'Ojha2023': 'Ojha2023',
                             'Ojha2023ResNet50': 'Ojha2023',
                             'Cozzolino2024-A': 'Cozzolino2024',
                             'Cozzolino2024-B': 'Cozzolino2024',
                             'Corvi2023': 'Corvi2023',
                             'Wang2020-A': 'Wang2020',
                             'Wang2020-B': 'Wang2020',
                             'NPR': 'Ojha2023',
                             'Mandelli2024': 'Mandelli2024',
                             'Mandelli2024-RT': 'Mandelli2024-RT',
                             'Mandelli2024-FT': 'Mandelli2024'}

SYN_DETECTORS = ['Grag2021_progan', 'Grag2021_latent', 'Ojha2023', 'Ojha2023ResNet50',
                 'Cozzolino2024-A', 'Cozzolino2024-B', 'Corvi2023', 'Wang2020-A', 'Wang2020-B',
                 'NPR', 'Mandelli2024', 'Mandelli2024-RT', 'Mandelli2024-FT']

SPLICING_DETECTORS = ['TruFor', 'ImageForensicsOSN', 'MMFusion']
