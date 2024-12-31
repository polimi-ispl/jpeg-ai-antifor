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
               'Ohja2023': 'Ohja2023',
               'Ohja2023ResNet50': 'Ohja2023ResNet50',
               'CLIP2024': 'clipdet_latent10k',
               'CLIP2024Plus': 'clipdet_latent10k_plus',
               'Corvi2023': 'Corvi2023',
               'Wang2020JPEG01': 'blur_jpg_prob0.1.pth',
               'Wang2020JPEG05': 'blur_jpg_prob0.5.pth'}

COMPRESSED_TEST_DATA = {'Grag2021_progan': ['imagenet', 'coco', 'ffhq'],
             'Grag2021_latent': ['imagenet', 'coco'],
             'Ohja2023': ['imagenet', 'coco', 'lsun', 'laion', 'raw_camera', 'celeba'],
             'Ohja2023ResNet50': ['imagenet', 'coco', 'lsun', 'laion', 'raw_camera', 'celeba'],
             'CLIP2024': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'CLIP2024Plus': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'Corvi2023': ['imagenet', 'coco', 'ffhq'],
             'Wang2020JPEG01': ['lsun', 'imagenet', 'coco', 'celeba', 'raw_camera'],
             'Wang2020JPEG05': ['lsun', 'imagenet', 'coco', 'celeba', 'raw_camera']}

SYN_TEST_DATA = {'Grag2021_progan': ['imagenet', 'coco', 'ffhq'],
             'Grag2021_latent': ['imagenet', 'coco'],
             'Ohja2023': ['imagenet', 'coco', 'lsun', 'laion', 'raw_camera', 'celeba'],
             'Ohja2023ResNet50': ['imagenet', 'coco', 'lsun', 'laion', 'raw_camera', 'celeba'],
             'CLIP2024': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'CLIP2024Plus': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise', 'raw_camera'],
             'Corvi2023': ['imagenet', 'coco', 'ffhq'],
             'Wang2020JPEG01': ['lsun', 'imagenet', 'coco', 'celeba', 'raw_camera'],
             'Wang2020JPEG05': ['lsun', 'imagenet', 'coco', 'celeba', 'raw_camera']}

SYN_DETECTOR_DATASET_MAPPING = {'Grag2021_progan': 'Corvi2023',
             'Grag2021_latent': 'Corvi2023',
             'Ohja2023': 'Ohja2023',
             'Ohja2023ResNet50': 'Ohja2023',
             'CLIP2024': 'CLIP2024',
             'CLIP2024Plus': 'CLIP2024',
             'Corvi2023': 'Corvi2023',
             'Wang2020JPEG01': 'Wang2020JPEG01',
             'Wang2020JPEG05': 'Wang2020JPEG01'}

DETECTORS = ['Grag2021_progan', 'Grag2021_latent', 'Ohja2023', 'Ohja2023ResNet50',
             'CLIP2024', 'CLIP2024Plus', 'Corvi2023', 'Wang2020JPEG01', 'Wang2020JPEG05']
