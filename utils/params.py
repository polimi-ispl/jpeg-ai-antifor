"""
Constant values used throughout the codebase.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

JPEG_AI_PATH = '/nas/home/ecannas/third_party_code/jpeg-ai-reference-software'  # put here the path to the jpeg-ai-reference-software

MODELS_LIST = {'Grag2021_progan': 'Grag2021_progan',
               'Grag2021_latent': 'Grag2021_latent',
               'Ohja2023': 'Ohja2023',
               'Ohja2023ResNet50': 'Ohja2023ResNet50',
               'CLIP2024': 'clipdet_latent10k',
               'CLIP2024Plus': 'clipdet_latent10k_plus',
               'Corvi2023': 'Corvi2023'}

TEST_DATA = {'Grag2021_progan': ['imagenet', 'coco'],
             'Grag2021_latent': ['imagenet', 'coco'],
             'Ohja2023': ['imagenet', 'coco', 'lsun', 'laion', 'raise', 'celeba'],
             'Ohja2023ResNet50': ['imagenet', 'coco', 'lsun', 'laion', 'raise', 'celeba'],
             'CLIP2024': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise'],
             'CLIP2024Plus': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise'],
             'Corvi2023': ['lsun', 'ffhq', 'imagenet', 'coco', 'laion', 'raise']}

DETECTORS = ['Grag2021_progan', 'Grag2021_latent', 'Ohja2023', 'Ohja2023ResNet50',
             'CLIP2024', 'CLIP2024Plus', 'Corvi2023']
