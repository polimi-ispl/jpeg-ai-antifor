"""
Constant values used throughout the codebase.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

JPEG_AI_PATH = '/nas/home/ecannas/third_party_code/jpeg-ai-reference-software'  # put here the path to the jpeg-ai-reference-software

MODELS_LIST = {'Grag2021_progan': 'Grag2021_progan',
               'Grag2021_latent': 'Grag2021_latent',
               'Ohja2023': 'Ohja2023',
               'Ohja2023ResNet50': 'Ohja2023ResNet50'}

TEST_DATA = {'Grag2021_progan': ['imagenet', 'coco'],
             'Grag2021_latent': ['imagenet', 'coco'],
             'Ohja2023': ['imagenet', 'coco', 'lsun', 'laion', 'raise', 'celeba'],
             'Ohja2023ResNet50': ['imagenet', 'coco', 'lsun', 'laion', 'raise', 'celeba']}

DETECTORS = ['Grag2021_progan', 'Grag2021_latent', 'Ohja2023', 'Ohja2023ResNet50']
