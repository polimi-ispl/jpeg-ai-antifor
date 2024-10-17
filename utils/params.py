"""
Constant values used throughout the codebase.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

JPEG_AI_PATH = '/nas/home/ecannas/third_party_code/jpeg-ai-reference-software'  # put here the path to the jpeg-ai-reference-software

MODELS_LIST = {'Grag2021_progan': 'Grag2021_progan',
               'Grag2021_latent': 'Grag2021_latent',
               'Wang2023': 'Wang2023'}

TEST_DATA = {'Grag2021_progan': ['imagenet', 'coco'],
             'Grag2021_latent': ['imagenet', 'coco'],}

DETECTORS = ['Grag2021_progan', 'Grag2021_latent', 'Wang2023']
