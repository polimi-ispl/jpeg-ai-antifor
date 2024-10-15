"""
Data utilities.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
import os
import torch
import glob
from PIL import Image
import sys
from torchvision import transforms as T
import pandas as pd
from typing import List

# --- Helpers functions and classes --- #

def get_transform_list(detector: str):
    if detector == 'Grag2021_progan':
        return T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    elif detector == 'Grag2021_latent':
        return T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    else:
        return T.Compose([T.ToTensor()])


class JPEGAIDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, data_df: pd.DataFrame, transform: torch.nn.Module=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path, label = self.data_df.iloc[idx]['path'], self.data_df.iloc[idx]['compressed']
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, torch.Tensor([label])
