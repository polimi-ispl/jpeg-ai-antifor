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

# --- Helpers functions and classes --- #
class JPEGAIDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform: T.Transform=None, data_df: pd.DataFrame=None):
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
        return image, torch.Tensor([label], dtype=torch.float32)