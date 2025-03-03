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
import numpy as np
import random

# --- Helpers functions and classes --- #

def get_transform_list(detector: str):
    if detector == 'Grag2021_progan':
        return T.Compose([T.CenterCrop(256),
                          T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    elif detector == 'Grag2021_latent':
        return T.Compose([T.ToTensor(),T.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])
    elif detector == 'Ohja2023':
        return T.Compose([T.CenterCrop(224), T.ToTensor(),
                          T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                      std=[0.26862954, 0.26130258, 0.27577711])])
    elif detector == 'Ohja2023ResNet50':
        return T.Compose([T.CenterCrop(224), T.ToTensor(),
                          T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                      std=[0.26862954, 0.26130258, 0.27577711])])
    elif detector in ['CLIP2024', 'CLIP2024Plus']:
        return T.Compose([T.Resize(224), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                      std=[0.26862954, 0.26130258, 0.27577711])])
    elif detector == 'Corvi2023':
        return T.Compose([T.CenterCrop(256), T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
    elif detector in ['Wang2020JPEG01', 'Wang2020JPEG05']:
        return T.Compose([T.CenterCrop(224), T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
    elif detector == 'NPR':
        return T.Compose([T.Resize((256, 256)), T.ToTensor(),
                          T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])])
    elif detector == 'Mandelli2024':
        return RandomPatchTransform(patch_size=96, n_patches=800)
    elif detector == 'TruFor':
        return T.Compose([T.ToTensor()])  # ToTensor already converts to [0, 1]
    elif detector == 'MMFusion':
        return T.Compose([T.ToTensor()])  # ToTensor already converts to [0, 1]
    elif detector == 'ImageForensicsOSN':
        return T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        return T.Compose([T.ToTensor()])

# --- Custom transforms --- #
class RandomPatchTransform(torch.nn.Module):
    def __init__(self, patch_size: int, n_patches: int):
        super(RandomPatchTransform, self).__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.random_crop = T.RandomCrop(patch_size)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = T.Resize(256, interpolation=T.InterpolationMode.BILINEAR)

    def forward(self, img: Image.Image):

        # set the seeds for the random extraction of patches
        random.seed(21)
        np.random.seed(21)
        torch.manual_seed(21)

        # Check on image format
        if img.ndim < 3:
            print('Gray scale image, converting to RGB')
            img2 = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            img2[:, :, 0] = img
            img2[:, :, 1] = img
            img2[:, :, 2] = img
            img = img2.copy()
        if img.shape[2] > 3:
            print('Omitting alpha channel')
            img = img[:, :, :3]

        # Resize the image if it is too small
        if img.size[0] < 256 or img.size[1] < 256:
            img = self.resize(img)

        # Extract patches
        patches = []
        for _ in range(self.n_patches):
            patch = self.random_crop(img)
            patch = T.ToTensor()(patch)
            patch = self.normalize(patch)
            patches.append(patch)

        return torch.stack(patches)

# --- Dataset classes --- #

class ImgDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading images using a Pandas DataFrame for the data info.
    The DataFrame must have the path as part of the index.
    The __getitem__ method returns the image and a dummy label.
    """
    def __init__(self, root_dir: str, data_df: pd.DataFrame, transform: torch.nn.Module=None):
        """
        Initialize the dataset.
        :param root_dir: the root directory where the images are stored.
        :param data_df: the DataFrame containing the data info.
        :param transform: the transformation to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx].name[-1]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.Tensor([0])

class ImgSplicingDataset(torch.utils.data.Dataset):
    """
    Dataset class for loading spliced images and info about them using a Pandas DataFrame for the data storage.
    The DataFrame must have the path as part of the index.
    The __getitem__ method returns the image and a dummy label.
    """
    def __init__(self, root_dir: str, data_df: pd.DataFrame, transform: torch.nn.Module=None):
        """
        Initialize the dataset.
        :param root_dir: the root directory where the images are stored.
        :param data_df: the DataFrame containing the data info.
        :param transform: the transformation to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_path = self.data_df.iloc[idx].name[-1]
        image = Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, torch.Tensor([0])


if __name__ == '__main__':
    # --- Test the dataset --- #
    import pandas as pd
    from utils.params import TEST_DATA
    from tqdm import tqdm
    data_info = pd.read_csv('/nas/public/exchange/JPEG-AI/data/TEST/data_info_complete.csv')
    data_info = data_info.loc[data_info['dataset'].isin(TEST_DATA['Grag2021_progan'])]
    transforms = get_transform_list('Grag2021_progan')
    dataset = ImgDataset(root_dir='/nas/public/exchange/JPEG-AI/data/TEST', data_df=data_info, transform=transforms)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
    for image, label in tqdm(dataloader):
        continue

