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
from utils.third_party.Mandelli2024.utils.blazeface import FaceExtractor, BlazeFace

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
    elif detector in ['Mandelli2024', 'Mandelli2024-FT', 'Mandelli2024-RT']:
        return MandelliRandomPatchTransform(patch_size=96, n_patches=800)
    elif detector == 'TruFor':
        return T.Compose([T.ToTensor()])  # ToTensor already converts to [0, 1]
    elif detector == 'MMFusion':
        return T.Compose([T.ToTensor()])  # ToTensor already converts to [0, 1]
    elif detector == 'ImageForensicsOSN':
        return T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        return T.Compose([T.ToTensor()])


def mandelli_collate_fn(batch):
    """
    Collate function for the Mandelli2024 detector.
    This function is needed to manage the output of the MandelliRandomPatchTransform.
    """
    data = torch.cat([item[0] for item in batch], dim=0)
    target = [item[1] for item in batch]
    return [data, target]


def return_collate_fn(detector: str):
    """
    Return the collate function for the specific detector
    """
    if detector in ['Mandelli2024', 'Mandelli2024-FT', 'Mandelli2024-RT']:
        return mandelli_collate_fn
    else:
        return torch.utils.data.default_collate

# --- Custom transforms --- #
class MandelliRandomPatchTransform(torch.nn.Module):
    """
    Custom transformation for the Mandelli2024 detector.
    This transformation extracts random patches from the image.
    If the image contains faces, it extracts patches only from the face areas.
    """
    def __init__(self, patch_size: int, n_patches: int):
        super(MandelliRandomPatchTransform, self).__init__()
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.random_crop = T.RandomCrop(patch_size)
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = T.Resize(256, interpolation=T.InterpolationMode.BILINEAR)
        # self.face_detector = BlazeFace()
        # self.face_detector.load_weights(os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                                              'third_party/Mandelli2024/utils/blazeface/blazeface.pth'))
        # self.face_detector.load_anchors(os.path.join(os.path.dirname(os.path.realpath(__file__)),
        #                                              'third_party/Mandelli2024/utils/blazeface/anchors.npy'))
        # self.face_extractor = FaceExtractor(facedet=self.face_detector)

    def forward(self, img: Image.Image or np.array):

        # --- set the seeds for the random extraction of patches
        random.seed(21)
        np.random.seed(21)
        torch.manual_seed(21)

        # --- Check on image format and convert it to RGB if needed
        img = np.array(img)
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

        # --- Detect the faces if present on the image

        # # Split the image into several tiles. Resize the tiles to 128x128.
        # tiles, resize_info = self.face_extractor._tile_frames(frames=np.expand_dims(img, 0),
        #                                                  target_size=self.face_detector.input_size)
        # # tiles has shape (num_tiles, target_size, target_size, 3)
        # # resize_info is a list of four elements [resize_factor_y, resize_factor_x, 0, 0]
        # # Run the face detector. The result is a list of PyTorch tensors,
        # # one for each tile in the batch.
        # detections = self.face_detector.predict_on_batch(tiles, apply_nms=False)
        # # Convert the detections from 128x128 back to the original image size.
        # image_size = (img.shape[1], img.shape[0])
        # detections = self.face_extractor._resize_detections(detections, self.face_detector.input_size, resize_info)
        # detections = self.face_extractor._untile_detections(1, image_size, detections)
        # # The same face may have been detected in multiple tiles, so filter out overlapping detections.
        # detections = self.face_detector.nms(detections)
        #
        # # Crop the faces out of the original frame.
        # frameref_detections = self.face_extractor._add_margin_to_detections(detections[0], image_size, 0.5)
        # faces = self.face_extractor._crop_faces(img, frameref_detections)
        #
        # # Add additional information about the frame and detections.
        # scores = list(detections[0][:, 16])
        # frame_dict = {"faces": faces,
        #               "scores": scores,
        #               }
        # # consider at most the two best detected faces
        # if len(faces) > 1:
        #     faces = [faces[x] for x in np.argsort(scores)]
        #     faces = [faces[-2], faces[-1]]
        # # if only one face is detected, consider it
        # elif len(faces) == 1:
        #     faces = [frame_dict['faces'][-1]]
        # # if a face has not been detected, consider the entire img
        # else:
        #     faces = [img]
        #
        # # --- Crop the patches from the faces
        #
        # # define the list containing all the analyzed patches (for all the considered faces)
        # all_patches = []
        # for face in faces:
        #
        # Convert the face to a PIL image
        img = Image.fromarray(img)

        # if the face size is smaller than 256 x 256, perform a little bit of upscaling to enlarge its size
        if img.size[0] < 256 or img.size[1] < 256:
            img = self.resize(img)

        # Extract patches
        all_patches = []
        for _ in range(self.n_patches):
            patch = self.random_crop(img)
            patch = T.ToTensor()(patch)
            patch = self.normalize(patch)
            all_patches.append(patch)

        # if the number of patches is too high (due to multiple faces detected), we still keep 800 patches
        if len(all_patches) > 800:
            # shuffle the patch-list
            random.shuffle(all_patches)
            all_patches = all_patches[:800]

        return torch.stack(all_patches)

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

