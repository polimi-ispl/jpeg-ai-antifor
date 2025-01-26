"""
Class for storing the class definitions of the synthetic image detectors.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

# --- Libraries --- #
import os
import sys
import argparse
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from multiprocessing import cpu_count
from .params import *
from torch.nn import functional as F
import cv2

# --- Helpers functions --- #

def gkern(kernlen=7, nsig=3, channels=1):
    """
    Returns a 2D Gaussian kernel with a customizable number of channels.

    Parameters:
    - kernlen: Length of the kernel (both width and height).
    - nsig: Standard deviation for the Gaussian function.
    - channels: Number of channels for the output kernel.

    Returns:
    - A Gaussian kernel tensor of shape (channels, kernlen, kernlen).
    """
    # Create a small 3x3 Gaussian kernel
    rtn = [[0, 0, 0],
           [0, 1, 0],
           [0, 0, 0]]
    rtn = np.array(rtn, dtype=np.float32)

    # Resize the kernel to the specified kernel length
    rtn = cv2.resize(rtn, (kernlen, kernlen))

    # Repeat the kernel for the specified number of channels
    rtn = np.concatenate([rtn[..., None]] * channels, axis=2)  # Shape: (kernlen, kernlen, channels)

    return torch.from_numpy(rtn)  # Convert to PyTorch tensor

# --- Classes --- #

class SynImgDetector:
    def __init__(self, detector, weights_path, device='cuda:0'):
        self.detector = detector
        self.weights_path = weights_path
        self.device = device
        self.model = self.init_model()

    def init_model(self):
        if self.detector == 'Grag2021_progan':
            from utils.third_party.DMImageDetection_test_code.get_method_here import get_method_here, def_model
            _, model_path, arch, norm_type, patch_size = get_method_here(self.detector,
                                                                         weights_path=self.weights_path)
            model = def_model(arch, model_path, localize=False)
            model = model.to(self.device).eval()
            print("Model loaded!")
            return model
        elif self.detector == 'Grag2021_latent':
            from utils.third_party.DMImageDetection_test_code.get_method_here import get_method_here, def_model
            _, model_path, arch, norm_type, patch_size = get_method_here(self.detector,
                                                                         weights_path=self.weights_path)
            model = def_model(arch, model_path, localize=False)
            model = model.to(self.device).eval()
            print("Model loaded!")
            return model
        elif self.detector == 'Ojha2023':
            from utils.third_party.UniversalFakeDetect_test_code.models import get_model
            # Load the backbone model
            model = get_model('CLIP:ViT-L/14')
            # Load the last fully connected layer
            state_dict = torch.load(os.path.join(self.weights_path, 'fc_weights.pth'), map_location='cpu')
            model.fc.load_state_dict(state_dict)
            model = model.to(self.device).eval()
            print("Model loaded!")
            return model
        elif self.detector == 'Ojha2023ResNet50':
            from utils.third_party.UniversalFakeDetect_test_code.models import get_model
            # Load the backbone model
            model = get_model('CLIP:RN50')
            # Load the last fully connected layer
            state_dict = torch.load(os.path.join(self.weights_path, 'fc_weights.pth'), map_location='cpu')
            model.fc.load_state_dict(state_dict)
            model = model.to(self.device).eval()
            print("Model loaded!")
            return model
        elif self.detector in ['Cozzolino2024-A', 'Cozzolino2024-B', 'Corvi2023']:
            from utils.third_party.ClipBased_SyntheticImageDetection_main.networks import create_architecture, load_weights
            # Load the config-file
            with open(os.path.join(self.weights_path, MODELS_LIST[self.detector], 'config.yaml')) as fid:
                data = yaml.load(fid, Loader=yaml.FullLoader)
            model_path = os.path.join(self.weights_path, MODELS_LIST[self.detector], data['weights_file'])
            arch = data['arch']
            # Load the model
            model = load_weights(create_architecture(arch), model_path)
            model = model.to(self.device).eval()
            return model
        elif self.detector in ['Wang2020-A', 'Wang2020-B']:
            from utils.third_party.Wang2020CNNDetection.networks.resnet import resnet50
            model = resnet50(num_classes=1)
            state_dict = torch.load(os.path.join(self.weights_path, MODELS_LIST[self.detector]), map_location='cpu')
            model.load_state_dict(state_dict['model'])
            model = model.to(self.device).eval()
            return model
        elif self.detector == 'NPR':
            from utils.third_party.NPR.networks.resnet import resnet50
            model = resnet50(num_classes=1)
            model.load_state_dict(torch.load(os.path.join(self.weights_path, MODELS_LIST[self.detector]),
                                             map_location='cpu'), strict=True)
            model = model.to(self.device).eval()
            return model
        else:
            raise NotImplementedError(f"SynImgDetector {self.detector} not implemented")


    def process_sample(self, sample: torch.Tensor):
        with torch.no_grad():
            sample = sample.to(self.device)
            output = self.model(sample)
            if self.detector == 'Grag2021_progan':
                output = output.cpu().numpy()
                if output.shape[1] == 1:
                    output = output[:, 0]
                elif output.shape[1] == 2:
                    output = output[:, 1] - output[:, 0]
                else:
                    assert False
                if len(output.shape) > 1:
                    output = np.mean(output, (1, 2))
                else:
                    output = output
            elif self.detector == 'Grag2021_latent':
                output = output.cpu().numpy()
                if output.shape[1] == 1:
                    output = output[:, 0]
                elif output.shape[1] == 2:
                    output = output[:, 1] - output[:, 0]
                else:
                    assert False
                if len(output.shape) > 1:
                    output = np.mean(output, (1, 2))
                else:
                    output = output
            elif self.detector == 'Ojha2023':
                output = output.flatten().cpu().numpy()
            elif self.detector in ['Ojha2023ResNet50', 'Cozzolino2024-A', 'Cozzolino2024-B', 'Corvi2023']:
                output = output.cpu().numpy()
                if output.shape[1] == 1:
                    output = output[:, 0]
                elif output.shape[1] == 2:
                    output = output[:, 1] - output[:, 0]
                else:
                    assert False
                if len(output.shape) > 1:
                    output = np.mean(output, (1, 2))
                else:
                    output = output
            elif self.detector in ['Wang2020-A', 'Wang2020-B']:
                output = output.cpu().numpy()
            return output

class ImgSplicingDetector:
    def __init__(self, detector, weights_path, device='cuda:0'):
        self.detector = detector
        self.weights_path = weights_path
        self.device = device
        self.model = self.init_model()

    def init_model(self):
        if self.detector == 'TruFor':
            # Load default config
            from utils.third_party.TruFor.test_docker.src.config import _C as config
            # Merge the other parameters from the yaml file
            config.merge_from_file(os.path.join(self.weights_path, 'src', 'trufor.yaml'))
            config.TEST.MODEL_FILE = os.path.join(self.weights_path, 'weights', 'trufor.pth.tar')
            if config.TEST.MODEL_FILE:
                model_state_file = config.TEST.MODEL_FILE
            else:
                raise ValueError("Model file is not specified.")

            print('=> loading model from {}'.format(model_state_file))
            checkpoint = torch.load(model_state_file, map_location='cpu')

            if config.MODEL.NAME == 'detconfcmx':
                from utils.third_party.TruFor.test_docker.src.models.cmx.builder_np_conf import myEncoderDecoder as confcmx
                model = confcmx(cfg=config)
            else:
                raise NotImplementedError('Model not implemented')

            model.load_state_dict(checkpoint['state_dict'])
            model = model.eval().to(self.device)
            return model
        elif self.detector == 'ImageForensicsOSN':
            from utils.third_party.ImageForensicsOSN_main.models.scse import SCSEUnet
            # Create the model
            model = SCSEUnet(backbone_arch='senet154', num_channels=3)
            # Load the weights
            state_dict = torch.load(self.weights_path, map_location='cpu')
            # Rename the keys to match the architecture
            det_net_state_dict = {k.replace('module.det_net.', ''): v for k, v in state_dict.items() if
                                  k.startswith('module.det_net.')}
            # Load the weights
            model.load_state_dict(det_net_state_dict)
            model = model.eval().to(self.device)
            return model
        else:
            raise NotImplementedError(f"ImgSplicingDetector {self.detector} not implemented")


    def process_sample(self, sample: torch.Tensor):
        with torch.no_grad():
            sample = sample.to(self.device)
            if self.detector == 'TruFor':
                # We are just interested in the final anomaly map, ignore for the moment the other outputs
                output, conf, det, npp = self.model(sample)
                # Process the output
                output = torch.squeeze(output, 0)
                output = F.softmax(output, dim=0)[1]
                output = output.cpu().numpy()
            elif self.detector == 'ImageForensicsOSN':
                # --- Decompose the sample into patches --- #
                _, _, H, W = sample.shape
                patch_size = 896  # Patch size

                # --- Check if the image is smaller than the patch size --- #
                if (H < patch_size) or (W < patch_size):
                    # Process the entire image
                    output = self.model(sample)
                else:

                    # Initialize list for patches
                    patches = []
                    idx = 0
                    # Calculate X and Y based on your patch size
                    X = H // (patch_size // 2) + 1
                    Y = W // (patch_size // 2) + 1

                    # Patch Extraction Logic
                    for x in range(X - 1):  # Loop up to X - 1
                        if x * patch_size // 2 + patch_size > H:
                            break
                        for y in range(Y - 1):  # Loop up to Y - 1
                            if y * patch_size // 2 + patch_size > W:
                                break

                            # Extract patch
                            patch = sample[:, :, x * patch_size // 2: x * patch_size // 2 + patch_size,
                                    y * patch_size // 2: y * patch_size // 2 + patch_size]
                            patches.append(patch)
                            idx += 1

                        # Extract the last column for the current row
                        if x * patch_size // 2 + patch_size <= H:
                            patch = sample[:, :, x * patch_size // 2: x * patch_size // 2 + patch_size,
                                    -patch_size:]  # Last column
                            patches.append(patch)
                            idx += 1

                    # Handle the last row separately
                    for y in range(Y - 1):  # Loop up to Y - 1
                        if y * patch_size // 2 + patch_size > W:
                            break

                        patch = sample[:, :, -patch_size:, y * patch_size // 2: y * patch_size // 2 + patch_size]
                        patches.append(patch)
                        idx += 1

                    # Extract the bottom-right corner patch
                    if (patches and len(patches) < idx + 1):
                        patch = sample[:, :, -patch_size:, -patch_size:]  # Bottom-right corner
                        patches.append(patch)

                    # --- Process the patches --- #
                    predictions = []
                    for patch in patches:
                        patch = patch.to(self.device)
                        patch_output = self.model(patch)
                        predictions.append(patch_output)
                    predictions = torch.cat(predictions, 0)

                    # --- Reconstruct the output from the single patches predictions --- #

                    # Create the Gaussian kernel
                    gk = gkern(kernlen=patch_size, channels=predictions.shape[1])
                    gk = 1 - gk
                    gk_tensor = gk.to(self.device)
                    gk_tensor = gk_tensor.unsqueeze(0).permute(0, 3, 1, 2) # unsqueeze gk_tensor to match the number of samples permute it channel first

                    # Prepare the output
                    output = torch.ones((sample.shape[0], 1, sample.shape[-2], sample.shape[-1])).to(self.device) * -1
                    patch_idx = 0

                    # Main patch cycle
                    for x in range(X - 1):
                        if x * patch_size // 2 + patch_size > H:
                            break
                        for y in range(Y - 1):
                            if y * patch_size // 2 + patch_size > W:
                                break
                            img_tmp = predictions[patch_idx].unsqueeze(0)  # get the current patch
                            weight_cur = output[:, :, x * patch_size // 2: x * patch_size // 2 + patch_size,
                                             y * patch_size // 2: y * patch_size // 2 + patch_size,].clone()  # get the current weight
                            h1, w1 = weight_cur.shape[-2], weight_cur.shape[-1]
                            gk_tmp = F.interpolate(gk_tensor, size=(h1, w1), mode='bilinear', align_corners=False)  # interpolate the Gaussian kernel to the current patch size
                            # Compute the weights (all of this make very little sense to me, but it's the way the original code works)
                            weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                            weight_cur[weight_cur == -1] = 0
                            weight_tmp = 1 - weight_cur
                            # Compute the final output
                            output[:, :, x * patch_size // 2: x * patch_size // 2 + patch_size, y * patch_size // 2: y * patch_size // 2 + patch_size] = (
                                    weight_cur * output[:, :, x * patch_size // 2: x * patch_size // 2 + patch_size,
                                                     y * patch_size // 2: y * patch_size // 2 + patch_size] +
                                    weight_tmp * img_tmp
                            )
                            patch_idx += 1

                        # Handle the last column (comments are the same as above)
                        img_tmp = predictions[patch_idx].unsqueeze(0)  # get the current patch
                        weight_cur = output[:, :, x * patch_size // 2: x * patch_size // 2 + patch_size, -patch_size:].clone() # get the current weight
                        h1, w1 = weight_cur.shape[-2], weight_cur.shape[-1]
                        gk_tmp = F.interpolate(gk_tensor, size=(h1, w1), mode='bilinear', align_corners=False) # interpolate the Gaussian kernel to the current patch size
                        # Compute the weights (all of this make very little sense to me, but it's the way the original code works)
                        weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                        weight_cur[weight_cur == -1] = 0
                        weight_tmp = 1 - weight_cur
                        # Compute the final output
                        output[:, :, x * patch_size // 2: x * patch_size // 2 + patch_size, -patch_size:] = (
                                weight_cur * output[:, :, x * patch_size // 2: x * patch_size // 2 + patch_size, -patch_size:] +
                                weight_tmp * img_tmp
                        )
                        patch_idx += 1

                    # Handle the last row (again, comments are the same as above)
                    for y in range(Y - 1):
                        if y * patch_size // 2 + patch_size > W:
                            break
                        img_tmp = predictions[patch_idx].unsqueeze(0)
                        weight_cur = output[:, :, -patch_size:, y * patch_size // 2: y * patch_size // 2 + patch_size].clone()
                        h1, w1 = weight_cur.shape[-2], weight_cur.shape[-1]
                        gk_tmp = F.interpolate(gk_tensor, size=(h1, w1), mode='bilinear', align_corners=False)

                        weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                        weight_cur[weight_cur == -1] = 0
                        weight_tmp = 1 - weight_cur

                        output[:, :, -patch_size:, y * patch_size // 2: y * patch_size // 2 + patch_size] = (
                                weight_cur * output[:, :, -patch_size:, y * patch_size // 2: y * patch_size // 2 + patch_size] +
                                weight_tmp * img_tmp
                        )
                        patch_idx += 1

                    # Handle the bottom-right corner (again, comments are the same as above)
                    img_tmp = predictions[patch_idx].unsqueeze(0)
                    weight_cur = output[:, :, -patch_size:, -patch_size:].clone()
                    h1, w1 = weight_cur.shape[-2], weight_cur.shape[-1]
                    gk_tmp = F.interpolate(gk_tensor, size=(h1, w1), mode='bilinear', align_corners=False)

                    weight_cur[weight_cur != -1] = gk_tmp[weight_cur != -1]
                    weight_cur[weight_cur == -1] = 0
                    weight_tmp = 1 - weight_cur

                    output[:, :, -patch_size:, -patch_size:] = (
                            weight_cur * output[:, :, -patch_size:, -patch_size:] +
                            weight_tmp * img_tmp
                    )
                output = output.squeeze().cpu().numpy()
            else:
                raise NotImplementedError(f"ImgSplicingDetector {self.detector} not implemented")
            return output
