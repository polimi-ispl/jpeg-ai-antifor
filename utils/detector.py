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
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from multiprocessing import cpu_count
from params import *

# --- Helpers functions and classes --- #
class Detector:
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
        elif self.detector == 'Wang2023':
            from utils.third_party.UniversalFakeDetect_test_code.models import get_model
            # Load the backbone model
            model = get_model('CLIP:ViT-L/14')
            # Load the last fully connected layer
            state_dict = torch.load(os.path.join(self.weights_path, 'fc_weights.pth'), map_location='cpu')
            model.fc.load_state_dict(state_dict)
            model = model.to(self.device).eval()
            print("Model loaded!")
        elif self.detector == 'Wang2023ResNet50':
            from utils.third_party.UniversalFakeDetect_test_code.models import get_model
            # Load the backbone model
            model = get_model('CLIP:RN50')
            # Load the last fully connected layer
            state_dict = torch.load(os.path.join(self.weights_path, 'fc_weights.pth'), map_location='cpu')
            model.fc.load_state_dict(state_dict)
            model = model.to(self.device).eval()
            print("Model loaded!")
            return model


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
            elif self.detector == 'Wang2023':
                output = output.flatten().cpu().numpy()
            return output
