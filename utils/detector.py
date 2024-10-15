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
            from third_party.DMImageDetection_test_code import get_method_here, def_model
            _, model_path, arch, norm_type, patch_size = get_method_here(self.detector,
                                                                         weights_path=self.weights_path)
            model = def_model(arch, model_path, localize=False)
            model = model.to(self.device).eval()
            return model
        elif self.detector == 'Grag2021_latent':
            from third_party.DMImageDetection_test_code import get_method_here, def_model
            _, model_path, arch, norm_type, patch_size = get_method_here(self.detector,
                                                                         weights_path=self.weights_path)
            model = def_model(arch, model_path, localize=False)
            model = model.to(self.device).eval()
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
            return output
