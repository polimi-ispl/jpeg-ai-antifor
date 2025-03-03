"""
Main file to test an image through the fully-synthetic vs laundered detector

Author:
Sara Mandelli - sara.mandelli@polimi.it
"""

# --- Libraries import --- #
import os
from collections import OrderedDict
import numpy as np
import torch
import random
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import albumentations as A
import albumentations.pytorch as Ap
from utils import architectures
from utils.blazeface import FaceExtractor, BlazeFace
from PIL import Image, ImageFile
import argparse
ImageFile.LOAD_TRUNCATED_IMAGES = True


# --- Class definition --- #
class FullySynthvsLaunderedDetector:

    def __init__(self, device: str, M: int = 600, select_face_test: bool = False):

        self.select_face_test = select_face_test
        self.M = M

        # GPU configuration if available
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')

        # Instantiate and load network
        network_class = getattr(architectures, 'EfficientNetB4')

        # model path
        self.model_path = 'fully-synth_vs_laundered.pth'
        net = network_class(n_classes=2, pretrained=False).eval().to(self.device)
        state_tmp = torch.load(self.model_path, map_location='cpu')
        if 'net' not in state_tmp.keys():
            state = OrderedDict({'net': OrderedDict()})
            [state['net'].update({'model.{}'.format(k): v}) for k, v in state_tmp.items()]
        else:
            state = state_tmp
        incomp_keys = net.load_state_dict(state['net'], strict=True)
        print(incomp_keys)
        self.net = net
        print('Model for fully-synthetic vs laundered image detection loaded!')

        net_normalizer = self.net.get_normalizer()
        transform = [
            A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std),
            Ap.transforms.ToTensorV2()
        ]
        self.trans = A.Compose(transform)
        self.cropper = A.RandomCrop(width=96, height=96, always_apply=True, p=1.)

    def laundered_img_detection(self, img: np.array) -> torch.Tensor:
        """
        Detection pipeline:
            1. 800 patches are randomly extracted from the image
            2. The patches are normalized
            3. All patches are processed through the fully-synth vs laundered detector
            4. A majority voting on the scores obtained is performed to take a decision on the overall image,
            according to the parameter M which specifies how many patches to aggregate
            (for more information, see: https://arxiv.org/pdf/2407.10736)

        :param img: np.array, probe to analyze
        :return: torch.Tensor, laundered image detection score:
        if score > 0, the image is detected as being laundered
        if score < 0, the image is detected as being fully-synthetic
        """

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

        if self.select_face_test:

            # Load face detector
            face_detector = BlazeFace()
            face_detector.load_weights("utils/blazeface/blazeface.pth")
            face_detector.load_anchors("utils/blazeface/anchors.npy")
            face_extractor = FaceExtractor(facedet=face_detector)

            # Face detector is the same used in https://github.com/polimi-ispl/icpr2020dfdc
            # Check the linked repository for more information

            #  Pipeline for face extraction
            ############################################################################################################

            # Split the image into several tiles. Resize the tiles to 128x128.
            tiles, resize_info = face_extractor._tile_frames(frames=np.expand_dims(img, 0),
                                                             target_size=face_detector.input_size)
            # tiles has shape (num_tiles, target_size, target_size, 3)
            # resize_info is a list of four elements [resize_factor_y, resize_factor_x, 0, 0]
            # Run the face detector. The result is a list of PyTorch tensors,
            # one for each tile in the batch.
            detections = face_detector.predict_on_batch(tiles, apply_nms=False)
            # Convert the detections from 128x128 back to the original image size.
            image_size = (img.shape[1], img.shape[0])
            detections = face_extractor._resize_detections(detections, face_detector.input_size, resize_info)
            detections = face_extractor._untile_detections(1, image_size, detections)
            # The same face may have been detected in multiple tiles, so filter out overlapping detections.
            detections = face_detector.nms(detections)

            # Crop the faces out of the original frame.
            frameref_detections = face_extractor._add_margin_to_detections(detections[0], image_size, 0.5)
            faces = face_extractor._crop_faces(img, frameref_detections)

            # Add additional information about the frame and detections.
            scores = list(detections[0][:, 16])
            frame_dict = {"faces": faces,
                          "scores": scores,
                          }
            # consider at most the two best detected faces
            if len(faces) > 1:
                faces = [faces[x] for x in np.argsort(scores)]
                faces = [faces[-2], faces[-1]]
            # if only one face is detected, consider it
            elif len(faces) == 1:
                faces = [frame_dict['faces'][-1]]
            # if a face has not been detected, consider the entire img
            else:
                faces = [img]

            ############################################################################################################

            # define the list containing all the analyzed patches (for all the considered faces)
            all_patches = []
            for face in faces:

                # if the face size is smaller than 256 x 256, perform a little bit of upscaling to enlarge its size
                if face.shape[0] < 256 or face.shape[1] < 256:
                    face = A.SmallestMaxSize(max_size=256, interpolation=1, always_apply=True, p=1)(image=face)['image']

                # randomly extract the patches:
                patches = [self.cropper(image=face)['image'] for x in range(800)]
                all_patches.extend(patches)

            # if the number of patches is too high (due to multiple faces detected), we still keep 800 patches
            if len(all_patches) > 800:
                # shuffle the patch-list
                random.shuffle(all_patches)
                all_patches = all_patches[:800]

        # if select_face_test = False
        else:

            # if the image size is smaller than 256 x 256, perform a little bit of upscaling to enlarge its size
            if img.shape[0] < 256 or img.shape[1] < 256:
                img = A.SmallestMaxSize(max_size=256, interpolation=1, always_apply=True, p=1)(image=img)['image']

            # extract patches from the image
            all_patches = [self.cropper(image=img)['image'] for x in range(800)]

        # Normalize the patches
        transf_patch_list = [self.trans(image=patch)['image'] for patch in all_patches]

        # Synthetic image detection
        ################################################################################################################

        # Compute scores
        transf_patch_tensor = torch.stack(transf_patch_list, dim=0).to(self.device)
        with torch.no_grad():
            patch_scores = self.net(transf_patch_tensor)

        # aggregate the scores to compute the final image score
        img_score = torch.mean(torch.sort(patch_scores[:, 1])[0][-self.M:])

        return img_score


def main():
    """
    Main function to detect if an image is laundered or fully-synthetic
    if score > 0, the image is detected as being laundered
    if score < 0, the image is detected as being fully-synthetic
    """

    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--select_face_test', action='store_true', help='If testing only the face')
    parser.add_argument('--M', type=int, default=600, help='Number of patches aggregated for computing '
                                                           'the final image score')
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    img_path = args.img_path
    select_face_test = args.select_face_test
    M = args.M
    gpu = args.gpu

    # Load image
    img = np.asarray(Image.open(img_path))

    # Process the image with the detector
    device = f'cuda:{gpu}'
    detector = FullySynthvsLaunderedDetector(device=device, M=M, select_face_test=select_face_test)
    img_score = detector.laundered_img_detection(img=img)

    # Print the scores
    print('Fully-synth vs Laundered image score: {}'.format(img_score))

    return 0


if __name__ == '__main__':
    main()


