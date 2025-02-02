# Third party code
This directory contains the third-party code used in our experiments.
We relied on the following detectors:
1. *Gragnaniello2021*, available in the [DMImageDetection](https://github.com/grip-unina/DMimageDetection/tree/main) repository;
2. *Corvi2023*, *Cozzolino2024-A*, and *Cozzolino2024-B*, available  in the [ClipBased-SyntheticImageDetection](https://github.com/grip-unina/ClipBased-SyntheticImageDetection) repository;
3. *Wang2020-A* and *Wang2020-B*, available in the [CNNDetection](https://github.com/PeterWang512/CNNDetection?tab=readme-ov-file) repository;
4. *Ojha2023*, available in the [UniversalFakeDetect](https://github.com/WisconsinAIVision/UniversalFakeDetect) repository;
5. *NPR*, available in the [NPR](https://github.com/chuangchuangtan/NPR-DeepfakeDetection/blob/main/test.py) repository;
6. *TruFor*, available in the [TruFor](https://github.com/grip-unina/TruFor/tree/main) repository;
7. *MMFusion*, available in the [MMFusion](https://github.com/IDT-ITI/MMFusion-IML?tab=readme-ov-file) repository;
8. *ImageForensicsOSN*, available in the [ImageForensicsOSN](https://github.com/HighwayWu/ImageForensicsOSN/tree/main) repository.

## Installation
Each detector (fortunately) is based on PyTorch, so the `environment.yml` file is the same for all of them.  
Download the pre-trained weights following the instructions in the respective repositories and extract them in the corresponding folders.  
Follow these paths:
1. `Gragnaniello2021`: `utils/third_party/DMImageDetection_test_code/weights`;
2. `Corvi2023`, `Cozzolino2024-A`, `Cozzolino2024-B`: `utils/third_party/ClipBased_SyntheticImageDetection_main/weights`;
3. `Wang2020-A`, `Wang2020-B`: `utils/third_party/Wang2020CNNDetectoin/weights`;
4. `Ojha2023`: `utils/third_party/UniversalFakeDetect_test_code/pretrained_weights`;
5. `NPR`: `utils/third_party/NPR/weights`;
6. `MMFusion`: `utils/third_party/MMFusion/weights`;
7. `ImageForensicsOSN`: `utils/third_party/ImageForensicsOSN_main/weights`;

Please notice that TruFor is actually shipped under a Docker image.  
To download the weights, following this [link](https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip) (you can find it inside the [Dockerfile](https://github.com/grip-unina/TruFor/blob/main/test_docker/Dockerfile)) 
and put them inside the `utils/third_party/TruFor/weights` folder.  

## License
Please notice that the each detector is licensed under the respective license (when available).  
We report each of them here:
1. [Wang2020](https://github.com/PeterWang512/CNNDetection/blob/master/LICENSE.txt);
2. [Gragnaniello2021](https://github.com/grip-unina/DMimageDetection/blob/main/LICENSE.md);
3. [Corvi2023, Cozzolino2024](https://github.com/grip-unina/ClipBased-SyntheticImageDetection/blob/main/LICENSE.md);
4. [TruFor](https://github.com/grip-unina/TruFor/blob/main/test_docker/LICENSE.txt);
5. [MMFusion](https://github.com/IDT-ITI/MMFusion-IML?tab=readme-ov-file);
6. [ImageForensicsOSN](https://github.com/HighwayWu/ImageForensicsOSN/blob/main/LICENSE).

**PLEASE FOLLOW THESE LICENSES THOROUGHLY IF YOU AIM AT USING THE CODE OUTSIDE THE SCOPE OF THIS REPOSITORY.**