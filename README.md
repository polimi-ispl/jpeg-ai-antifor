# Is JPEG AI going to change image forensics?

![](assets/teaser.jpg)

This is the official code repository for the paper *Is JPEG AI going to change image forensics?* currently under revision.  
The repository is **under development**, so feel free to open an issue if you encounter any problems.

# Getting started

In order to run our code, you need to:
1. install [conda](https://docs.conda.io/en/latest/miniconda.html)
2. create the `jpeg-ai-antifor` environment using the *environment.yml* file
```bash
conda env create -f envinroment.yml
conda activate jpeg-ai-antifor
```
3. download the [dataset] of the paper and extract it in the *data* folder.
4. 

# Running the code

## Deepfake image detection
The script `test_detector.py` allows to test the performance of the different considered deepfake detectors on the dataset.


