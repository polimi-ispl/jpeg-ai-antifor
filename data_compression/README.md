# Data compression
This directory contains the scripts used for compressing the data in our experiments.
Two scripts are available:
1. `jpeg_compress_directory.py`: This script compresses all the images inside a directory with standard JPEG using Pillow;
2. `jpegai_compress_directory.py`: This script compresses all the images inside a directory with the official JPEG AI Reference software.

## Environment
For the `jpeg_compress_directory.py` script, you can simply refer to the `environment.yml` file to install the required packages.  

For the `jpegai_compress_directory.py` script, you need to have the JPEG AI Reference software installed. You can find the software at this link: https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software/.  
Follow the instructions needed for the installation, and remember to activate the conda environment before running the script.

## Usage
Both scripts require the same arguments:
- `--input_dir`: the directory containing the images to compress;
- `--output_dir`: the directory where the compressed images will be saved;  

For the `jpegai_compress_directory.py` script, you also need to specify the path to the JPEG AI Reference software using the
`JPEG_AI_PATH` parameter in `utils/params.py`, and the path to the JPEG AI models location using the `--models_dir_name` argument.

## License
Please notice that the `jpegai_compress_directory.py` is heavily based on the JPEG AI Reference software, which is licensed under the following license:

```
JPEG AI Reference Software License
# The copyright in this software is being made available under the BSD
# License, included below. This software may be subject to other third party
# and contributor rights, including patent rights, and no such rights are
# granted under this license.


# Copyright (c) 2010-2022, ITU/ISO/IEC
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
# be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
# BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
# THE POSSIBILITY OF SUCH DAMAGE.
```