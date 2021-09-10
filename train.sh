#!/bin/bash -ex

# Common packages are already installed on the compute server:
#  - numpy pandas sklearn scipy ipdb
#  - torch torchvision torchaudio
#  - albumentations imgaug jax pytorch-lightning

# Need an additional package? Install it here via:
#  pip3 install package-name

# Edit the line below to run your experiment (this is just an example). Note:
#  - This script will be run from your output directory
#  - Imported Data is accessible via the relative path ../input/


export TF_CPP_MIN_LOG_LEVEL=2

python3 ../code/main.py --epochs 10 -b 16 --seed 0
