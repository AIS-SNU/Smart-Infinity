#!/bin/bash

DS_BUILD_OPS=1 pip install deepspeed
conda install -c conda-forge regex -y

git clone https://github.com/NVIDIA/apex.git
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

# Write environemnt name of your conda
ENV= 
# Backup for original deepseed
cp -r ~/miniconda3/envs/${ENV}/lib/python3.9/site-packages/deepspeed  ~/miniconda3/envs/${ENV}/lib/python3.9/site-packages/_deepspeed

# SmartInfinity
rm -rf ~/miniconda3/envs/${ENV}/lib/python3.9/site-packages/deepspeed

ln -s ../deepspeed ~/miniconda3/envs/${ENV}/lib/python3.9/site-packages/deepspeed
