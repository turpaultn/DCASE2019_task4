#!/bin/bash
conda create -y -n dcase2019 python=3.6
source activate dcase2019
conda install -y pandas h5py scipy
conda install -y pytorch torchvision cudatoolkit=9.0 -c pytorch # for gpu install (or cpu in MAC)
# conda install pytorch-cpu torchvision-cpu -c pytorch (cpu linux)
conda install -y pysoundfile librosa youtube-dl tqdm -c conda-forge
conda install -y ffmpeg -c conda-forge

pip install dcase_util
pip install sed-eval
