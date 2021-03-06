# S-cuda: Self-Cleansing Unsupervised Domain Adaptation for Medical Image Segmentation
This repository provides code for the paper, S-CUDA: Self-Cleansing Unsupervised Domain Adaptation for Medical Image Segmentation. Please read our paper to understand our proposed method.
## Pipeline
![image](https://user-images.githubusercontent.com/38779372/110201691-84edaa00-7e9f-11eb-94bb-1043dc82eba7.png)
## Getting started
### Environments
* python 3.5
* tensorflow 1.4.0
* keras 2.2.0
* pytorch 0.4.0
* CUDA 9.2
### Packages
* tqdm
* skimage
* opencv
* scipy
* matplotlib
### Datasets
Download from [Refuge](https://refuge.grand-challenge.org/), prepare dataset in data directory as follows.
```
S-cuda
│   Network-1
|   Network-2
│   scripts
└───dataset
│   │   source
│   │   │   images
│   │   │   labels
│   │
│   └───target
│   │   │   images
│   │   │   labels
│   │   │   pseudo_label 
│   │ 
│   └───test
│   │   │   images
│   │   │   labels
│   │
│   └───source.txt
│   │   target.txt
│   │   test.txt
│        
└───README.md
```
### Initial weights and pre-trained model
Initial weights and pre-trained model download link:
* [Initial weights](https://pan.baidu.com/s/1EUfmEAyUn6NdBbJ7Pq8C_Q), code:4y69
* [Pre-trained model](https://pan.baidu.com/s/1R05swgfBVpXSscVI07mxpg), code:9koj
```
unzip Initial_weights.zip 
unzip Pre-trained model.zip 
```
### Running
__0.Clone this repo:__  
```
git clone https://github.com/zzdxjtu/S-cuda.git
cd S-cuda
```
__1.Train:__  
All training script is stored in scripts directory.
```
sh scripts/run1.sh
sh scripts/run2.sh
sh scripts/run3.sh
sh scripts/run4.sh
```
Before training, please check whether all the model weight and dataset path is correct.  
__2.Test:__  
```
cd S-cuda
python Network-1/evaluation.py
```
### Acknowledge  
Some codes are revised according to [liyunsheng13/BDL] (https://github.com/liyunsheng13/BDL)and [EmmaW8/pOSAL](https://github.com/EmmaW8/pOSAL). Thank them very much.
