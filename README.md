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
project
│   README.md
│   file001.txt    
│
└───folder1
│   │   file011.txt
│   │   file012.txt
│   │
│   └───subfolder1
│       │   file111.txt
│       │   file112.txt
│       │   ...
│   
└───folder2
    │   file021.txt
    │   file022.txt
```
