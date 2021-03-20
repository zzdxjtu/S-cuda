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
```
##The two networks select the noise data and clean data simultaneously, and then input the clean data into each other's network for finetune, and you can change the remember rate and noise rate according to the noise ratio of the source training set.
```
sh scripts/run2.sh  
```
##The two networks select the noise data and clean data simultaneously, and then input the clean data into each other's network for finetune.
```
sh scripts/run3.sh  
```
##Correct the common part of the noise data selected by the two networks, and then the two networks select the noise data and clean data simultaneously, at last input the clean data into each other's network for finetune.
```
sh scripts/run4.sh  
```
##Correct the common part of the noise data selected by the two networks, and then the two networks select the noise data and clean data simultaneously, at last input the clean data into each other's network for finetune.  
Before training, please check whether all the model weight and dataset path is correct.  
__2.Test:__  
```
cd S-cuda
python Network-1/evaluation.py
```
__3.Performance__  

**Pretrained**  
Level_0.5-0.7/noise_labels_0.5 | select_0.1 | select_0.2 | select_0.3 | select_0.4  
---- | ---- | ----| ---- | ----  
Disc_dice | 0.95 | 0.95 | 0.948 | 0.95  
Cup_dice | 0.882 | 0.873 | 0.871 | 0.871  

**Scratch**  
Level_0.5-0.7/noise_labels_0.5 | select_0.1 | select_0.2 | select_0.3 | select_0.4   
---- | ---- | ----| ---- | ----  
Disc_dice | 0.947 | 0.943 | 0.943 | 0.941  
Cup_dice | 0.889 | 0.886 | 0.886 | 0.872  
### Supplementary notes  
```
boudary.ipynb  ##Calculate the weight map of the optic cup, optic disc, and background  
calculate_dice.py  ##Calculate dice coefficient  
get_contour.ipynb  ##Obtain the edge contour of the target object  
hausdorff_dis.py  ##Calculate hausdorff distance  
noise_label.ipynb  ##Generate labels with different levels of noise and different proportions, including corrosion, expansion, deformation operations
```
### Acknowledge  
Some of our codes are referring to [liyunsheng13/BDL](https://github.com/liyunsheng13/BDL) and [EmmaW8/pOSAL](https://github.com/EmmaW8/pOSAL). Thanks for their helpful works.
