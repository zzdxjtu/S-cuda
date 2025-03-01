import cv2
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import os.path as osp
import os
import numpy as np
import scipy
import torch.backends.cudnn as cudnn
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from skimage.measure import label, regionprops

gt = ".\dataset\source\labels"
pre = ""
dir = ".\dataset"

def get_contours(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours[0]

def calculate_hausdorff(gt_dir, pred_dir, devkit_dir=''):
    distance_disc = []
    distance_cup = []
    image_path_list = osp.join(devkit_dir, 'source.txt')
    img_ids = [i_id.strip() for i_id in open(image_path_list)]
    gt_imgs = [osp.join(gt_dir, x.split('.')[0]+'.png') for x in img_ids]
    pred_imgs = [osp.join(pred_dir, x.split('.')[0]+'.png') for x in img_ids]
    print(len(gt_imgs))
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    for ind in range(len(gt_imgs)):
        num = 0
        gt_img = cv2.imread(gt_imgs[ind])
        gt_img_disc = cv2.imread(gt_imgs[ind])
        for i in range(3):
            gt_img[:,:,i][gt_img[:,:,i]>50] = 255
            gt_img[:,:,i][gt_img[:,:,i]>50] = 255
            gt_img[:,:,i][gt_img[:,:,i]>50] = 255
        print(gt_img[:,:,0])
        pred_img = cv2.imread(pred_imgs[ind])
        pred_img_disc = cv2.imread(pred_imgs[ind])
        for i in range(3):
            pred_img[:,:,i][pred_img[:,:,i]>50] = 255
            pred_img[:,:,i][pred_img[:,:,i]>50] = 255
            pred_img[:,:,i][pred_img[:,:,i]>50] = 255
        gt_i = get_contours(gt_img)
        gt_i_disc = get_contours(gt_img_disc)
        pred_i = get_contours(pred_img)
        pred_i_disc = get_contours(pred_img_disc)         
        distance_cup.append(hausdorff_sd.computeDistance(gt_i, pred_i))
        distance_disc.append(hausdorff_sd.computeDistance(gt_i_disc, pred_i_disc))
    print(distance_disc)
    print(distance_cup)
    return sum(distance_disc) / (1. * len(distance_disc)), sum(distance_cup) / (1. * len(distance_cup))

distance_disc, distance_cup = calculate_hausdorff(gt, pre, dir)
print("distance_disc = " + str(round(distance_disc, 3)) + ", distance_cup = " + str(round(distance_cup, 3)))

