import cv2
import matplotlib.pyplot as plt
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#from options.test_ssl_options import TestOptions
#from data import CreateTrgDataSSLLoader
from PIL import Image
import os.path as osp
import os
import numpy as np
import scipy
#from model import CreateSSLModel
import torch.backends.cudnn as cudnn
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from skimage.measure import label, regionprops

'''
#gt = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/new-dataset_2/mask"
gt = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/source/labels"
#pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/weights/refuge-new/source/disc_small/level_0.2-0.3/noise_mask_0.9"
#pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/new-dataset_2/level_0.5-0.7/noise_mask_0.9_scratch"
#pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/source/level_0.2-0.3/noise_labels_0.9"
#pre = "/extracephonline/medai_data2/zhengdzhang/eyes/path/to/dataset/source/level_0.5-0.7/noise_labels_0.9"
pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/new-dataset/level_0.5-0.7/noise_labels_0.9"
#pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/scratch/source/level_0.2-0.3/noise_mask_0.9"
#dir = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction_scratch/update_list/level_0.5-0.7/noise_labels_0.9/select_0.3"
dir = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/refuge-new/level_0.5-0.7/noise_labels_0.9/select_0.9"
#dir = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/refuge/level_0.5-0.7/select_0.4"
'''
gt = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/scgm/data/source/labels"
#pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/new-dataset/level_0.5-0.7/noise_labels_0.1_new"
pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/scgm/data/source/level_0.5-0.7/noise_labels_0.1"
#pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/source/level_0.5-0.7/noise_labels_0.9"
dir = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/scgm/data/source/level_0.5-0.7/noise_labels_0.1"

#gt = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/new-dataset_2/mask"
#pre = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/new-dataset_2/level_0.2-0.3/noise_mask_0.1_new"
#dir = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/path/to/source/level_0.2-0.3/noise_labels_0.1"
#dir = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction_scratch/update_list/level_0.2-0.3/noise_labels_0.9/select_0.5"


def get_contours(img):

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours[0]


def calculate_hausdorff(gt_dir, pred_dir, devkit_dir=''):
    distance_disc = []
    distance_cup = []
    image_path_list = osp.join(devkit_dir, 'noise_label.txt')
    img_ids = [i_id.strip() for i_id in open(image_path_list)]
    gt_imgs = [osp.join(gt_dir, x.split('.')[0]+'.png') for x in img_ids]
    pred_imgs = [osp.join(pred_dir, x.split('.')[0]+'.png') for x in img_ids]
    #import pdb;pdb.set_trace()
    print(len(gt_imgs))
    hausdorff_sd = cv2.createHausdorffDistanceExtractor()
    for ind in range(len(gt_imgs)):
        '''
        prediction = np.asarray(Image.open(pred_imgs[ind]))
        print(pred_imgs[ind])
        mask = np.asarray(Image.open(gt_imgs[ind]))
        mask_binary = np.zeros((mask.shape[0], mask.shape[1],2))
        prediction_binary = np.zeros((prediction.shape[0], prediction.shape[1],2))
        mask_binary[mask < 200] = [1, 0]
        mask_binary[mask < 50] = [1, 1]
        prediction_binary[prediction < 200] = [1, 0]
        prediction_binary[prediction < 50] = [1, 1]
        #disc_dice.append(dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0]))
        #cup_dice.append(dice_coef(mask_binary[:,:,1], prediction_binary[:,:,1]))
        '''
        num = 0
        gt_img = cv2.imread(gt_imgs[ind])
        gt_img_disc = cv2.imread(gt_imgs[ind])
        #[m, n] = size(gt_img)
        for i in range(3):
            gt_img[:,:,i][gt_img[:,:,i]>50] = 255
            gt_img[:,:,i][gt_img[:,:,i]>50] = 255
            gt_img[:,:,i][gt_img[:,:,i]>50] = 255
        print(gt_img[:,:,0])
        #print(gt_img.size)
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
        '''
        distance_disc.append(hausdorff_sd.computeDistance(get_contours(mask_binary[:,:,0]), get_contours(prediction_binary[:,:,0])))
        distance_disc.append(hausdorff_sd.computeDistance(get_contours(mask_binary[:,:,1]), get_contours(prediction_binary[:,:,1])))
        '''
    print(distance_disc)
    print(distance_cup)
    return sum(distance_disc) / (1. * len(distance_disc)), sum(distance_cup) / (1. * len(distance_cup))

distance_disc, distance_cup = calculate_hausdorff(gt, pre, dir)
print("distance_disc = " + str(round(distance_disc, 3)) + ", distance_cup = " + str(round(distance_cup, 3)))

