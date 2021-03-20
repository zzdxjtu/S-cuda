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

gt = "./dataset/source/labels"
pre = ""
dir = "./dataset"

def dice_coef(y_true, y_pred):     
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)

def calculate_dice(gt_dir, pred_dir, devkit_dir=''):
    disc_dice = []
    cup_dice = []
    image_path_list = osp.join(devkit_dir, 'source.txt')
    img_ids = [i_id.strip() for i_id in open(image_path_list)]
    gt_imgs = [osp.join(gt_dir, x.split('.')[0]+'.bmp') for x in img_ids]
    pred_imgs = [osp.join(pred_dir, x.split('.')[0]+'.bmp') for x in img_ids]
    print(len(gt_imgs))
    for ind in range(len(gt_imgs)):
        prediction = np.asarray(Image.open(pred_imgs[ind]))
        print(pred_imgs[ind])
        mask = np.asarray(Image.open(gt_imgs[ind]))
        mask_binary = np.zeros((mask.shape[0], mask.shape[1],2))
        prediction_binary = np.zeros((prediction.shape[0], prediction.shape[1],2))
        mask_binary[mask < 200] = [1, 0]
        mask_binary[mask < 50] = [1, 1]
        prediction_binary[prediction < 200] = [1, 0]
        prediction_binary[prediction < 50] = [1, 1]
        disc_dice.append(dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0]))
        cup_dice.append(dice_coef(mask_binary[:,:,1], prediction_binary[:,:,1]))
    print(disc_dice)
    print(cup_dice)
    return sum(disc_dice) / (1. * len(disc_dice)), sum(cup_dice) / (1. * len(cup_dice))

disc_dice, cup_dice = calculate_dice(gt, pre, dir)
print('===> disc_dice:' + str(round(disc_dice,3)) + '\t' +'cup_dice:' + str(round(cup_dice,3)))
