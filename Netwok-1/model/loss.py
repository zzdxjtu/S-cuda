import torch.nn as nn
import torch
from skimage.measure import label, regionprops
import scipy.ndimage as ndimage
import numpy as np
from PIL import Image

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)

def calculate_dice(mask, prediction):
    mask = np.asarray(mask)
    mask_binary = np.zeros((mask.shape[0], mask.shape[1],2))
    prediction_binary = np.zeros((mask.shape[0], mask.shape[1], 2))
    mask_binary[mask == 1] = [1, 0]
    mask_binary[mask == 0] = [1, 1]
    prediction_binary [prediction == 1] = [1, 0]
    prediction_binary[prediction == 0] = [1, 1]
    disc_dice = dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0])
    cup_dice = dice_coef(mask_binary[:,:,1], prediction_binary[:,:,1])
    return round(disc_dice,3), round(cup_dice,3)

def get_obj_dis_weight(dis_map, w0=10, eps=1e-20):
    max_dis = np.amax(dis_map)
    std = max_dis / 2.58 + eps
    weight_matrix = w0 * np.exp(-1 * pow(dis_map, 2) / (2 * pow(std, 2)))
    return weight_matrix


def get_bck_dis_weight(dis_map, w0=10, eps=1e-20):
    max_dis = np.amax(dis_map)
    std = max_dis / 2.58 + eps
    weight_matrix = w0 * np.exp(-1 * pow((max_dis - dis_map), 2) / (2 * pow(std, 2)))
    return weight_matrix


def caculate_weight_map(mask, weight_cof=30):
    mask = mask.resize((1024, 1024), Image.NEAREST)
    mask = np.asarray(mask, np.float32)
    mask_cup = np.zeros(mask.shape, dtype=np.float32)
    ind = {0: 255, 128: 0, 255: 0}
    for k, v in ind.items():
        mask_cup[mask == k] = v

    mask_disc = np.zeros(mask.shape, dtype=np.float32)
    ind = {0: 255, 128: 200, 255: 0}
    for k, v in ind.items():
        mask_disc[mask == k] = v
    labeled, label_num = label(mask, neighbors=4, background=255, return_num=True)  # label_num = 2
    image_props = regionprops(labeled, cache=False)
    dis_trf = ndimage.distance_transform_edt(255 - mask)
    adaptive_cup_dis_weight = np.zeros(mask.shape, dtype=np.float32)
    adaptive_disc_dis_weight = np.zeros(mask.shape, dtype=np.float32)
    adaptive_cup_dis_weight = adaptive_cup_dis_weight + (mask_cup / 255) * weight_cof
    adaptive_disc_dis_weight = adaptive_disc_dis_weight + (mask_disc / 255) * weight_cof
    adaptive_bck_dis_weight = np.ones(mask.shape, dtype=np.float32)

    for num in range(1, label_num + 1):
        image_prop = image_props[num - 1]
        bool_dis = np.zeros(image_prop.image.shape)
        bool_dis[image_prop.image] = 1.0
        (min_row, min_col, max_row, max_col) = image_prop.bbox
        temp_dis = dis_trf[min_row: max_row, min_col: max_col] * bool_dis

        adaptive_cup_dis_weight[min_row: max_row, min_col: max_col] = adaptive_cup_dis_weight[min_row: max_row,
                                                                      min_col: max_col] + get_bck_dis_weight(
            temp_dis) * bool_dis
        adaptive_disc_dis_weight[min_row: max_row, min_col: max_col] = adaptive_disc_dis_weight[min_row: max_row,
                                                                       min_col: max_col] + get_bck_dis_weight(
            temp_dis) * bool_dis
        adaptive_bck_dis_weight[min_row: max_row, min_col: max_col] = adaptive_bck_dis_weight[min_row: max_row,
                                                                      min_col: max_col] + get_bck_dis_weight(
            temp_dis) * bool_dis

    # get weight map for loss
    bck_maxinum = np.max(adaptive_bck_dis_weight)
    bck_mininum = np.min(adaptive_bck_dis_weight)
    adaptive_bck_dis_weight = (adaptive_bck_dis_weight - bck_mininum) / (bck_maxinum - bck_mininum)

    obj_maxinum = np.max(adaptive_disc_dis_weight)
    obj_mininum = np.min(adaptive_disc_dis_weight)
    adaptive_disc_dis_weight = (adaptive_disc_dis_weight - obj_mininum) / (obj_maxinum - obj_mininum)
    adaptive_cup_dis_weight = (adaptive_cup_dis_weight - obj_mininum) / (obj_maxinum - obj_mininum)

    adaptive_cup_dis_weight = adaptive_cup_dis_weight[np.newaxis, :, :]
    adaptive_disc_dis_weight = adaptive_disc_dis_weight[np.newaxis, :, :]
    adaptive_bck_dis_weight = adaptive_bck_dis_weight[np.newaxis, :, :]
    adaptive_dis_weight = np.concatenate((adaptive_bck_dis_weight, adaptive_cup_dis_weight, adaptive_disc_dis_weight),axis=0)

    return adaptive_dis_weight


class WeightMapLoss(nn.Module):
    """
    calculate weighted loss with weight maps in two channels
    """
    def __init__(self, _eps=1e-20, _dilate_cof=1):
        super(WeightMapLoss, self).__init__()
        self._eps = _eps

    def forward(self, pred, target, weight):
        """
        target: The target map, LongTensor, unique(target) = [0 1]
        weight_maps: The weights for two channels，weight_maps = [weight_bck_map, weight_obj_map]
        method：Select the type of loss function
        """
        pred, target, weight = pred.float(), target.float(), weight.float()
        weight_disc, weight_cup, weight_bck = weight[:, 2, :, :], weight[:, 1, :, :], weight[:, 0, :, :]
        logit = torch.softmax(pred, dim=1)
        logit = logit.clamp(self._eps, 1. - self._eps)
        #loss = - weight_disc * torch.log(logit[:, 1, :, :]) - weight_cup * torch.log(logit[:, 0, :, :])
        loss = -1 * weight_bck * torch.log(logit[:, 1, :, :]) - weight_bck * torch.log(logit[:, 0, :, :])
        weight_sum = weight_bck +  weight_disc + weight_cup
        return loss.sum() / weight_sum.sum()

class WeightedBCEWithLogitsLoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()

    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        if weight is not None:
            loss = alpha * loss + beta * weight * loss
        return loss.mean()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    #import pdb;pdb.set_trace()
    n, c, h, w = prob.size()
    return (-torch.sum(torch.mul(prob, torch.log2(prob + 1e-30)), 1) / np.log2(c)).view(1, 1, h, w)



