import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from options.test_ssl_options import TestOptions
from data import CreateTrgDataSSLLoader
from PIL import Image
import os.path as osp
import os
import numpy as np
import scipy
from model import CreateSSLModel
import torch.backends.cudnn as cudnn
from scipy.ndimage.morphology import binary_fill_holes
from skimage import morphology
from skimage.measure import label, regionprops

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)

def calculate_dice(gt_dir, pred_dir, devkit_dir=''):
    disc_dice = []
    cup_dice = []
    image_path_list = osp.join(devkit_dir, 'jiao.txt')
    img_ids = [i_id.strip() for i_id in open(image_path_list)]
    gt_imgs = [osp.join(gt_dir, x.split('.')[0]+'.png') for x in img_ids]
    pred_imgs = [osp.join(pred_dir, x.split('.')[0]+'.png') for x in img_ids]
    #import pdb;pdb.set_trace()
    for ind in range(len(gt_imgs)):
        prediction = np.asarray(Image.open(pred_imgs[ind]))
        mask = np.asarray(Image.open(gt_imgs[ind]))
        mask_binary = np.zeros((mask.shape[0], mask.shape[1],2))
        prediction_binary = np.zeros((prediction.shape[0], prediction.shape[1],2))
        mask_binary[mask < 200] = [1, 0]
        mask_binary[mask < 50] = [1, 1]
        prediction_binary[prediction < 200] = [1, 0]
        prediction_binary[prediction < 50] = [1, 1]
        disc_dice.append(dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0]))
        cup_dice.append(dice_coef(mask_binary[:,:,1], prediction_binary[:,:,1]))
    return sum(disc_dice) / (1. * len(disc_dice)), sum(cup_dice) / (1. * len(cup_dice))

def get_bool(img):
    return img == 1

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist)*2 / (hist.sum(1) + hist.sum(0))

def get_bool(img, type = 0):
    if type == 0:
        return img == 0
    if type == 1:
        return img == 1

def get_largest_fillhole(binary):
    label_image = label(binary)
    regions = regionprops(label_image)
    area_list = []
    for region in regions:
        area_list.append(region.area)
    if area_list:
        idx_max = np.argmax(area_list)
        binary[label_image != idx_max + 1] = 0
    return scipy.ndimage.binary_fill_holes(np.asarray(binary).astype(int))


def main():
    opt = TestOptions()
    args = opt.initialize()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = CreateSSLModel(args)
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()
    targetloader = CreateTrgDataSSLLoader(args)

    #predicted_label = np.zeros((len(targetloader), 1634, 1634))
    predicted_label = np.zeros((len(targetloader), 512, 512))
    #predicted_prob = np.zeros((len(targetloader), 1634, 1634))
    predicted_prob = np.zeros((len(targetloader), 512, 512))
    image_name = []

    for index, batch in enumerate(targetloader):
        if index % 10 == 0:
            print('%d processd' % index)
        image, _, name = batch
        image = Variable(image).cuda()
        _, output, _, _ = model(image, ssl=True)
        output = nn.functional.softmax(output/args.alpha, dim=1)
        #output = nn.functional.upsample(output, (1634, 1634), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = nn.functional.upsample(output, (512, 512), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        cup_mask = get_bool(label, 0)
        disc_mask = get_bool(label, 0) + get_bool(label, 1)
        disc_mask = disc_mask.astype(np.uint8)
        cup_mask = cup_mask.astype(np.uint8)
        for i in range(5):
            disc_mask = scipy.signal.medfilt2d(disc_mask, 19)
            cup_mask = scipy.signal.medfilt2d(cup_mask, 19)
        disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        disc_mask = get_largest_fillhole(disc_mask)
        cup_mask = get_largest_fillhole(cup_mask)

        disc_mask = morphology.binary_dilation(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
        cup_mask = morphology.binary_dilation(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1

        disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8)  # return 0,1
        cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
        label = disc_mask + cup_mask
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])

    thres = []
    for i in range(3):
        x = predicted_prob[predicted_label == i]
        if len(x) == 0:
            thres.append(0)
            continue
        x = np.sort(x)
        #print(i,x)
        print(np.sum(x<0.6),'/',len(x),np.sum(x<0.6)/len(x))
        #thres.append(x[np.int(np.round(len(x) *int(args.p[i])))])
        thres.append(0.9)
    thres = np.array(thres)
    print(thres)
   # thres[thres > 0.6] = 0.6

    color_labels = {0:255, 1:128, 2:0}
    for index in range(len(targetloader)):
        name = image_name[index]
        label = predicted_label[index]# label_min:0 label_max=2
        prob = predicted_prob[index] # prob_min:0.5 prob_max=1.0
        for i, color in color_labels.items():
            label[label == i] =color
            label[(prob > thres[i]) * (label == i)] = color
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output.astype(np.uint8))
        #name = name.split('.')[0] + '.png'
        name = name.split('_')[0]
        output.save('%s/%s' % (args.save, name))
    disc_dice, cup_dice = calculate_dice(args.gt_dir, args.save, args.devkit_dir)
    print('===> disc_dice:' + str(round(disc_dice,3)) + '\t' +'cup_dice:' + str(round(cup_dice,3)))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()

