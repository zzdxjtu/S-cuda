import torch
import torch.nn as nn
from torch.autograd import Variable
from options.test_options import TestOptions
from data import CreateTrgDataLoader
from PIL import Image
import scipy
import os.path as osp
import os
import numpy as np
from model import CreateModel
from skimage.measure import label, regionprops
from skimage import morphology

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)

def calculate_dice(gt_dir, pred_dir, devkit_dir=''):
    disc_dice = []
    cup_dice = []
    image_path_list = osp.join(devkit_dir, 'test.txt')
    img_ids = [i_id.strip() for i_id in open(image_path_list)]
    gt_imgs = [osp.join(gt_dir, x.split('.')[0]+'.bmp') for x in img_ids]
    pred_imgs = [osp.join(pred_dir,'color_'+x.split('.')[0]+'.png') for x in img_ids]
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


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist)*2 / (hist.sum(1) + hist.sum(0))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)

def get_bool(img, type = 0):
    if type == 0:
        return img == 0
    if type == 1:
        return img == 1

def main():
    opt = TestOptions()
    args = opt.initialize()    
    
    if not os.path.exists(args.save):
        os.makedirs(args.save)
        
    model = CreateModel(args) 
    
    model.eval()
    model.cuda()    
    targetloader = CreateTrgDataLoader(args)
    id_to_trainid = {2 : 0, 1 : 128, 0 : 255}
    for index, batch in enumerate(targetloader):
        if index % 10 == 0:
            print ('%d processd' % index)
        image, _, name = batch
        _, output, _, _ = model(Variable(image).cuda()) #[1,3,129,129]
        #import pdb;pdb.set_trace()
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (1634, 1634), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1,2,0) #(1634,1634,3)
        '''
        output_crop = output[14:526, 626:1138, 0:2]
        np.save("/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/output_crop_1.npy", output_crop)
        #crop_img = Image.fromarray(output_crop)
        #crop_img.save("/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/crop_img.png")
        '''
        output_nomask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8) #(1644,1634) unique:[0,1,2]
        cup_mask = get_bool(output_nomask, 0)
        disc_mask = get_bool(output_nomask, 0) + get_bool(output_nomask, 1)
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
        output_nomask = disc_mask + cup_mask

        output_col = np.ones(output_nomask.shape, dtype=np.float32)
        for k, v in id_to_trainid.items():
            output_col[output_nomask == k] = v
        output_col = Image.fromarray(output_col.astype(np.uint8))
        output_nomask = Image.fromarray(output_nomask)    
        name = name[0].split('.')[0] + '.png'
        output_nomask.save('%s/%s' % (args.save, name))
        output_col.save('%s/color_%s' % (args.save, name)) 

    disc_dice, cup_dice = calculate_dice(args.gt_dir, args.save, args.devkit_dir)
    print('===> disc_dice:' + str(round(disc_dice,3)) + '\t' +'cup_dice:' + str(round(cup_dice,3)))

if __name__ == '__main__':
       
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  
    main()
    
    

