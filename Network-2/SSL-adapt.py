'''
Created by SJWANG  07/21/2018
For refuge image segmentation
'''

import numpy as np
import scipy.io as sio
import scipy.misc
from keras.preprocessing import image
from skimage import transform
from skimage.measure import label, regionprops
from time import time
from Utils.utils_get_disc_area import pro_process, BW_img, disc_crop
from PIL import Image
from matplotlib.pyplot import imshow, imsave
from skimage import morphology
from tqdm import tqdm
import cv2
import os
from Model.models import Model_CupSeg, Model_DiscSeg
from keras.applications import imagenet_utils
from skimage.measure import label, regionprops
import random
import tensorflow as tf
import scipy
from skimage.transform import resize
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, help = 'refuge-new, Drishti-GS', default = 'refuge-new')
#parser.add_argument('--label_path', type = str, default = '/extracephonline/medai_data2/zhengdzhang/eyes/qikan/scratch/source/mask/')
parser.add_argument('--label_path', type = str, default = '/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/disc_small/mask/')
#parser.add_argument('--list_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/refuge/level_0.5-0.7/select_0.1/jiao.txt', help="Path to the file listing the images in the source dataset.")
parser.add_argument('--list_path', type = str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/disc_small/crop.txt')
#parser.add_argument('--data-list-target', type=str, default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/target.txt', help="Path to the file listing the images in the source dataset.")
#parser.add_argument('--load_selected_samples', type = str, help = 'dir to load selected samples', default = '../path/to/dataset-new/source/eyesgan/level_0.5-0.7/noise_labels_0.1/selected_sample1.txt')
parser.add_argument('--load_selected_samples', type = str, help = 'dir to load selected samples', default = None)
#parser.add_argument('--save_selected_samples', type = str, help = 'dir to save selected samples', default = None)
#parser.add_argument('--save_selected_samples', type = str, help = 'dir to save selected samples', default = './data/refuge-new/level_0.5-0.7/noise_labels_0.5/selected_sample_0.1.txt')
#parser.add_argument('--noise_selected_samples', type = str, help = 'dir to save noise samples', default = './data/refuge-new/level_0.5-0.7/noise_labels_0.5/noise_sample_0.1.txt')
parser.add_argument('--load_from', type=str, default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/weights/refuge-new/DA_patch/Generator/generator_50.h5')
#parser.add_argument('--weight-root', type=str, default='../weights/refuge-new/level_0.5-0.7/noise_labels_0.5/select_0.1')
#parser.add_argument('--data_img_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/weights/refuge-new/source/disc_small/image/')
#parser.add_argument('--data_img_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/scratch/source/image/')
parser.add_argument('--data_img_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/disc_small/image/')
#parser.add_argument('--total_number', type = int, help = 'total number ', default = 200)
#parser.add_argument('--epoch', type = int, help = 'total number ', default = 1)
#parser.add_argument('--remember_rate', type = float, help = 'remember rate ', default = 0.1)
#parser.add_argument('--data_save_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/refuge-new/level_0.5-0.7/noise_labels_0.5/select_0.1/')
parser.add_argument('--data_save_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/result_0/')
#parser.add_argument('--mask_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/weights/refuge-new/source/disc_small/mask/')
#parser.add_argument('--mask_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/scratch/source/mask/')
parser.add_argument('--mask_path', type=str, default='/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/disc_small/mask/')
args = parser.parse_args()
print(args)


def save_img(data_save_path, img_name, prob_map, err_coord, crop_coord, DiscROI_size, org_img_size, threshold=0.5):
    path = os.path.join(data_save_path, img_name)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    disc_map = resize(prob_map[:, :, 0], (DiscROI_size, DiscROI_size)) # (640,640)
    cup_map = resize(prob_map[:, :, 1], (DiscROI_size, DiscROI_size)) # (640,640)

    disc_mask = (disc_map > threshold) # return binary mask
    cup_mask = (cup_map > threshold)
    disc_mask = disc_mask.astype(np.uint8)
    cup_mask = cup_mask.astype(np.uint8)

    for i in range(5):
        disc_mask = scipy.signal.medfilt2d(disc_mask, 7)
        cup_mask = scipy.signal.medfilt2d(cup_mask, 7)
    disc_mask = morphology.binary_erosion(disc_mask, morphology.diamond(7)).astype(np.uint8) # return 0,1
    cup_mask = morphology.binary_erosion(cup_mask, morphology.diamond(7)).astype(np.uint8) # return 0,1
    disc_mask = get_largest_fillhole(disc_mask)
    cup_mask = get_largest_fillhole(cup_mask)

    disc_mask = morphology.binary_dilation(disc_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1
    cup_mask = morphology.binary_dilation(cup_mask, morphology.diamond(7)).astype(np.uint8)  # return 0,1

    disc_mask = get_largest_fillhole(disc_mask).astype(np.uint8) # return 0,1
    cup_mask = get_largest_fillhole(cup_mask).astype(np.uint8)
    ROI_result = disc_mask + cup_mask #(640,640)
    ROI_result[ROI_result < 1] = 255
    ROI_result[ROI_result < 2] = 128
    ROI_result[ROI_result < 3] = 0

    Img_result = np.zeros((org_img_size[0], org_img_size[1], 3), dtype=int) + 255
    Img_result[crop_coord[0]:crop_coord[1], crop_coord[2]:crop_coord[3], 0] = ROI_result[err_coord[0]:err_coord[1],
                                                                              err_coord[2]:err_coord[3]]
    # for submit
    return Img_result # (512,512,3)

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


def get_bbox_source(img):
    img = np.array(img)
    h, w = img.shape
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    r = rmax - rmin
    c = cmax - cmin
    x = np.round((rmax+rmin)/2)
    y = np.round((cmax+cmin)/2)
    x1 = x-256
    x2 = x+256
    y1 = y-256
    y2 = y+256
    if x1 < 0:
        x2 += -x1
        x1 = 0
    if y1 < 0:
        y2 += -y1
        y1 = 0
    return np.uint16(x1), np.uint16(x2), np.uint16(y1), np.uint16(y2)


def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)


def calculate_dice(mask_path, prediction_path, list_path):
    disc_dice = []
    cup_dice = []
    #file_test_list = [file for file in os.listdir(mask_path) if file.lower().endswith('bmp')]
    file_test_list = [id.strip() for id in open(list_path)]
    for filename in file_test_list:
        mask = np.asarray(image.load_img(os.path.join(mask_path, filename + '.png'), grayscale=True)) # mask [512,512] (0,128,255)
        prediction = np.asarray(image.load_img(os.path.join(prediction_path, filename.split('.')[0]+'.png'), grayscale=True))# prediction [512,512] (0,128,255)
        mask_binary = np.zeros((mask.shape[0], mask.shape[1],2))
        prediction_binary = np.zeros((prediction.shape[0], prediction.shape[1],2))
        mask_binary[mask < 200] = [1, 0]
        mask_binary[mask < 50] = [1, 1]
        prediction_binary[prediction < 200] = [1, 0]
        prediction_binary[prediction < 50] = [1, 1]
        disc_dice.append(dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0]))
        cup_dice.append(dice_coef(mask_binary[:,:,1], prediction_binary[:,:,1]))
    return sum(disc_dice) / (1. * len(disc_dice)), sum(cup_dice) / (1. * len(cup_dice))


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
   
    ''' parameters '''
    DiscROI_size = 640
    DiscSeg_size = 640
    CDRSeg_size = 512

    data_type = '.png'
    save_data_type = 'png'
    dataset = "refuge-new"
    #dataset = "Drishti-GS-new"
    #dataset_t = "Drishti-GS"
    #dataset = "RIM-ONE-r3-new"
    #dataset_t = "RIM-ONE-r3"
    scale = False
    both = True

    #phase = "target"
    phase = "source"
    #phase = "train"
    #data_img_path = '../weights/' + dataset + '/'+ phase +'/disc_small/image/'
    #data_save_path = '../correction/' + dataset + '/level_0.5-0.7/noise_labels_0.5/select_0.1/'
    #mask_path = '../weights/' + dataset + '/'+ phase +'/disc_small/mask/'
    #label_path = '../weights/' + dataset + '/'+ phase +'/disc_small/mask/'
    # change location to wherever you like
    #load_from = "/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/weights/refuge-new/DA_patch/Generator/generator_50.h5"
    #list_path = "/extracephonline/medai_data2/zhengdzhang/eyes/qikan/correction/refuge/level_0.5-0.7/select_0.1/jiao.txt"

    if not os.path.exists(args.data_save_path):
        print("Creating save path {}\n\n".format(args.data_save_path))
        os.makedirs(args.data_save_path)

    #file_test_list = [file for file in os.listdir(data_img_path) if file.lower().endswith(data_type)]
    file_test_list = [id.strip() for id in open(args.list_path)]
    #random.shuffle(file_test_list)
    #import pdb;pdb.set_trace()
    print(file_test_list)
    print(str(len(file_test_list)))

    ''' create model and load weights'''
    DiscSeg_model = Model_DiscSeg(inputsize=DiscSeg_size)
    DiscSeg_model.load_weights('../weights/Model_DiscSeg_pretrain.h5')  # download from M-Net

    CDRSeg_model = Model_CupSeg(input_shape = (CDRSeg_size, CDRSeg_size, 3), classes=2, backbone='mobilenetv2')
    CDRSeg_model.load_weights(args.load_from)

    ''' predict each image '''
    for lineIdx in tqdm(range(0, len(file_test_list))):
        temp_txt = [elt.strip() for elt in file_test_list[lineIdx].split(',')]
        # load image
        org_img = np.asarray(image.load_img(args.data_img_path + temp_txt[0] + '.png')) #(512,512,3)
        #true_label = Image.open(os.path.join(label_path+ temp_txt[0][:-4]+'.bmp'))
        true_label = Image.open(os.path.join(args.label_path + temp_txt[0] + '.png'))
        #import pdb;pdb.set_trace()
        ind = {0: 1, 128: 1}
        s = np.asarray(true_label, np.float32)
        s_copy = np.zeros(s.shape, dtype=np.float32)
        for k, v in ind.items():
            s_copy[s == k] = v
        x1, x2, y1, y2 = get_bbox_source(s_copy)
        print(temp_txt,x1, x2, y1, y2)
        output = np.zeros((512, 512), dtype=int) + 255

        # Disc region detection by U-Net
        temp_img = transform.resize(org_img, (DiscSeg_size, DiscSeg_size, 3))*255
        temp_img = np.reshape(temp_img, (1,) + temp_img.shape) #(1,640,640,3)
        #import pdb;pdb.set_trace()
        [prob_10] = DiscSeg_model.predict([temp_img]) #(640,640,1)

        disc_map = BW_img(np.reshape(prob_10, (DiscSeg_size, DiscSeg_size)), 0.5) # (640,640)
        regions = regionprops(label(disc_map))
        C_x = int(regions[0].centroid[0] * org_img.shape[0] / DiscSeg_size) # 232
        C_y = int(regions[0].centroid[1] * org_img.shape[1] / DiscSeg_size) # 278

        ''' get disc region'''
        disc_region, err_coord, crop_coord = disc_crop(org_img, DiscROI_size, C_x, C_y)
        disc_region_img = disc_region.astype(np.uint8) #(640,640,3)
        disc_region_img = Image.fromarray(disc_region_img)

        if not os.path.exists(args.data_save_path):
            os.makedirs(args.data_save_path)
        #disc_region_img.save(data_save_path + temp_txt[0][:-4] + '.png')
        disc_region_img.save(args.data_save_path + temp_txt[0] + '.png')
        run_time_start = time()

        temp_img = pro_process(disc_region, CDRSeg_size) #(512,152,3)
        temp_img = np.reshape(temp_img, (1,) + temp_img.shape)#(1,512,152,3)
        temp_img = imagenet_utils.preprocess_input(temp_img.astype(np.float32), mode='tf')
        prob = CDRSeg_model.predict(temp_img) #(1,512,152,2)

        run_time_end = time()
        print(' Run time MNet: ' + str(run_time_end - run_time_start) + '   Img number: ' + str(lineIdx + 1))
        #import pdb;pdb.set_trace()
        prob_map_1 = np.squeeze(prob) #(512,152,2)
        #s_label = tf.one_hot(true_label, 2) 

        
        #import pdb;pdb.set_trace()
        s_label = np.zeros((s.shape[0], s.shape[1], 2))
        s_label[s < 200] = [1, 0]
        s_label[s < 50] = [1, 1]
        n1_prob_map = np.load("/extracephonline/medai_data2/zhengdzhang/eyes/qikan/cai/output_crop_1.npy")
        prob_map = n1_prob_map * s_label + (1 - n1_prob_map) * prob_map_1
        prob_map**=(1/0.5)
        


        #img_name = temp_txt[0][:-4] + '.png'
        img_name = temp_txt[0] + '.png'
        #import pdb;pdb.set_trace()
        Img_result = save_img(
                     data_save_path=args.data_save_path, img_name=img_name, prob_map=prob_map, err_coord=err_coord,
                     crop_coord=crop_coord, DiscROI_size=DiscROI_size,
                     org_img_size=org_img.shape, threshold=0.5)
        output[x1:x2,y1:y2] = Img_result[:, :, 0]
        output = Image.fromarray(output.astype(np.uint8))
        output.save('%s/%s' % (args.data_save_path, img_name))
        if dataset == 'RIM-ONE-r3-new':
            output = Image.fromarray(Img_result[:, :, 0].astype(np.uint8))
            output = output.resize((512, 512), Image.NEAREST)
            output.save('%s/%s' % (args.data_save_path, img_name))
    disc_dice, cup_dice = calculate_dice(args.label_path, args.data_save_path, args.list_path)
    print("disc dice:" + str(round(disc_dice, 5)), "cup dice:" + str(round(cup_dice, 5)))
    print(args.load_from)
    print(args.data_save_path)
