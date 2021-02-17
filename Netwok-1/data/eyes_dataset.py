import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from model.loss import caculate_weight_map

class eyestargetDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), set='val'):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.mean = mean
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        self.set = set

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):

        #name = self.img_ids[index]
        name = self.img_ids[index] + '_fake' + '.png'
        #name = self.img_ids[index] + '.png'
        #name = self.img_ids[index] + '.jpg'
        image = Image.open(osp.join(self.root, "images/%s" % name)).convert('RGB')
        #import pdb;pdb.set_trace()
        #print(osp.join(self.root, "image/%s" % name))
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)

        image = np.asarray(image, np.float32)

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= self.mean
        image = image.transpose((2, 0, 1))

        return image.copy(), np.array(size), name


class eyessourceDataSet(data.Dataset):
    def __init__(self, root, list_path,load_selected_samples=None,  max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        if (load_selected_samples == None):
            self.img_ids = [i_id.strip() for i_id in open(list_path)]
        else:
            self.img_ids = [i_id.strip('\t')[0:5] for i_id in open(load_selected_samples)]
            print(self.img_ids)
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        self.id_to_trainid = {0: 0, 128: 1, 255: 2}
    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        name = self.img_ids[index]
        print(name)
        name_img = name.split('.')[0] + '_fake' + '.png'
        name_label = name.split('.')[0] + '.bmp'
        image = Image.open(osp.join(self.root, "images/%s" % name_img)).convert('RGB')
        label = Image.open(osp.join(self.root,"level_0.5-0.7", "noise_labels_0.9/%s" % name_label))
        label_new = Image.open(osp.join(self.root,"level_0.5-0.7", "noise_labels_0.9_scratch/%s" % name_label))
        #label = Image.open(osp.join(self.root,"labels/%s" % name_label))
        #print(image.size, label.size)
        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)
        label_new = label_new.resize(self.crop_size, Image.NEAREST)
        dis_weight = caculate_weight_map(label, weight_cof=30)
        dis_weight_new = caculate_weight_map(label_new, weight_cof=30)
        #print(image.size, label.size, dis_weight.shape)
        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)
        label_new = np.asarray(label_new, np.float32)
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)##255*array(1280, 720)
        label_copy_new = self.ignore_label * np.ones(label_new.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
            label_copy_new[label_new == k] = v            
        size = image.shape
        image = image[:, :, ::-1]  # change to BGR(1280*720*3)
        image -= self.mean
        image = image.transpose((2, 0, 1))
        return image.copy(), label_copy.copy(), label_copy_new.copy(), dis_weight, dis_weight_new, name.split('.')[0]
