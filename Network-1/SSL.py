import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data
from options.test_ssl_options import TestOptions
from data import CreateTrgDataSSLLoader
from PIL import Image
import json
import os.path as osp
import os
import numpy as np
from model import CreateSSLModel


def main():
    opt = TestOptions()
    args = opt.initialize()    

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    model = CreateSSLModel(args)
    model.eval()
    model.cuda()   
    targetloader = CreateTrgDataSSLLoader(args)

    predicted_label = np.zeros((len(targetloader), 1634, 1634))
    predicted_prob = np.zeros((len(targetloader), 1634, 1634))
    image_name = []
    for index, batch in enumerate(targetloader):
        if index % 100 == 0:
            print ('%d processd' % index)
        image, _, name = batch
        image = Variable(image).cuda()
        _, output, _, _ = model(image, ssl=True)
        output = nn.functional.softmax(output, dim=1)
        output = nn.functional.upsample(output, (1634, 1634), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1,2,0)
        #import pdb;pdb.set_trace() 
        label, prob = np.argmax(output, axis=2), np.max(output, axis=2)
        predicted_label[index] = label.copy()
        predicted_prob[index] = prob.copy()
        image_name.append(name[0])
        
    thres = []
    for i in range(3):
        x = predicted_prob[predicted_label==i]
        if len(x) == 0:
            thres.append(0)
            continue        
        x = np.sort(x)
        thres.append(x[np.int(np.round(len(x)*0.5))])
    thres = np.array(thres)
    thres[thres>0.9]=0.9
    
    color_labels ={0:0, 128:1, 255:2}
    for index in range(len(targetloader)):
        name = image_name[index]
        label = predicted_label[index]
        prob = predicted_prob[index]
        for color, i in color_labels.items():
            label[(prob>thres[i])*(label==i)] = color  
        output = np.asarray(label, dtype=np.uint8)
        output = Image.fromarray(output)
        name = name.split('.')[0]+'.png'
       # arr = name.split('/')[-1].split('.')[0] + '.npy'
       # np.save('%s/%s' % (args.save, arr), prob)
        output.save('%s/%s' % (args.save, name)) 
    
    
if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '3'  
    main()
    
