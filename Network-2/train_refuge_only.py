'''
Created by SJWANG  07/21/2018
For refuge image segmentation
'''

from Model.models import *
from Utils.data_generator import *
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import *

import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.utils import plot_model
import matplotlib.pyplot as plt
import os
import numpy as np
import tensorflow as tf
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, help = 'refuge-new, Drishti-GS', default = 'refuge-new')
parser.add_argument('--data-list-source', type=str, default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/source.txt', help="Path to the file listing the images in the source dataset.")
parser.add_argument('--data-list-target', type=str, default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/path/to/dataset-new/target.txt', help="Path to the file listing the images in the source dataset.")
#parser.add_argument('--load_selected_samples', type = str, help = 'dir to load selected samples', default = '../path/to/dataset-new/source/eyesgan/level_0.5-0.7/noise_labels_0.1/selected_sample1.txt')
parser.add_argument('--load_selected_samples', type = str, help = 'dir to load selected samples', default = None)
parser.add_argument('--save_selected_samples', type = str, help = 'dir to save selected samples', default = None)
#parser.add_argument('--save_selected_samples', type = str, help = 'dir to save selected samples', default = './data/refuge-new/selected_sample.txt')
parser.add_argument('--restore-from', type=str, default='/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/weights/weights1.h5')
parser.add_argument('--weight-root', type=str, default='../weights/')
parser.add_argument('--total_number', type = int, help = 'total number ', default = 400)
parser.add_argument('--epoch', type = int, help = 'total number ', default = 1)
args = parser.parse_args()
print(args)


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def change_learning_rate(model, base_lr, iter, max_iter, power):
    new_lr = lr_poly(base_lr, iter, max_iter, power)
    K.set_value(model.optimizer.lr, new_lr)
    return K.get_value(model.optimizer.lr)


def change_learning_rate_D(model, base_lr, iter, max_iter, power):
    new_lr = lr_poly(base_lr, iter, max_iter, power)
    K.set_value(model.optimizer.lr, new_lr)
    return K.get_value(model.optimizer.lr)


if __name__ == '__main__':

    ''' parameter setting '''
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    DiscROI_size = 512
    CDRSeg_size = 512
    lr = 2.5e-5
    LEARNING_RATE_D = 1e-5
    batch_size = 8

    dataset_t = args.dataset
    load_selected_samples = args.load_selected_samples
    print(load_selected_samples)
    total_epoch = args.epoch
    power = 0.9
    total_num =5000
    loss_list, name_list = [], []
    load_from = args.restore_from
    weights_root = args.weight_root
    G_weights_root = os.path.join(weights_root, 'Generator')
    D_weights_root = os.path.join(weights_root, 'Discriminator')


    if not os.path.exists(G_weights_root):
        print("Create save weights folder on %s\n\n" % weights_root)
        os.makedirs(G_weights_root)
        os.makedirs(D_weights_root)
    _MODEL = os.path.basename(__file__).split('.')[0]

    logs_path = "./log_tf/" + dataset_t + "/DA_patch2/"
    logswriter = tf.summary.FileWriter
    print("logtf path: %s \n\n" % logs_path)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    summary_writer = logswriter(logs_path)

    ''' define model '''
    model_Generator = Model_CupSeg(input_shape = (CDRSeg_size, CDRSeg_size, 3), classes=2, backbone='mobilenetv2', lr=lr)

    model_Discriminator = Discriminator(input_shape=(CDRSeg_size, CDRSeg_size, 2),
                                                  learning_rate=LEARNING_RATE_D)
    model_Adversarial = Sequential()
    model_Discriminator.trainable = False
    model_Adversarial.add(model_Generator)
    model_Adversarial.add(model_Discriminator)
    model_Adversarial.compile(optimizer=SGD(lr=lr), loss='binary_crossentropy')

    if os.path.exists(load_from):
        print('Loading weight for generator model from file {}\n\n'.format(load_from))
        model_Generator.load_weights(load_from)
    else:
        print('[ERROR:] CANNOT find weight file {}\n\n'.format(load_from))

    ''' define data generator '''
    # train0 means 4/5 training data from REFUGE dataset
    trainGenerator_Gene = Generator_Gene(args.data_list_source, batch_size, '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/data/'+dataset_t+'/source/disc_small', DiscROI_size, CDRSeg_size = CDRSeg_size, pt=False, phase='train')
    valGenerator_Gene = Generator_Gene(args.data_list_target, batch_size, '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/data/' + dataset_t + '/target/disc_small', DiscROI_size, CDRSeg_size=CDRSeg_size, pt=False, phase='val')
    trainAdversarial_Gene = Adversarial_Gene(batch_size, '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/data/' + dataset_t + '/source/disc_small', DiscROI_size,
                                             CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)
    trainDS_Gene = GD_Gene(batch_size, '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/data/' + dataset_t + '/source/disc_small', True,
                           CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)
    trainDT_Gene = GD_Gene(batch_size, '/extracephonline/medai_data2/lolitazhang/my-master/eyes-master/data/' + dataset_t + '/target/disc_small', False,
                           CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)

    ''' train for epoch and iter one by one '''
    epoch = 0
    dice_loss_val = 0
    disc_coef_val = 0
    cup_coef_val = 0
    results_eva = [0, 0, 0]
    results_DS = 0
    results_DT = 0
    for epoch in range(total_epoch):
        loss = 0
        smooth_loss = 0
        dice_loss = 0
        disc_coef = 0
        cup_coef = 0
        sum_disc_coef = 0
        sum_cup_coef = 0

        loss_A = 0
        loss_GD = 0
        loss_DS = 0
        loss_DT = 0
        loss_A_map = 0
        loss_A_scale = 0
        loss_DS_map = 0
        loss_DS_scale = 0
        loss_DT_map = 0
        loss_DT_scale = 0
        results_A = 0

        iters_total = int(total_num)
        print('iters_total:', iters_total)
        for iters in range(iters_total):

            ''' train Generator '''
            #source domain
            img_S, mask_S, noise_mask, name = next(trainGenerator_Gene)
            results_G = model_Generator.train_on_batch(img_S, mask_S)
            #import pdb;pdb.set_trace()
            print(iters, name)
            loss = results_G[0]
            disc_coef = results_G[1]
            cup_coef = results_G[2]
            smooth_loss = results_G[3]
            dice_loss = results_G[4]
            sum_disc_coef += disc_coef
            sum_cup_coef += cup_coef
            print ('[it %d][src loss %.4f][src dice loss %.4f]' % \
                   (iters + 1, loss, dice_loss))
            #target domain
            img_T, output_T = next(trainAdversarial_Gene)
            results_A = model_Adversarial.train_on_batch(img_T, output_T)
            loss_A += np.array(results_A) / iters_total
            #print(loss_A)

            # print log information every 10 iterations
            if (iters + 1) % 10 == 0:
                img, mask, _, _ = next(valGenerator_Gene)
                results_eva = model_Generator.evaluate(img, mask)
                dice_loss_val += results_eva[0] / (iters_total/20)
                disc_coef_val += results_eva[1] / (iters_total/20)
                cup_coef_val += results_eva[2] / (iters_total/20)
                print('[EVALUATION: (iter: {})]\n{}:{},{}:{},{}:{}' \
                      .format(iters+1, model_Generator.metrics_names[0],results_eva[0],
                                                 model_Generator.metrics_names[1],results_eva[1],
                                                 model_Generator.metrics_names[2], results_eva[2]))
          
            
            img, label = next(trainDS_Gene)
            prediction = model_Generator.predict(img)
            results_DS = model_Discriminator.train_on_batch(prediction, label)
            loss_DS += results_DS / iters_total

            img, label = next(trainDT_Gene)
            prediction = model_Generator.predict(img)
            results_DT = model_Discriminator.train_on_batch(prediction, label)
            loss_DT += results_DT / iters_total

     
            if (iters+1) % 40 == 0:
                G_weights_path = os.path.join(G_weights_root, 'generator_%s.h5' % ( iters + 1 ))
                D_weights_path = os.path.join(D_weights_root, 'discriminator_%s.h5' % ( iters + 1 ))
                print("Save model to %s" % G_weights_path)
                model_Generator.save_weights(G_weights_path, overwrite=True)
                print("Save model to %s" % D_weights_path)
                model_Discriminator.save_weights(D_weights_path, overwrite=True)

        # update learning rate
        change_learning_rate(model_Generator, lr, epoch, total_epoch, power)
        change_learning_rate(model_Adversarial, lr, epoch, total_epoch, power)
        change_learning_rate_D(model_Discriminator, LEARNING_RATE_D, epoch, total_epoch, power)

