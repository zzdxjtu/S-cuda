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
parser.add_argument('--label_dir', type = str, default = '..\\disc_small\\source\\mask')
parser.add_argument('--data-list-source', type=str, default='..\\dataset\\source.txt', help="Path to the file listing the images in the source dataset.")
parser.add_argument('--data-list-target', type=str, default='..\\dataset\\target.txt', help="Path to the file listing the images in the source dataset.")
parser.add_argument('--load_selected_samples', type = str, help = 'dir to load selected samples', default = None)
parser.add_argument('--save_selected_samples', type = str, help = 'dir to save selected samples', default = '..\\clean\\n2\\clean_list\\level_0.2-0.3\\noise_labels_0.1\\clean_selected_0.1.txt')
parser.add_argument('--noise_selected_samples', type = str, help = 'dir to save noise samples', default = '..\\noisy\\n2\noise_list\\level_0.2-0.3\\noise_labels_0.1\\noise_selected_0.1.txt')
parser.add_argument('--restore-from', type=str, default='..\\weights\\Generator\\generator.h5')
parser.add_argument('--weight-root', type=str, default='..\\weights-n2\\level_0.2-0.3\\noise_labels_0.1\\select_0.1')
parser.add_argument('--total_number', type = int, help = 'total number ', default = 200)
parser.add_argument('--epoch', type = int, help = 'total number ', default = 1)
parser.add_argument('--remember_rate', type = float, help = 'remember rate ', default = 0.1)
parser.add_argument('--noise_rate', type = float, help = 'noise rate ', default = 0.1)
args = parser.parse_args()
print(args)

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)

def calculate_dice(gt_dir, prediction, name):
    gt_imgs = os.path.join(gt_dir, name +'.png')
    mask = np.asarray(Image.open(gt_imgs))
    mask_binary = np.zeros((mask.shape[0], mask.shape[1],2))
    prediction_binary = np.zeros((mask.shape[0], mask.shape[1], 2))
    mask_binary[mask < 200] = [1, 0]
    mask_binary[mask < 50] = [1, 1]
    prediction_binary [prediction < 200] = [1, 0]
    prediction_binary[prediction < 50] = [1, 1]
    disc_dice = dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0])
    cup_dice = dice_coef(mask_binary[:,:,1], prediction_binary[:,:,1])
    return disc_dice, cup_dice

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
    batch_size = 2

    dataset_t = args.dataset
    load_selected_samples = args.load_selected_samples
    print(load_selected_samples)
    total_epoch = args.epoch
    power = 0.9
    total_num =5000
    loss_list, name_list = [], []
    if load_selected_samples is None:
        total_num = args.total_number
    else:
        clean_ids = [i_id.strip() for i_id in open(load_selected_samples)]
        total_num = len(clean_ids)
    remember_rate = args.remember_rate
    noise_rate = args.noise_rate
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
    if load_selected_samples is None:
        data_list = args.data_list_source
    else:
        data_list = load_selected_samples
    trainGenerator_Gene = Generator_Gene(data_list, batch_size, '..\\disc_small\\source\\', DiscROI_size,
                                         CDRSeg_size = CDRSeg_size, pt=False, phase='train')

    trainAdversarial_Gene = Adversarial_Gene(batch_size, '..\\disc_small\\source\\', DiscROI_size,
                                             CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)
    trainDS_Gene = GD_Gene(batch_size, '..\\disc_small\\source\\', True,
                           CDRSeg_size=CDRSeg_size, phase='train', noise_label=False)
    trainDT_Gene = GD_Gene(batch_size, '..\\disc_small\\target', False,
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
            print(iters, name)
            loss = results_G[0]
            disc_coef = results_G[1]
            cup_coef = results_G[2]
            smooth_loss = results_G[3]
            dice_loss = results_G[4]
            sum_disc_coef += disc_coef
            sum_cup_coef += cup_coef
            print ('[it %d][src loss %.4f][src dice loss %.4f][disc coef %.4f][cup coef %.4f]' % \
                   (iters + 1, loss, dice_loss, disc_coef, cup_coef))
            noise_disc_coef, noise_cup_coef = calculate_dice(args.label_dir, noise_mask, name)
            print ('===>noise_included: [it %d][noise disc coef %.4f][noise cup coef %.4f]' % \
                   (iters + 1, noise_disc_coef, noise_cup_coef))
            loss_list.append(loss)
            name_list.append(name)
            
            #target domain
            img_T, output_T = next(trainAdversarial_Gene)
            results_A = model_Adversarial.train_on_batch(img_T, output_T)
            loss_A += np.array(results_A) / iters_total


            
            img, label = next(trainDS_Gene)
            prediction = model_Generator.predict(img)
            results_DS = model_Discriminator.train_on_batch(prediction, label)
            loss_DS += results_DS / iters_total

            img, label = next(trainDT_Gene)
            prediction = model_Generator.predict(img)
            results_DT = model_Discriminator.train_on_batch(prediction, label)
            loss_DT += results_DT / iters_total

            if iters+1 == iters_total and load_selected_samples is None:
                ind_sorted = np.argsort(loss_list)
                noise_ind_sorted = np.argsort(-np.array(loss_list))
                loss_sorted = np.array(loss_list)[ind_sorted]
                noise_loss_sorted = np.array(loss_list)[noise_ind_sorted]
                num_remember = int(remember_rate * len(loss_sorted))
                noise_num_remember = int(noise_rate * len(noise_loss_sorted))
                clean_ind_update = ind_sorted[:num_remember]
                noise_ind_update = noise_ind_sorted[:noise_num_remember]
                clean_name_update = np.array(name_list)[clean_ind_update]
                noise_name_update = np.array(name_list)[noise_ind_update]

                with open(os.path.join(args.save_selected_samples), "w") as f:
                    for i in range(len(clean_name_update)):
                        print('clean:',clean_name_update[i])
                        f.write(str(clean_name_update[i]) + '\n')
                        if int(str(clean_name_update[i])[-3:]) < 10:
                            f.write(str(clean_name_update[i])[0:4] + str(int(str(clean_name_update[i])[-3:])-1) + '\n')
                        elif int(str(clean_name_update[i])[-3:])==10:
                            f.write(str(clean_name_update[i])[0:3] + '0' + str(int(str(clean_name_update[i])[-3:])-1) + '\n')
                        elif int(str(clean_name_update[i])[-3:]) < 100:
                            f.write(str(clean_name_update[i])[0:3] + str(int(str(clean_name_update[i])[-3:])-1) + '\n')
                        elif int(str(clean_name_update[i])[-3:])==100:
                            f.write(str(clean_name_update[i])[0:2] + '0' + str(int(str(clean_name_update[i])[-3:])-1) + '\n')
                        else:
                            f.write(str(clean_name_update[i])[0:2] + str(int(str(clean_name_update[i])[-3:])-1) + '\n')
                
                with open(os.path.join(args.noise_selected_samples), "w") as g:
                    for j in range(len(noise_name_update)):
                        print('noise:',noise_name_update[j])
                        g.write(str(noise_name_update[j]) + '\n')
                        if int(str(noise_name_update[j])[-3:]) < 10:
                            g.write(str(noise_name_update[j])[0:4] + str(int(str(noise_name_update[j])[-3:])-1) + '\n')
                        elif int(str(noise_name_update[j])[-3:])==10:
                            g.write(str(noise_name_update[j])[0:3] + '0' + str(int(str(noise_name_update[j])[-3:])-1) + '\n')
                        elif int(str(noise_name_update[j])[-3:]) < 100:
                            g.write(str(noise_name_update[j])[0:3] + str(int(str(noise_name_update[j])[-3:])-1) + '\n')
                        elif int(str(noise_name_update[j])[-3:])==100:
                            g.write(str(noise_name_update[j])[0:2] + '0' + str(int(str(noise_name_update[j])[-3:])-1) + '\n')
                        else:
                            g.write(str(noise_name_update[j])[0:2] + str(int(str(noise_name_update[j])[-3:])-1) + '\n')
                print(args.save_selected_samples,'Sample selection finished!')
                break
            
            if (iters+1) == iters_total:
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

