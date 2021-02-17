import torch.optim as optim
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options.train_weight_options import TrainOptions
import os
import numpy as np
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from data import CreateTrgDataSSLLoader
from model import CreateModel
from model import CreateDiscriminator
from utils.timer import Timer
from model.loss import prob_2_entropy, WeightedBCEWithLogitsLoss
import tensorboardX
from PIL import Image

def dice_coef(y_true, y_pred):
    smooth = 1e-8
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (np.sum(y_true*y_true) + np.sum(y_pred*y_pred) + smooth)

def calculate_dice(gt_dir, prediction, name):
    #import pdb;pdb.set_trace()
    gt_imgs = os.path.join(gt_dir, 'labels', name[0] + '.bmp')
    mask = np.asarray(Image.open(gt_imgs))
    mask_binary = np.zeros((mask.shape[0], mask.shape[1],2))
    mask_binary[mask < 200] = [1, 0]
    mask_binary[mask < 50] = [1, 1]
    prediction_binary = np.zeros((mask.shape[0], mask.shape[1], 2))
    prediction_binary [prediction == 1] = [1, 0]
    prediction_binary[prediction == 0] = [1, 1]
    disc_dice = dice_coef(mask_binary[:,:,0], prediction_binary[:,:,0])
    cup_dice = dice_coef(mask_binary[:,:,1], prediction_binary[:,:,1]) 
    return disc_dice, cup_dice


def main():
    
    opt = TrainOptions()
    args = opt.initialize()
    
    _t = {'iter time' : Timer()}
    
    model_name = args.source + '_to_' + args.target
    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)   
        os.makedirs(os.path.join(args.snapshot_dir, 'logs'))
    opt.print_options(args)
    
    sourceloader, targetloader = CreateSrcDataLoader(args), CreateTrgDataLoader(args)
    targetloader_iter, sourceloader_iter = iter(targetloader), iter(sourceloader)
    
    model, optimizer = CreateModel(args)
    model_D1, optimizer_D1 = CreateDiscriminator(args, 1)
    model_D2, optimizer_D2 = CreateDiscriminator(args, 2)
    start_iter = 0
    if args.restore_from is not None:
        start_iter = int(args.restore_from.rsplit('/', 1)[1].rsplit('_')[1])
        
    #train_writer = tensorboardX.SummaryWriter(os.path.join(args.snapshot_dir, "logs", model_name))
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    interp_target = nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=True)
    interp_source = nn.Upsample(size=(1024, 1024), mode='bilinear', align_corners=True)
    cudnn.enabled = True
    cudnn.benchmark = True
    model.train()
    model.cuda()
    model_D1.train()
    model_D1.cuda()
    model_D2.train()
    model_D2.cuda()
    weight_loss = WeightedBCEWithLogitsLoss()
    loss = ['loss_seg_src', 'loss_seg_trg', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real']
    _t['iter time'].tic()

    load_selected_samples = args.load_selected_samples
    if load_selected_samples is None:
        total_num = args.total_number
    else:
        clean_ids = [i_id.strip() for i_id in open(load_selected_samples)]
        total_num = len(clean_ids)
    remember_rate = args.remember_rate
    loss_list, name_list, dice_list = [], [], []
    print('total_num:',total_num)
    predict_sum_disc = 0
    predict_sum_cup = 0
    noise_sum_disc = 0
    noise_sum_cup = 0
    threshold = 0.4
    for i in range(start_iter, start_iter + total_num):
       
        model.adjust_learning_rate(args, optimizer, i)
        model_D1.adjust_learning_rate(args, optimizer_D1, i)
        model_D2.adjust_learning_rate(args, optimizer_D2, i)
        optimizer.zero_grad()
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()

        ##train G
        for param in model_D1.parameters():
            param.requires_grad = False 
        for param in model_D2.parameters():
            param.requires_grad = False


        #import pdb;pdb.set_trace()    
        try:
            src_img, src_lbl, weight_map, name = sourceloader_iter.next()
        except StopIteration:
            sourceloader_iter = iter(sourceloader)
            src_img, src_lbl, weight_map, name = sourceloader_iter.next()
        src_img, src_lbl, weight_map = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda(), Variable(weight_map.long()).cuda()
        src_seg_score1, src_seg_score2, src_seg_score3, src_seg_score4 = model(src_img, lbl=src_lbl, weight=None)
        loss_seg_src = model.loss   
        loss_seg_src.backward()
        loss_list.append(loss_seg_src)
        name_list.append(name)
        print(i,name)
        #import pdb;pdb.set_trace()
        output = nn.functional.softmax(src_seg_score2, dim=1)
        output = nn.functional.upsample(output, (2056, 2124), mode='bilinear', align_corners=True).cpu().data[0].numpy()
        output = output.transpose(1, 2, 0)  # (1634,1634,3)
        output_mask = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)  # (1644,1634) unique:[0,1,2]
        predict_disc_dice, predict_cup_dice = calculate_dice(args.data_dir, output_mask, name)
        print('===>predict disc_dice:' + str(round(predict_disc_dice, 3)) + '\t' + 'cup_dice:' + str(round(predict_cup_dice, 3)))
        #import pdb;pdb.set_trace()
        label = torch.unsqueeze(src_lbl,0)
        label = nn.functional.upsample(label.float(),size= (2056, 2124),mode= 'bilinear', align_corners=True).cpu().data[0].numpy().squeeze()
        noise_disc_dice, noise_cup_dice = calculate_dice(args.data_dir, label, name)
        print('===>noise-included disc_dice:' + str(round(noise_disc_dice, 3)) + '\t' + 'cup_dice:' + str(round(noise_cup_dice, 3)))
        predict_sum_disc += predict_disc_dice
        predict_sum_cup += predict_cup_dice
        noise_sum_disc += noise_disc_dice
        noise_sum_cup += noise_cup_dice
        
        #import pdb;pdb.set_trace()
        if load_selected_samples is not None:
            if (2- predict_disc_dice - predict_cup_dice)>threshold:
                src_seg_score1, src_seg_score2, src_seg_score3, src_seg_score4 = model(src_img, lbl=src_lbl, weight=weight_map)
                weightloss = 0.05*model.loss
                print('weightloss:', weightloss)
                weightloss.backward()


        if args.data_label_folder_target is not None:
            trg_img, trg_lbl, _, _ = targetloader_iter.next()
            trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda()
            trg_seg_score1, trg_seg_score2, trg_seg_score3, trg_seg_score4 = model(trg_img, lbl=trg_lbl) 
            loss_seg_trg = model.loss
        else:
            trg_img, _, _ = targetloader_iter.next()
            trg_img = Variable(trg_img).cuda()
            trg_seg_score1, trg_seg_score2, trg_seg_score3, trg_seg_score4 = model(trg_img)
            loss_seg_trg = 0
        outD1_trg = model_D1(F.softmax(trg_seg_score1), 0)
        outD2_trg = model_D2(F.softmax(trg_seg_score2), 0)
        #import pdb;pdb.set_trace()
        outD1_trg = interp_target(outD1_trg)
        outD2_trg = interp_target(outD2_trg)
       
        if i > 9001:
            #import pdb;pdb.set_trace()
            weight_map1 = prob_2_entropy(F.softmax(trg_seg_score1))
            weight_map2 = prob_2_entropy(F.softmax(trg_seg_score2))
            loss_D1_trg_fake = weight_loss(outD1_trg, Variable(torch.FloatTensor(outD1_trg.data.size()).fill_(0)).cuda(), weight_map1, 0.3, 1)
            loss_D2_trg_fake = weight_loss(outD2_trg, Variable(torch.FloatTensor(outD2_trg.data.size()).fill_(0)).cuda(), weight_map2, 0.3, 1)
        else:
            loss_D1_trg_fake = model_D1.loss
            loss_D2_trg_fake = model_D2.loss
        #loss_D_trg_fake = model_D1.loss*0.2 + model_D2.loss
        loss_D_trg_fake = loss_D1_trg_fake*0.2 + loss_D2_trg_fake
        loss_trg = args.lambda_adv_target * loss_D_trg_fake + loss_seg_trg
        loss_trg.backward()
        
        ###train D
        for param in model_D1.parameters():
            param.requires_grad = True
        for param in model_D2.parameters():
            param.requires_grad = True

        src_seg_score1, src_seg_score2, src_seg_score3, src_seg_score4, trg_seg_score1, trg_seg_score2, trg_seg_score3, trg_seg_score4 = src_seg_score1.detach(), src_seg_score2.detach(), src_seg_score3.detach(), src_seg_score4.detach(), trg_seg_score1.detach(), trg_seg_score2.detach(), trg_seg_score3.detach(), trg_seg_score4.detach()
        

        outD1_src = model_D1(F.softmax(src_seg_score1), 0)
        outD2_src = model_D2(F.softmax(src_seg_score2), 0)
        
        loss_D1_src_real = model_D1.loss / 2
        loss_D1_src_real.backward()
        loss_D2_src_real = model_D2.loss / 2
        loss_D2_src_real.backward()
        loss_D_src_real = loss_D1_src_real+loss_D2_src_real
        
        
        outD1_trg = model_D1(F.softmax(trg_seg_score1), 1)
        outD2_trg = model_D2(F.softmax(trg_seg_score2), 1)

        outD1_trg = interp_target(outD1_trg)
        outD2_trg = interp_target(outD2_trg)
        if i > 9001:
            weight_map1 = prob_2_entropy(F.softmax(trg_seg_score1))
            weight_map2 = prob_2_entropy(F.softmax(trg_seg_score2))
            loss_D1_trg_real = weight_loss(outD1_trg, Variable(torch.FloatTensor(outD1_trg.data.size()).fill_(1)).cuda(), weight_map1, 0.3, 1)/2
            loss_D2_trg_real = weight_loss(outD2_trg, Variable(torch.FloatTensor(outD2_trg.data.size()).fill_(1)).cuda(), weight_map2, 0.3, 1)/2    

        else:
            loss_D1_trg_real = model_D1.loss/2
            loss_D2_trg_real = model_D2.loss/2


        loss_D1_trg_real.backward()       
        loss_D2_trg_real.backward()
        loss_D_trg_real = loss_D1_trg_real+loss_D2_trg_real  
        
        optimizer.step()
        optimizer_D1.step()
        optimizer_D2.step()
        
        
       # for m in loss:
       #     train_writer.add_scalar(m, eval(m), i+1)
            
        if (i+1 == start_iter + total_num) and load_selected_samples is not None:
            print ('taking snapshot ', args.snapshot_dir, args.source+'_'+str(total_num))
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source)+str(total_num)+'.pth' ))   
            torch.save(model_D1.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source)+str(total_num)+'_D1.pth' ))
            torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source)+str(total_num)+'_D2.pth' )) 
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print ('[it %d][src seg loss %.4f][trg loss %.4f][trg seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data,loss_trg.data, loss_seg_trg.data,optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))
            if i + 1 > args.num_steps_stop:
                print ('finish training')
                break
            _t['iter time'].tic()
         
        if i + 1 == start_iter + total_num and load_selected_samples is None:
            ind_sorted = np.argsort(loss_list)
            loss_sorted = np.array(loss_list)[ind_sorted]
            num_remember = int(remember_rate * len(loss_sorted))
            clean_ind_update = ind_sorted[:num_remember]
            clean_name_update = np.array(name_list)[clean_ind_update]
            #import pdb;pdb.set_trace()
            #dice = np.array(dice_list)[clean_ind_update]
            noise_ind_sorted = np.argsort(-np.array(loss_list))
            noise_loss_sorted = np.array(loss_list)[noise_ind_sorted]
            noise_num_remember = int(remember_rate * len(noise_loss_sorted))
            noise_ind_update = noise_ind_sorted[:noise_num_remember]
            noise_name_update = np.array(name_list)[noise_ind_update]
  

            with open(os.path.join(args.save_selected_samples), "w") as f:
                for i in range(len(clean_name_update)):
                    #f.write(str(clean_name_update[i][0]) +'\t'+ str(dice[:,0][i]) + '\t'+ str(dice[:,1][i]) +'\n')
                    f.write(str(clean_name_update[i][0])+'\n') 
            with open(os.path.join(args.noise_selected_samples), "w") as g:
                for j in range(len(noise_name_update)):
                    g.write(str(noise_name_update[j][0])+'\n')  
            print(args.save_selected_samples, 'Sample selection finished!')
            break
           
    print('\n predict disc_coef = {0:.4f}, cup_coef = {1:.4f}'.format(predict_sum_disc/total_num, predict_sum_cup/total_num))
    print('\n noise-included disc_coef = {0:.4f}, cup_coef = {1:.4f}'.format(noise_sum_disc / total_num, noise_sum_cup / total_num))

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    main()
    
    
        
