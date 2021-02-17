import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from options.train_only_options import TrainOptions
import os
import numpy as np
from data import CreateSrcDataLoader
from data import CreateTrgDataLoader
from model import CreateModel
from model import CreateDiscriminator
from utils.timer import Timer
import tensorboardX
from model.loss import prob_2_entropy, WeightedBCEWithLogitsLoss, WeightMapLoss


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
        
    train_writer = tensorboardX.SummaryWriter(os.path.join(args.snapshot_dir, "logs", model_name))
    
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
    weight_map_loss = WeightMapLoss()
    loss = ['loss_seg_src', 'loss_seg_trg', 'loss_D_trg_fake', 'loss_D_src_real', 'loss_D_trg_real']
    _t['iter time'].tic()
    for i in range(start_iter, args.num_steps):
        print(i)
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

        try:
            src_img, src_lbl, weight_map, _ = sourceloader_iter.next()
        except StopIteration:
            sourceloader_iter = iter(sourceloader)
            src_img, src_lbl, weight_map, _ = sourceloader_iter.next()
        src_img, src_lbl, weight_map = Variable(src_img).cuda(), Variable(src_lbl.long()).cuda(), Variable(weight_map.long()).cuda()
        src_seg_score1, src_seg_score2, src_seg_score3, src_seg_score4 = model(src_img, lbl=src_lbl, weight=weight_map)
        #import pdb;pdb.set_trace()
        #WeightLoss1 = weight_map_loss(src_seg_score1, src_lbl, weight_map)
        #WeightLoss2 = weight_map_loss(src_seg_score2, src_lbl, weight_map)
        loss_seg_src = model.loss
        #print('WeightLoss2, WeightLoss1:', WeightLoss2.data, WeightLoss1.data)
        loss_seg_src.backward()
        
        if args.data_label_folder_target is not None:
            trg_img, trg_lbl, _, name = targetloader_iter.next()
            trg_img, trg_lbl = Variable(trg_img).cuda(), Variable(trg_lbl.long()).cuda()
            trg_seg_score1, trg_seg_score2, trg_seg_score3, trg_seg_score4 = model(trg_img, lbl=trg_lbl) 
            loss_seg_trg = model.loss
        else:
            trg_img, _, name = targetloader_iter.next()
            trg_img = Variable(trg_img).cuda()
            trg_seg_score1, trg_seg_score2, trg_seg_score3, trg_seg_score4 = model(trg_img)
            loss_seg_trg = 0
        outD1_trg = model_D1(F.softmax(trg_seg_score1), 0)
        outD2_trg = model_D2(F.softmax(trg_seg_score2), 0)
        #import pdb;pdb.set_trace()
        outD1_trg = interp_target(outD1_trg) #[1, 1, 1024, 1024]
        outD2_trg = interp_target(outD2_trg)

        '''
        if i > 9001:
            #import pdb;pdb.set_trace()
            weight_map1 = prob_2_entropy(F.softmax(trg_seg_score1)) #[1, 1, 1024, 1024]
            weight_map2 = prob_2_entropy(F.softmax(trg_seg_score2)) #[1, 1, 1024, 1024]
            loss_D1_trg_fake = weight_loss(outD1_trg, Variable(torch.FloatTensor(outD1_trg.data.size()).fill_(0)).cuda(), weight_map1, 0.3, 1)
            loss_D2_trg_fake = weight_loss(outD2_trg, Variable(torch.FloatTensor(outD2_trg.data.size()).fill_(0)).cuda(), weight_map2, 0.3, 1)
        else:
            loss_D1_trg_fake = model_D1.loss
            loss_D2_trg_fake = model_D2.loss
        loss_D_trg_fake = loss_D1_trg_fake*0.2 + loss_D2_trg_fake
        '''

        loss_D_trg_fake = model_D1.loss*0.2 + model_D2.loss
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
        
        
        for m in loss:
            train_writer.add_scalar(m, eval(m), i+1)
            
        if (i+1) % args.save_pred_every == 0:
            print ('taking snapshot ...')
            torch.save(model.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source)+str(i+1)+'.pth' ))   
            torch.save(model_D1.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source)+str(i+1)+'_D1.pth' ))
            torch.save(model_D2.state_dict(), os.path.join(args.snapshot_dir, '%s_' %(args.source)+str(i+1)+'_D2.pth' )) 
        if (i+1) % args.print_freq == 0:
            _t['iter time'].toc(average=False)
            print ('[it %d][src seg loss %.4f][trg seg loss %.4f][lr %.4f][%.2fs]' % \
                    (i + 1, loss_seg_src.data,loss_seg_trg.data, optimizer.param_groups[0]['lr']*10000, _t['iter time'].diff))
            if i + 1 > args.num_steps_stop:
                print ('finish training')
                break
            _t['iter time'].tic()
            
if __name__ == '__main__':
    #os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    #memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]
    #os.system('rm tmp')    
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    main()
    
    
        
