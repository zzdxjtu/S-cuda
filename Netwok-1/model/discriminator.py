import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np

class WeightedBCEWithLogitsLoss(nn.Module):

    def __init__(self, size_average=True):
        super(WeightedBCEWithLogitsLoss, self).__init__()


    def weighted(self, input, target, weight, alpha, beta):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)  
        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

        if weight is not None:
            #import pdb;pdb.set_trace()
            #weight = weight.view(1, 1, weight.size(1), weight.size(2))
            #weight = weight.type(torch.cuda.FloatTensor) 
            #weight = Variable(torch.FloatTensor(weight.data.size()).fill_(1)).cuda() - weight
            #weight = Variable(weight).cuda()
            #arr = Variable(arr).cuda()
            #weight = arr - weight
            #loss = loss.type(torch.cuda.FloatTensor)
            #loss = loss.float()
            #weight.detach()
            loss = alpha * loss + beta * weight * loss

        return loss.mean()

    def forward(self, input, target, weight, alpha, beta):
        if weight is not None:
            return self.weighted(input, target, weight, alpha, beta)
        else:
            return self.weighted(input, target, None, alpha, beta)

class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf = 64):
        super(FCDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.bce_loss = nn.BCEWithLogitsLoss()


    def forward(self, x, lbl):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        #import pdb;pdb.set_trace()
        self.loss = self.bce_loss(x, Variable(torch.FloatTensor(x.data.size()).fill_(lbl)).cuda())
        #print(Variable(torch.FloatTensor(x.data.size()).fill_(lbl)).cuda())
        #print(self.loss)
        return x

    def adjust_learning_rate(self, args, optimizer, i):
        if args.model == 'DeepLab':
            lr = args.learning_rate_D * ((1 - float(i) / args.num_steps) ** (args.power))
            optimizer.param_groups[0]['lr'] = lr
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = lr * 10 
        else:
            optimizer.param_groups[0]['lr'] = args.learning_rate_D * (0.1**(int(i/50000)))
            if len(optimizer.param_groups) > 1:
                optimizer.param_groups[1]['lr'] = args.learning_rate_D * (0.1**(int(i/50000))) * 2  			
