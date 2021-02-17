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
           # import pdb;pdb.set_trace()
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





def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    #import pdb;pdb.set_trace()
    n, c, h, w = prob.size()
    return (-torch.sum(torch.mul(prob, torch.log2(prob + 1e-30)), 1) / np.log2(c)).view(1, 1, h, w)



