from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample
from model.attention import PAM_Module, CAM_Module
import torch.optim as optim
from model.loss import WeightMapLoss, calculate_dice

###new  start
def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    #import pdb;pdb.set_trace()
    #print('one-hot',input.size())
    input = input.cpu()
    input = np.array(input)
    shape = np.array(input.shape)
    #shape = np.expand_dims(shape, axis=1)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    input = Variable(torch.from_numpy(input)).type(torch.LongTensor)
    result = result.scatter_(1, input, 1)
    result = Variable(result[:, :3, :, :]).cuda()
    return result


class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        #import p#db;pdb.set_trace()
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target))*2 + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p)) + self.smooth

        dice = num / den
        loss = 1 - dice
        return loss


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = [1,1,0.1]
        self.ignore_index = ignore_index

    def forward(self, predict, target,fake_label=None):
        #import pdb;pdb.set_trace()
        target = target.view((target.shape[0], 1, *target.shape[1:]))
        target = make_one_hot(target, 256)
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = F.softmax(predict, dim=1)
        #import pdb;pdb.set_trace()
        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    #assert self.weight.shape[0] == target.shape[1], \
                    #    'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
        loss = total_loss/target.shape[1]

        if fake_label is not None:
            for i in range(target_fake.shape[1]):
                dice_loss = dice(predict[:, i], target_fake[:, i])
                if self.weight is not None:
                    dice_loss *= self.weight[i]
                total_loss += dice_loss
            loss = 0.1*total_loss/target_fake.shape[1] + loss

        return loss


def sum_tensor(inp, axes, keepdim=False):
    #import pdb;pdb.set_trace()
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes:
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    #import pdb;pdb.set_trace()
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = sum_tensor(tp, axes, keepdim=False)
    fp = sum_tensor(fp, axes, keepdim=False)
    fn = sum_tensor(fn, axes, keepdim=False)

    return tp, fp, fn


def softmax_helper(x):
    #import pdb;pdb.set_trace()
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=softmax_helper, batch_dice=False, do_bg=False, smooth=1.,
                 square=False):
        """

        """
        super(SoftDiceLoss, self).__init__()

        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        #import pdb;pdb.set_trace()
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn = get_tp_fp_fn(x, y, axes, loss_mask, self.square)

        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        '''
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        '''
        #import pdb;pdb.set_trace()
        dc = dc[:, :2]
        dc = dc.mean()

        return 1 - dc

class DANet(nn.Module):
    def __init__(self, num_classes, channel):
    #def __init__(self, num_classes=19, channel, norm_layer=nn.BatchNorm2d):
        super(DANet, self).__init__()
        self.head = DANetHead(channel, num_classes, nn.BatchNorm2d)
        #self.conv = nn.Conv2d(2048, 19, kernel_size=3, stride=1, padding=1, bias=False)      
    def forward(self, x):
        #import pdb;pdb.set_trace()
        #imsize = self.imsize
        
        #initial = self.conv(x)
        x = self.head(x)
        x = list(x)
        #x[0] = upsample(x[0], imsize, mode='bilinear', align_corners=True)
        #x[1] = upsample(x[1], imsize, mode='bilinear', align_corners=True)
        #x[2] = upsample(x[2], imsize, mode='bilinear', align_corners=True)

        #outputs = [x[0]]
        #outputs.append(x[1])
        #outputs.append(x[2])
        return x[0], x[1]

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        #import pdb;pdb.set_trace()
        feat1 = self.conv5a(x)##(channel/4,w,h)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)##(19,w,h)

        feat2 = self.conv5c(x)##(channel/4,w,h)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)##(19,w,h)

        feat_sum = sa_conv+sc_conv##(channel/4,w,h)

        sasc_output = self.conv8(feat_sum)##(19,w,h)
        
        output = [feat_sum]
        output.append(sasc_output)
        output.append(sa_output)
        output.append(sc_output)
        return tuple(output)
###new  end



affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, int(inplanes/4), kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            #import pdb;pdb.set_trace()
            out = torch.cat([out, self.conv2d_list[i + 1](x)], 1)
        return out


class ResNet101(nn.Module):
    def __init__(self, block, layers, num_classes, phase):
        self.inplanes = 64
        self.phase = phase
        super(ResNet101, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 1024, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer_att1 = self._get_danet(num_classes, 1024)
        self.layer_att2 = self._get_danet(num_classes, 2048)
        self.layer_conv1 = nn.Conv2d(1280, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        self.layer_conv2 = nn.Conv2d(2560, num_classes, kernel_size=3, stride=1, padding=1, bias=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def _get_danet(self, num_classes, channel):  ###new
        model = DANet(num_classes=num_classes, channel=channel)
        return model


    def forward(self, x, ssl=False, lbl=None, lbl_new=None, weight=None, alpha=0.5):
        _, _, h, w = x.size()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3(x)
        x2 = self.layer4(x1)

        xd1 = self.layer5(x1)
        xa1, xaa1 = self.layer_att1(x1)#(channel/4,w,h);(19,w,h)
        x1 = torch.cat([xd1, xa1], 1)#(5*channel/4,w,h)
        x1 = self.layer_conv1(x1)#(19,w,h)

        xd2 = self.layer6(x2)
        xa2, xaa2 = self.layer_att2(x2)#(channel/4,w,h);(19,w,h)
        x2 = torch.cat([xd2, xa2], 1)#(5*channel/4,w,h)
        x2 = self.layer_conv2(x2)#(19,w,h)

        if self.phase == 'train' and not ssl:
            x1 = nn.functional.upsample(x1, (h, w), mode='bilinear', align_corners=True)
            x2 = nn.functional.upsample(x2, (h, w), mode='bilinear', align_corners=True)
            xaa1 = nn.functional.upsample(xaa1, (h, w), mode='bilinear', align_corners=True)
            xaa2 = nn.functional.upsample(xaa2, (h, w), mode='bilinear', align_corners=True)

            if lbl is not None: 
                self.loss1 = self.CrossEntropy2d(x1, lbl) + DiceLoss()(x1, lbl)*20 + self.Smoothloss(x1, lbl)*30
                self.loss2 = self.CrossEntropy2d(x2, lbl) + DiceLoss()(x2, lbl)*20 + self.Smoothloss(x2, lbl)*30
                self.loss = self.loss1*0.1 + self.loss2
            if lbl_new is not None:
                self.loss1_new = self.CrossEntropy2d(x1, lbl_new) + DiceLoss()(x1, lbl_new)*20 + self.Smoothloss(x1, lbl_new)*30
                self.loss2_new = self.CrossEntropy2d(x2, lbl_new) + DiceLoss()(x2, lbl_new)*20 + self.Smoothloss(x2, lbl_new)*30
                self.loss_new = self.loss1_new*0.1 + self.loss2_new
            self.loss_sum = (1-alpha) * self.loss + alpha * self.loss_new

            if weight is not None:
                weight_map_loss = WeightMapLoss()
                WeightLoss1 = weight_map_loss(x1, lbl, weight)*5  # [1,3,1024,1024] [1,1024,1024] [1,1024,1024,3]
                WeightLoss2 = weight_map_loss(x2, lbl, weight)*5

                weight_bck = weight[:,0,:,:]
                self.loss1 = DiceLoss()(x1, weight_bck)*5 + self.Smoothloss(x1, lbl)*20 #(1,1024,1024)
                self.loss2 = DiceLoss()(x2, weight_bck)*5 + self.Smoothloss(x2, lbl)*20
                self.loss = self.loss1*0.1 + self.loss2 + WeightLoss1*0.1 + WeightLoss2
            
        return x1, x2, xaa1, xaa2


    def get_1x_lr_params_NOscale(self):

        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)
     
        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):

        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        b.append(self.layer_att1.parameters())
        b.append(self.layer_att2.parameters())
        b.append(self.layer_conv1.parameters())
        b.append(self.layer_conv2.parameters())
        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]
        #return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate}]


    def adjust_learning_rate(self, args, optimizer, i):
        lr = args.learning_rate * ((1 - float(i) / args.num_steps) ** (args.power))
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10  
            
    def CrossEntropy2d(self, predict, target, fake_label=None, weight=None, size_average=True):
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        #import pdb;pdb.set_trace()
        n, c, h, w = predict.size()
        weight = torch.FloatTensor([1,1,0.1]).cuda()
        #print('1',target)
        target_mask = (target >= 0)
        if fake_label is not None:
            fake_mask = (fake_label == 200)
            target_fake = target[fake_mask]
        #print('2',target_mask)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        if fake_label is not None:
            predict_fake = predict[fake_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        #predict = predict.view(-1, c)
        #target = target.view(-1, 1)
        #import pdb;pdb.set_trace()
        loss = F.cross_entropy(predict, target, weight=weight, size_average=size_average)
        if fake_label is not None:
            loss = loss + F.cross_entropy(predict_fake, target_fake, weight=weight, size_average=size_average)*0.1
        return loss    

    def Smoothloss(self, predict, target):
        #import pdb;pdb.set_trace()
        n, c, h, w = predict.size()
        predict = F.sigmoid(predict)
        
        loss = abs(predict[:, 0, 1: h-1, 1: w-1] - predict[:, 0, 0: h-2, 1: w-1]) + \
               abs(predict[:, 0, 1: h-1, 1: w-1] - predict[:, 0, 2: h, 1: w-1]) + \
               abs(predict[:, 0, 1: h-1, 1: w-1] - predict[:, 0, 1: h-1, 0: w-2]) + \
               abs(predict[:, 0, 1: h-1, 1: w-1] - predict[:, 0, 1: h-1, 2: w])       
        
        M1 = torch.zeros(loss.shape).cuda()
        M2 = torch.zeros(loss.shape).cuda()
        M3 = torch.zeros(loss.shape).cuda()
        M4 = torch.zeros(loss.shape).cuda()

        M1[target[:, 1: h-1, 1: w-1] ==  target[:, 0: h-2, 1: w-1]] = 1 
        M2[target[:, 1: h-1, 1: w-1] ==  target[:, 2: h, 1: w-1]] = 1
        M3[target[:, 1: h-1, 1: w-1] ==  target[:, 1: h-1, 0: w-2]] = 1
        M4[target[:, 1: h-1, 1: w-1] ==  target[:, 1: h-1, 2: w]] = 1
        loss = loss * M1 * M2 * M3 * M4
        loss = loss.mean()
        return loss

def Deeplab(num_classes=3, init_weights=None, restore_from=None, phase='train'):
    model = ResNet101(Bottleneck, [3, 4, 23, 3], num_classes, phase)
    if init_weights is not None:
        saved_state_dict = torch.load(init_weights, map_location=lambda storage, loc: storage)
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if  not i_parts[1] == 'layer5':
                new_params['.'.join(i_parts[1:])] = saved_state_dict[i]
                #new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)
        print("load deeplab successfully")
    
    if restore_from is not None: 
        model.load_state_dict(torch.load(restore_from + '.pth', map_location=lambda storage, loc: storage))   
        print(restore_from + '.pth. successful')
    #summary(model,(3, 224,224))
    
    return model

'''
class DeepLab(nn.Module):
    def __init__(self, num_classes=19, init_weights=None, restore_from=None):
        super(DeepLab, self).__init__()
        self.pretrained = deeplab(num_classes=num_classes, init_weights=None, restore_from=None)
'''





