# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class myResnet(nn.Module):
    def __init__(self, resnet):
        super(myResnet, self).__init__()
        self.resnet = resnet

    def forward(self, img, att_size=14):
        x = img.unsqueeze(0)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)


        # fc means fully connection.
        # fc的尺寸是 fc = list[], len(fc)=2048
        fc = x.mean(3).mean(2).squeeze()
        # 自适应avg pooling,使得x经过pooling之后变成[att_size,att_size]的尺寸。
        '''
        >>> x = torch.randn(2, 3, 5)
        >>> x.size()
        torch.Size([2, 3, 5])
        >>> x.permute(2, 0, 1).size()
        torch.Size([5, 2, 3])
        '''
        # att的尺寸是 14*14*2048
        att = F.adaptive_avg_pool2d(x,[att_size,att_size]).squeeze().permute(1, 2, 0)
        
        return fc, att

