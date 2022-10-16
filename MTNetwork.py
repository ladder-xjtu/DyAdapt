# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:30:32 2019

@author: chlian
"""
import future
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from adaptive_convolution import AdaptiveConv2d

import numpy as np

class HeadConv(nn.Module):
    def __init__(self, in_chns, c1_chns, c2_chns, c4_chns,
                 kernel_size=3, scale_factor=4, activation=F.relu, with_bn=False):
        super(HeadConv, self).__init__()
        self.conv1 = nn.Conv2d(in_chns, c1_chns, kernel_size,
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(c1_chns, c2_chns, kernel_size,
                               stride=1, padding=1)
        self.conv4 = nn.Conv2d(c2_chns, c4_chns, scale_factor,
                               stride=scale_factor, padding=0, groups=c2_chns) #
        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(c1_chns)
            self.bn2 = nn.BatchNorm2d(c2_chns)
            self.bn4 = nn.BatchNorm2d(c4_chns)
        else:
            self.gn1 = nn.GroupNorm(16, c1_chns)
            self.gn2 = nn.GroupNorm(16, c2_chns)
            self.gn4 = nn.GroupNorm(16, c4_chns)
        self.activation = activation
        self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        if self.with_bn:
            x = self.bn1(x)
        else:
            x = self.gn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        if self.with_bn:
            x = self.bn2(x)
        else:
            x = self.gn2(x)
        x = self.activation(x)
        c1 = self.conv4(x)
        if self.with_bn:
            c1 = self.bn4(c1)
        else:
            c1 = self.gn4(c1)
        c1 = self.activation(c1)

        return c1, x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)  # defaults to 1
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class ConvBlock(nn.Module):
    def __init__(self, in_chns, c1_chns, c2_chns, kernel_size=3, 
                 activation=F.relu, with_res=True,
                 with_bn=False, with_drop=False, drop_rate=0.5):
        super(ConvBlock, self).__init__()        
        self.conv1 = nn.Conv2d(in_chns, c1_chns, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c1_chns, c2_chns, kernel_size, stride=1, padding=1)
        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(c1_chns)
            self.bn2 = nn.BatchNorm2d(c2_chns)
        else:
            self.gn1 = nn.GroupNorm(16, c1_chns)
            self.gn2 = nn.GroupNorm(16, c2_chns)

        self.with_drop = with_drop
        if with_drop:
            self.dropout = nn.Dropout2d(drop_rate)
        self.activation = activation
        self.with_res = with_res
        
        self.in_chns = in_chns
        self.ot_chns = c2_chns
        
        if with_res:
            if not in_chns == c2_chns:
                self.convX = nn.Conv2d(in_chns, c2_chns, kernel_size=1,
                                       stride=1, padding=0)
                if with_bn: 
                    self.bnX = nn.BatchNorm2d(c2_chns)
                else:
                    self.gnX = nn.GroupNorm(16, c2_chns)
                    
        self._initialize_weights()
    
    def forward(self, x):
        c1 = self.conv1(x)
        if self.with_bn:
            c1 = self.bn1(c1)
        else:
            c1 = self.gn1(c1)
        c1 = self.activation(c1)        
        c2 = self.conv2(c1)
        if self.with_bn:
            c2 = self.bn2(c2)
        else:
            c2 = self.gn2(c2)
        if self.with_res:
            if not self.in_chns == self.ot_chns:
                x = self.convX(x)
                if self.with_bn:
                    x = self.bnX(x)
                else:
                    x = self.gnX(x)
            c2 = c2 + x
        c2 = self.activation(c2)
        if self.with_drop:
            c2 = self.dropout(c2)
        return c1, c2
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight) # defaults to 1
                init.constant_(m.bias, 0)
            elif isinstance (m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class DeconvBlock(nn.Module):
    def __init__(self, in_hchns, in_lchns, c1_chns, c2_chns, kernel_size=3, 
                 interpolator='bilinear', with_res=True,
                 with_bn=False, with_drop=False, drop_rate=0.5):
        super(DeconvBlock, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode=interpolator, align_corners=True)
        #self.up = nn.ConvTranspose2d(in_lchns, in_lchns, 2, stride=2)
        '''
        if interpolator:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_lchns, in_lchns, 2, stride=2)
        '''
        self.conv1 = nn.Conv2d(in_hchns+in_lchns, c1_chns, kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c1_chns, c2_chns, kernel_size, stride=1, padding=1)
        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(c1_chns)
            self.bn2 = nn.BatchNorm2d(c2_chns)
        else:
            self.gn1 = nn.GroupNorm(16, c1_chns)
            self.gn2 = nn.GroupNorm(16, c2_chns)
        self.with_drop = with_drop
        if with_drop:
            self.dropout = nn.Dropout2d(drop_rate)
        self.activation1 = F.relu
        self.activation2 = F.relu
        self.with_res = with_res
        
        self.in_chns = in_hchns + in_lchns
        self.ot_chns = c2_chns
        
        if with_res:
            if not (in_hchns+in_lchns) == c2_chns:
                self.convX = nn.Conv2d(in_hchns+in_lchns, c2_chns, 1, stride=1, padding=0)
                if with_bn: 
                    self.bnX = nn.BatchNorm2d(c2_chns)
                else:
                    self.gnX = nn.GroupNorm(16, c2_chns)
        self._initialize_weights()
        
    def forward(self, hx, x):
        x = self.up(x)
        x = torch.cat((hx, x), dim=1)
        c1 = self.conv1(x)
        if self.with_bn:
            c1 = self.bn1(c1)
        else:
            c1 = self.gn1(c1)
        c1 = self.activation1(c1)        
        c2 = self.conv2(c1)
        if self.with_bn:
            c2 = self.bn2(c2)
        else:
            c2 = self.gn2(c2)
        if self.with_res:
            if not self.in_chns == self.ot_chns:
                x = self.convX(x)
                if self.with_bn:
                    x = self.bnX(x)
                else:
                    x = self.gnX(x)
            c2 = c2 + x
        c2 = self.activation2(c2)
        if self.with_drop:
            c2 = self.dropout(c2)
        return c1, c2
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight) # defaults to 1
                init.constant_(m.bias, 0)
            elif isinstance (m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class OTLayer(nn.Module):
    def __init__(self, in_chns, ot_chns, num_ots,
                 with_bn=False, with_drop=True, drop_rate=0.3):
        super(OTLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_chns, ot_chns,
                               1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(ot_chns, num_ots,
                               1, stride=1, padding=0)

        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(ot_chns)
        else:
            self.gn1 = nn.GroupNorm(16, ot_chns)
        self.with_drop = with_drop
        if with_drop:
            self.dropout = nn.Dropout2d(drop_rate)
        self.activation = F.relu

        self._initialize_weights()

    def forward(self, x):
        '''
        x = self.conv1(x)
        
        if self.with_bn:
            x = self.bn1(x)
        else:
            x = self.gn1(x)
        
        x = self.activation(x)
        
        if self.with_drop:
            x = self.dropout(x)
        '''
        y = self.conv2(x)
        return y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)  # defaults to 1
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class EndDeconv(nn.Module):
    def __init__(self, in_hchns, in_lchns, c1_chns,
                 num_ots, kernel_size=3,
                 interpolator='bilinear', scale_factor=2,
                 with_bn=False, with_drop=True, drop_rate=0.3):
        super(EndDeconv, self).__init__()
	
        self.up1 = nn.Upsample(scale_factor=scale_factor,
                              mode=interpolator, align_corners=True)

        self.up = nn.ConvTranspose2d(in_lchns, in_lchns, scale_factor, stride=scale_factor)
        self.conv1 = nn.Conv2d(in_hchns + in_lchns, c1_chns,
                               kernel_size, stride=1, padding=1)
        self.conv2 = nn.Conv2d(c1_chns, num_ots,
                               1, stride=1, padding=0) #+num_ots

        self.with_bn = with_bn
        if with_bn:
            self.bn1 = nn.BatchNorm2d(c1_chns)
        else:
            self.gn1 = nn.GroupNorm(16, c1_chns)
        self.with_drop = with_drop
        if with_drop:
            self.dropout = nn.Dropout2d(drop_rate)
        self.activation = F.relu

        self._initialize_weights()

    def forward(self, hx, x, s):
        x = self.up(x)
        x = torch.cat((hx, x), dim=1)
        x = self.conv1(x)
        if self.with_bn:
            x = self.bn1(x)
        else:
            x = self.gn1(x)
        x = self.activation(x)
        if self.with_drop:
            x = self.dropout(x)

        #s = self.up1(s)
        #x = torch.cat((x, s), dim=1)
        
        y = self.conv2(x)
        return x, y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)  # defaults to 1
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class TransAtt(nn.Module):
    def __init__(self, sx1_chns, sx2_chns, t_chns, a1_chns, e1_chns,
                 e2_chns, ot_chns, ebd_factor=2, with_bn=False, with_res=False):
        super(TransAtt, self).__init__()
        ebd_chns = int((sx2_chns) / ebd_factor)  # low-dimensional feature embedding
        self.trans1 = nn.Conv2d(sx1_chns + t_chns, e1_chns, 1, stride=1, padding=0)
        self.trans2 = nn.Conv2d(e1_chns, e2_chns, 1, stride=1, padding=0, bias=False)
        self.trans3 = nn.Conv2d(e2_chns, sx2_chns, 1, stride=1, padding=0, bias=False)
        self.U = nn.Conv2d(sx2_chns, sx2_chns, 1, stride=1, padding=0)
        self.V = nn.Conv2d(sx2_chns, ebd_chns, 1, stride=1, padding=0)
        self.att1 = nn.Conv2d(ebd_chns + t_chns, a1_chns, 1, stride=1, padding=0)
        self.att2 = nn.Conv2d(a1_chns, ebd_chns + t_chns, 1, stride=1, padding=0)
        self.ot_conv = nn.Conv2d(ebd_chns + t_chns, ot_chns, 3, stride=1, padding=1)
        self.with_bn = with_bn
        if with_bn:
            self.t1_bn = nn.BatchNorm2d(e1_chns)
            self.t2_bn = nn.BatchNorm2d(e2_chns)
            self.U_bn = nn.BatchNorm2d(sx2_chns)
            self.V_bn = nn.BatchNorm2d(ebd_chns)
            self.a_bn1 = nn.BatchNorm2d(a1_chns)
            self.a_bn2 = nn.BatchNorm2d(ebd_chns + t_chns)
            self.ot_bn = nn.BatchNorm2d(ot_chns)
        self.relu = F.relu
        self.sigmoid = torch.sigmoid  # instead of nn.functional.sigmoid
        self.t_activation = F.relu
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.with_res = with_res
        self._initialize_weights()

    def forward(self, sx1, sx2, tx):
        b = torch.cat((sx1, tx), dim=1)
        w = self.trans1(b)
        if self.with_bn:
            w = self.t1_bn(w)
        w = self.relu(w)
        w = self.gap(w)
        w = self.trans2(w)
        if self.with_bn:
            w = self.t2_bn(w)
        w = self.relu(w)
        w = self.trans3(w)
        w = F.normalize(w)
        b = self.U(sx2)
        if self.with_bn:
            b = self.U_bn(b)
        b = self.relu(b)
        aconv = AdaptiveConv2d(b.size(0) * b.size(1),
                               b.size(0) * b.size(1),
                               1, stride=1, padding=0,
                               groups=b.size(0) * b.size(1),
                               bias=False)  # adpative 1x1x1 conv
        b = aconv(b, w)
        b = self.V(b)
        if self.with_bn:
            b = self.V_bn(b)
        b = self.relu(b)
        # attention
        b = torch.cat((b, tx), dim=1)
        a = self.att1(b)
        if self.with_bn:
            a = self.a_bn1(a)
        a = self.relu(a)
        a = self.att2(a)
        if self.with_bn:
            a = self.a_bn2(a)
        a = self.sigmoid(a)
        a = a * b
        a = self.ot_conv(a)
        if self.with_bn:
            a = self.ot_bn(a)
        a = self.relu(a)
        return a, w

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)  # defaults to 1
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class TransDeAtt(nn.Module):
    def __init__(self, sx1_chns, sx2_chns, thx_chns, tlx_chns,
                 a1_chns, e1_chns, e2_chns, ot_chns, ebd_factor=2,
                 interpolator='bilinear', with_bn=False, with_res=True):
        super(TransDeAtt, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode=interpolator, align_corners=True)
        #self.up = nn.ConvTranspose2d(tlx_chns, tlx_chns, 2, stride=2)
        ebd_chns = int((sx2_chns) / ebd_factor)
        tx_chns = tlx_chns + thx_chns
        self.trans1 = nn.Conv2d(sx1_chns + tx_chns, e1_chns, 1, stride=1, padding=0)
        self.trans2 = nn.Conv2d(e1_chns, e2_chns, 1, stride=1, padding=0, bias=False)
        self.trans3 = nn.Conv2d(e2_chns, sx2_chns, 1, stride=1, padding=0, bias=False)
        self.U = nn.Conv2d(sx2_chns, sx2_chns, 1, stride=1, padding=0)
        self.V = nn.Conv2d(sx2_chns, ebd_chns, 1, stride=1, padding=0)
        self.att1 = nn.Conv2d(ebd_chns + tx_chns, a1_chns, 1, stride=1, padding=0)
        self.att2 = nn.Conv2d(a1_chns, ebd_chns + tx_chns, 1, stride=1, padding=0)
        self.ot_conv = nn.Conv2d(ebd_chns + tx_chns, ot_chns, 3, stride=1, padding=1)
        self.with_bn = with_bn
        if with_bn:
            self.t1_bn = nn.BatchNorm2d(e1_chns)
            self.t2_bn = nn.BatchNorm2d(e2_chns)
            self.U_bn = nn.BatchNorm2d(sx2_chns)
            self.V_bn = nn.BatchNorm2d(ebd_chns)
            self.a_bn1 = nn.BatchNorm2d(a1_chns)
            self.a_bn2 = nn.BatchNorm2d(ebd_chns + tx_chns)
            self.ot_bn = nn.BatchNorm2d(ot_chns)
        self.relu = F.relu
        self.sigmoid = torch.sigmoid  # instead of nn.functional.sigmoid
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self._initialize_weights()

    def forward(self, sx1, sx2, t_hx, tx):
        tx = self.up(tx)
        tx = torch.cat((t_hx, tx), dim=1)
        # embedding
        st = torch.cat((sx1, tx), dim=1)
        w = self.trans1(st)
        if self.with_bn:
            w = self.t1_bn(w)
        w = self.relu(w)
        w = self.gap(w)
        w = self.trans2(w)
        if self.with_bn:
            w = self.t2_bn(w)
        w = self.relu(w)
        w = self.trans3(w)
        w = F.normalize(w)
        b = self.U(sx2)
        if self.with_bn:
            b = self.U_bn(b)
        b = self.relu(b)
        aconv = AdaptiveConv2d(b.size(0) * b.size(1),
                               b.size(0) * b.size(1),
                               1, stride=1, padding=0,
                               groups=b.size(0) * b.size(1),
                               bias=False)  # adpative 1x1x1 conv
        b = aconv(b, w)
        b = self.V(b)
        if self.with_bn:
            b = self.V_bn(b)
        b = self.relu(b)
        # attention
        b = torch.cat((b, tx), dim=1)
        a = self.att1(b)
        if self.with_bn:
            a = self.a_bn1(a)
        a = self.relu(a)
        a = self.att2(a)
        if self.with_bn:
            a = self.a_bn2(a)
        a = self.sigmoid(a)
        a = a * b
        a = self.ot_conv(a)
        if self.with_bn:
            a = self.ot_bn(a)
        a = self.relu(a)
        return a, w

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)  # defaults to 1
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class DyAdaptNet(nn.Module):
    def __init__(self, in_chns=3, num_ot1=2, num_ot2=3,
                 scale_factor=2,
                 with_res=True, with_bn=True,
                 with_drop=False, drop_rate=0.5):
        super(DyAdaptNet, self).__init__()
        self.s_hconv = HeadConv(in_chns, 32, 32, 32,
                                scale_factor=scale_factor, with_bn=with_bn)

        self.s_conv1 = ConvBlock(in_chns, 32, 32, with_res=False, with_bn=with_bn)
        self.s_pool1 = nn.MaxPool2d(2)

        self.head_fuse = nn.Conv2d(32 + 32, 32, 3, stride=1, padding=1)
        self.head_fuse_bn = nn.BatchNorm2d(32)
        self.head_fuse_gn = nn.GroupNorm(16, 32)
        self.head_fuse_activation = F.relu

        self.s_conv2 = ConvBlock(64, 64, 64, with_res=with_res, with_bn=with_bn)
        self.s_pool2 = nn.MaxPool2d(2)
        self.s_conv3 = ConvBlock(64, 64, 128, with_res=with_res, with_bn=with_bn)
        self.s_pool3 = nn.MaxPool2d(2)
        self.s_conv4 = ConvBlock(128, 128, 256, with_res=with_res, with_bn=with_bn)
        self.s_pool4 = nn.MaxPool2d(2)
        self.s_conv5 = ConvBlock(256, 256, 512, with_res=with_res, with_bn=with_bn)
        # in_hchns, in_lchns, c1_chns, c2_chns
        self.s_dconv0 = DeconvBlock(256, 512, 256, 256, with_res=with_res, with_bn=with_bn)
        self.s_dconv1 = DeconvBlock(128, 256, 128, 128, with_res=with_res, with_bn=with_bn)
        # in_hchns, in_lchns, c1_chns, c2_chns
        self.s_dconv2 = DeconvBlock(64, 128, 64, 64, with_res=with_res, with_bn=with_bn)
        self.s_dconv3 = DeconvBlock(64, 64, 64, 64, with_res=with_res, with_bn=with_bn)
        # in_hchns, in_lchns, c1_chns, c2_chns, num_ots
        self.s_output = EndDeconv(32, 64, 64, num_ot1, scale_factor=scale_factor,
                                  with_drop=with_drop, drop_rate=drop_rate)
        self.ds_output = OTLayer(64, 64, num_ot1, with_bn=with_bn, with_drop=with_drop, drop_rate=drop_rate)

        self.t1_fuse = nn.Conv2d(32 + 64, 32, 3, stride=1, padding=1)
        self.t1_fuse_bn = nn.BatchNorm2d(32)
        self.t1_fuse_gn = nn.GroupNorm(16, 32)
        self.t1_fuse_activation = F.relu

        self.t1_transformer1 = TransAtt(32, 64, 64, 128, 128, 8, 64, ebd_factor=1, with_res=True, with_bn=with_bn)
        self.t1_pool1 = nn.MaxPool2d(2)

        self.t1_transformer2 = TransAtt(64, 64, 64, 128, 128, 16, 128, ebd_factor=1, with_res=True, with_bn=with_bn)
        self.t1_pool2 = nn.MaxPool2d(2)

        self.t1_transformer3 = TransAtt(64, 128, 128, 128, 128, 16, 256, ebd_factor=2, with_res=True, with_bn=with_bn)
        self.t1_pool3 = nn.MaxPool2d(2)

        self.t1_transformer4 = TransAtt(128, 256, 256, 256, 256, 32, 512, ebd_factor=2, with_res=True, with_bn=with_bn)
        self.t1_pool4 = nn.MaxPool2d(2)

        self.t1_transformer5 = TransAtt(256, 512, 512, 512, 512, 64, 512, ebd_factor=2, with_res=True, with_bn=with_bn)

        self.t1_detrans0 = TransDeAtt(256, 256, 512, 512, 512, 512, 64, 512, ebd_factor=2, with_res=True, with_bn=with_bn)

        self.t1_detrans1 = TransDeAtt(128, 128, 256, 512, 256, 256, 32, 256, ebd_factor=2, with_res=True, with_bn=with_bn)

        self.t1_detrans2 = TransDeAtt(64, 64, 128, 256, 128, 128, 16, 128, ebd_factor=1, with_res=True, with_bn=with_bn)

        self.t1_detrans3 = TransDeAtt(64, 64, 64, 128, 128, 128, 16, 64, ebd_factor=1, with_res=True, with_bn=with_bn)

        self.t1_output = EndDeconv(32, 64, 64, num_ot2, scale_factor=scale_factor,
                                   with_drop=with_drop, drop_rate=drop_rate)

        self.dt1_output = OTLayer(64, 64, num_ot2, with_bn=with_bn, with_drop=with_drop, drop_rate=drop_rate)

        self.softmax = nn.Softmax(dim=1)

        self.h_pool = nn.MaxPool2d(2)
        self.with_bn = with_bn

        self.in_chs = in_chns
        self._initialize_weights()

    def forward(self, hx, x):
        s_eh_1, s_eh_2 = self.s_hconv(hx)

        # --- Backbone ---
        # encoder:
        s_e1_1, s_e1_2 = self.s_conv1(x)

        s_e1_2 = torch.cat((s_eh_1, s_e1_2), dim=1)

        s_e1_p = self.s_pool1(s_e1_2)
        s_e2_1, s_e2_2 = self.s_conv2(s_e1_p)
        s_e2_p = self.s_pool2(s_e2_2)
        s_e3_1, s_e3_2 = self.s_conv3(s_e2_p)
        s_e3_p = self.s_pool3(s_e3_2)
        s_e4_1, s_e4_2 = self.s_conv4(s_e3_p)
        s_e4_p = self.s_pool4(s_e4_2)
        s_e5_1, s_e5_2 = self.s_conv5(s_e4_p)
        # decoder:
        s_d0_1, s_d0_2 = self.s_dconv0(s_e4_2, s_e5_2)
        s_d1_1, s_d1_2 = self.s_dconv1(s_e3_2, s_d0_2)
        s_d2_1, s_d2_2 = self.s_dconv2(s_e2_2, s_d1_2)
        s_d3_1, s_d3_2 = self.s_dconv3(s_e1_2, s_d2_2)

        ds_output = self.ds_output(s_d3_2)
        ds_output = self.softmax(ds_output)
        s_source, s_output = self.s_output(s_eh_2, s_d3_2, ds_output)
        s_output = self.softmax(s_output)

        # --- Dependent Task ---
        # encoder:
        t1_e1, w1_e1 = self.t1_transformer1(s_e1_1, s_e1_2, s_d3_2)  # , self.h_pool(s_source)
        t1_e1_p = self.t1_pool1(t1_e1)

        t1_e2, w1_e2 = self.t1_transformer2(s_e2_1, s_e2_2, t1_e1_p)
        t1_e2_p = self.t1_pool2(t1_e2)

        t1_e3, w1_e3 = self.t1_transformer3(s_e3_1, s_e3_2, t1_e2_p)
        t1_e3_p = self.t1_pool3(t1_e3)

        t1_e4, w1_e4 = self.t1_transformer4(s_e4_1, s_e4_2, t1_e3_p)
        t1_e4_p = self.t1_pool4(t1_e4)

        t1_e5, w1_e5 = self.t1_transformer5(s_e5_1, s_e5_2, t1_e4_p)
        # decoder:
        t1_d0, w1_d0 = self.t1_detrans0(s_d0_1, s_d0_2, t1_e4, t1_e5)
        t1_d1, w1_d1 = self.t1_detrans1(s_d1_1, s_d1_2, t1_e3, t1_d0)
        t1_d2, w1_d2 = self.t1_detrans2(s_d2_1, s_d2_2, t1_e2, t1_d1)
        t1_d3, w1_d3 = self.t1_detrans3(s_d3_1, s_d3_2, t1_e1, t1_d2)

        dt1_output = self.dt1_output(t1_d3)
        dt1_output = self.softmax(dt1_output)
        _, t1_output = self.t1_output(s_eh_2, t1_d3, dt1_output)
        t1_output = self.softmax(t1_output)

        w1 = torch.cat((w1_e4, w1_e5, w1_d0), dim=1)  # w1_e1, w1_e2, w1_e3, , w1_d1, w1_d2, w1_d3

        return s_output, t1_output, ds_output, dt1_output, w1

    def name(self):
        return 'Dynamic Adaptive Transformer Network (Input channel = {0:d})'.format(self.in_chs)

    def _initialize_weights(self):
        init.xavier_normal_(self.t1_fuse.weight)
        init.constant_(self.t1_fuse.bias, 0)
        init.xavier_normal_(self.head_fuse.weight)
        init.constant_(self.head_fuse.bias, 0)
        if self.with_bn:
            init.constant_(self.t1_fuse_bn.weight, 1)
            init.constant_(self.t1_fuse_bn.bias, 0)
            init.constant_(self.head_fuse_bn.weight, 1)
            init.constant_(self.head_fuse_bn.bias, 0)


if __name__ == '__main__':

    net = DyAdaptNet(in_chns=3, num_ot1=3, num_ot2=2, scale_factor=2,
                with_res=True, with_bn=True, with_drop=True, drop_rate=0.5)
    x = torch.rand(2, 3, 128, 128)
    hx = torch.rand(2, 3, 256, 256)
    y = net2(hx, x)
    print(y[0].shape, y[1].shape)

