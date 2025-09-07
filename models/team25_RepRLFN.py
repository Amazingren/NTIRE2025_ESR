# -*- coding: utf-8 -*-
# Copyright 2022 ByteDance
import copy
from collections import OrderedDict
import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
import os

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)

def activation(act_type, inplace=False, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------

    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class RepRLFN(nn.Module):
    """
    Residual Local Feature Network (RLFN)
    Model definition of RLFN in `Residual Local Feature Network for
    Efficient Super-Resolution`
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=3,
                 feature_channels=48,
                 upscale=4,
                 deploy=True):
        super(RepRLFN, self).__init__()

        self.conv_1 = conv_layer(in_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.block_1 = RepRLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)
        self.block_2 = RepRLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)
        self.block_3 = RepRLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)
        self.block_4 = RepRLFB(in_channels=feature_channels, act_type='lrelu', deploy=deploy)

        self.conv_2 = conv_layer(feature_channels,
                                       feature_channels,
                                       kernel_size=3)

        self.upsampler = pixelshuffle_block(feature_channels,
                                                  out_channels,
                                                  upscale_factor=upscale)

    def forward(self, x):
        out_feature = self.conv_1(x)

        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)

        out_low_resolution = self.conv_2(out_b4) + out_feature
        output = self.upsampler(out_low_resolution)

        return output


class RepRLFB(nn.Module):
    """
    Reparameterized Residual Local Feature Block (RepRLFB).
    """
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 act_type = 'lrelu',
                 deploy = False,
                 # esa_channels=16):
                 esa_channels=15):
        super(RepRLFB, self).__init__()

        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.c1_r = RepBlock(in_channels=in_channels, out_channels=mid_channels, act_type=act_type, deploy=deploy)
        self.c2_r = RepBlock(in_channels=in_channels, out_channels=mid_channels, act_type=act_type, deploy=deploy)
        self.c3_r = RepBlock(in_channels=in_channels, out_channels=mid_channels, act_type=act_type, deploy=deploy)

        self.c5 = conv_layer(in_channels, out_channels, 1)
        self.esa = ESA(esa_channels, out_channels, nn.Conv2d)

        self.act = activation('lrelu', neg_slope=0.05)

    def forward(self, x):
        out = (self.c1_r(x))
        out = self.act(out)

        out = (self.c2_r(out))
        out = self.act(out)

        out = (self.c3_r(out))
        out = self.act(out)

        out = out + x
        out = self.esa(self.c5(out))

        return out


class ESA(nn.Module):
    """
    Modification of Enhanced Spatial Attention (ESA), which is proposed by
    `Residual Feature Aggregation Network for Image Super-Resolution`
    """

    def __init__(self, esa_channels, n_feats, conv):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)),
                           mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m

# Reparameterization: (3*3) U (3*1) U (1*3) U (1*1) U (identity) -> (3*3)
class RepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, act_type, deploy=False):
        super(RepBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation('lrelu')

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(3, 3), stride=1,
                                         padding=1, dilation=1, groups=1, bias=True,
                                         padding_mode='zeros')
        else:
            self.rbr_3x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3),
                                            stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros')
            self.rbr_3x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                            stride=1, padding=(1, 0), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x3_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                            stride=1, padding=(0, 1), dilation=1, groups=1, padding_mode='zeros')
            self.rbr_1x1_branch = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                            stride=1, padding=(0, 0), dilation=1, groups=1, padding_mode='zeros')

    def forward(self, inputs):
        if self.deploy:
            return self.activation(self.rbr_reparam(inputs))
        else:
            return self.activation( (self.rbr_3x3_branch(inputs)) +
                                   (self.rbr_3x1_branch(inputs) + self.rbr_1x3_branch(inputs) + self.rbr_1x1_branch(inputs)) +
                                   (inputs) )

    def switch_to_deploy(self):
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=3,
                                     stride=1, padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros')
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        self.__delattr__('rbr_3x3_branch')
        self.__delattr__('rbr_3x1_branch')
        self.__delattr__('rbr_1x3_branch')
        self.__delattr__('rbr_1x1_branch')
        self.deploy = True

    def get_equivalent_kernel_bias(self):
        # 3x3 branch
        kernel_3x3, bias_3x3 = self.rbr_3x3_branch.weight.data, self.rbr_3x3_branch.bias.data
        # 1x1 1x3 3x1 branch
        kernel_1x1_1x3_3x1_fuse, bias_1x1_1x3_3x1_fuse = self._fuse_1x1_1x3_3x1_branch(self.rbr_1x1_branch,
                                                                                       self.rbr_1x3_branch,
                                                                                       self.rbr_3x1_branch)
        # identity branch
        device = kernel_1x1_1x3_3x1_fuse.device  # just for getting the device
        kernel_identity = torch.zeros(self.out_channels, self.in_channels, 3, 3, device=device)
        for i in range(self.out_channels):
            kernel_identity[i, i, 1, 1] = 1.0

        return kernel_3x3 + kernel_1x1_1x3_3x1_fuse + kernel_identity, \
               bias_3x3 + bias_1x1_1x3_3x1_fuse


    def _fuse_1x1_1x3_3x1_branch(self, conv1, conv2, conv3):
        weight = F.pad(conv1.weight.data, (1, 1, 1, 1)) + F.pad(conv2.weight.data, (0, 0, 1, 1)) + F.pad(
            conv3.weight.data, (1, 1, 0, 0))
        bias = conv1.bias.data + conv2.bias.data + conv3.bias.data
        return weight, bias


def get_RepRLFN(checkpoint=None, deploy=False):
    model = RepRLFN(in_channels=3, out_channels=3, feature_channels=48, deploy=deploy)

    # param_key_g = 'params'
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)

    return model


def repvgg_model_convert(model:torch.nn.Module, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'switch_to_deploy'):
            module.switch_to_deploy()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model


if __name__ == '__main__':
    seed_everything(2025)

    mynet = get_RepRLFN(checkpoint=None, deploy=False)
    img = torch.ones((1,3,40,40))
    out = mynet(img)
    state_dict = mynet.state_dict()
    torch.save(state_dict, 'RepRLFN.pth')
    print("size: {}".format(out.size()))
    print(out[0][0][0][:4]) # [-0.3382, -0.1902,  0.2992,  0.0100]


    mynet = get_RepRLFN(checkpoint='RepRLFN.pth', deploy=False)
    img = torch.ones((1,3,40,40))
    out = mynet(img)
    print("size: {}".format(out.size()))
    print(out[0][0][0][:4]) # [-0.3382, -0.1902,  0.2992,  0.0100]
    # model convert
    deploy_model = repvgg_model_convert(mynet, save_path='RepRLFN_deploy.pth')


    mynet = get_RepRLFN(checkpoint='RepRLFN_deploy.pth', deploy=True)
    img = torch.ones((1,3,40,40))
    out = mynet(img)
    print("size: {}".format(out.size()))
    print(out[0][0][0][:4]) # [-0.3382, -0.1902,  0.2992,  0.0100]


    from utils.model_summary import get_model_flops, get_model_activation
    model = get_RepRLFN(checkpoint=None, deploy=True)
    input_dim = (3, 256, 256)  # set the input dimension
    activations, num_conv = get_model_activation(model, input_dim)
    activations = activations / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Activations", activations))
    print("{:>16s} : {:<d}".format("#Conv2d", num_conv))
    flops = get_model_flops(model, input_dim, False)
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))
    num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    num_parameters = num_parameters / 10 ** 6
    print("{:>16s} : {:<.4f} [M]".format("#Params", num_parameters))
