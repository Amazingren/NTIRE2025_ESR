import math
from collections import OrderedDict

import torch
from torch import nn as nn
import torch.nn.functional as F


class GCT(nn.Module):
    def __init__(self, num_channels, epsilon=1e-5, mode='l2', after_relu=False):
        super(GCT, self).__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.epsilon = epsilon
        self.mode = mode
        self.after_relu = after_relu

    def forward(self, x):
        if self.mode == 'l2':
            embedding = (x.pow(2).sum((2, 3), keepdim=True) + self.epsilon).pow(0.5) * self.alpha
            norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon).pow(0.5)

        elif self.mode == 'l1':
            if not self.after_relu:
                _x = torch.abs(x)
            else:
                _x = x
            embedding = _x.sum((2, 3), keepdim=True) * self.alpha
            norm = self.gamma / (torch.abs(embedding).mean(dim=1, keepdim=True) + self.epsilon)

        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate


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


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
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


class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, s=1, bias=True):
        super(Conv3XC, self).__init__()
        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)

    def forward(self, x):
        out = self.eval_conv(x)
        return out


class EARB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(EARB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels)
        self.c2_r = Conv3XC(mid_channels, mid_channels)
        self.c3_r = Conv3XC(mid_channels, out_channels)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.gct1 = GCT(mid_channels)
        self.gct2 = GCT(mid_channels)
        self.gct3 = GCT(out_channels)


    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_gct = self.gct1(out1)
        out1_act = self.act1(out1_gct)


        out2 = (self.c2_r(out1_act))
        out2_gct = self.gct2(out2)
        out2_act = self.act1(out2_gct)


        out3 = (self.c3_r(out2_act))
        out3_gct = self.gct3(out3)

        sim_att = torch.sigmoid(out3_gct) - 0.5
        out = (out3_gct + x) * sim_att

        return out, out1, sim_att


class ECAS(nn.Module):
    """
    Efficient channel attention super-resolution network acting on space
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=40,
                 upscale=4,
                 bias=True,
                 img_range=1.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)
                 ):
        super(ECAS, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_1 = Conv3XC(in_channels, feature_channels, s=1)
        self.block_1 = EARB(feature_channels, bias=bias)
        self.block_2 = EARB(feature_channels, bias=bias)
        self.block_3 = EARB(feature_channels, bias=bias)
        self.block_4 = EARB(feature_channels, bias=bias)
        self.block_5 = EARB(feature_channels, bias=bias)
        self.block_6 = EARB(feature_channels, bias=bias)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        gamma = 2
        b = 1
        kernel_size = int(abs((math.log(feature_channels * 4, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.conv_2 = Conv3XC(feature_channels, feature_channels, s=1)

        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        out_feature = self.conv_1(x)

        out_b1, _, att1 = self.block_1(out_feature)
        out_b2, _, att2 = self.block_2(out_b1)
        out_b3, _, att3 = self.block_3(out_b2)

        out_b4, _, att4 = self.block_4(out_b3)
        out_b5, _, att5 = self.block_5(out_b4)
        out_b6, out_b5_2, att6 = self.block_6(out_b5)

        out_b6 = self.conv_2(out_b6)
        out_cat = torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1)

        b, c, h, w = out_cat.size()
        avg = self.avg_pool(out_cat).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])

        out = self.conv_cat(out_cat + out_cat * out)
        output = self.upsampler(out)

        return output
