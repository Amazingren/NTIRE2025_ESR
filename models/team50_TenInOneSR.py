from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F
import math

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

class LocalAttention(nn.Module):
    def __init__(self, channels, f=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, f, 1),  
            torch.nn.MaxPool2d(kernel_size=7, stride=3, padding=0),
            nn.Conv2d(f, f, kernel_size=3, stride=2, padding=1),   
            nn.Conv2d(f, channels, 3, padding=1),                  
            nn.Sigmoid(), 
        )

    def forward(self, x):
        g = torch.sigmoid(x[:,:1]) 
        w = F.interpolate(self.body(x), (x.size(2), x.size(3)), mode='bilinear', align_corners=False)  
        return w * g * x 

class TenInOneConv(nn.Module):
    def __init__(self, c_in, c_out, gains=[1, 2, 3], s=1, bias=True):
        super(TenInOneConv, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.stride = s
        num_branches = len(gains)  

        # # Multiple branches with multiple gains
        # self.branches = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Conv2d(c_in, c_in * gain, kernel_size=1, padding=0, bias=bias),
        #         nn.Conv2d(c_in * gain, c_out * gain, kernel_size=3, stride=s, padding=0, bias=bias),
        #         nn.Conv2d(c_out * gain, c_out, kernel_size=1, padding=0, bias=bias),
        #     )
        #     for gain in gains
        # ])

        # # Skip
        # self.sk = nn.Conv2d(c_in, c_out, kernel_size=1, padding=0, stride=s, bias=bias)

        # Eval Conv
        self.eval_conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = True
        self.eval_conv.bias.requires_grad = True

        # self.update_params()

    def update_params(self):
        total_weight = 0
        total_bias = 0

        for branch in self.branches:
            w1 = branch[0].weight.data.clone().detach()
            b1 = branch[0].bias.data.clone().detach()
            w2 = branch[1].weight.data.clone().detach()
            b2 = branch[1].bias.data.clone().detach()
            w3 = branch[2].weight.data.clone().detach()
            b3 = branch[2].bias.data.clone().detach()

            w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
            b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

            branch_weight = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
            branch_bias = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

            total_weight += branch_weight
            total_bias += branch_bias

        sk_w = self.sk.weight.data.clone().detach()
        sk_b = self.sk.bias.data.clone().detach()
        target_kernel_size = 3
        padding = (target_kernel_size - 1) // 2
        sk_w_padded = F.pad(sk_w, [padding, padding, padding, padding])

        self.weight_concat = total_weight + sk_w_padded
        self.bias_concat = total_bias + sk_b

        self.eval_conv.weight.data = self.weight_concat
        self.eval_conv.bias.data = self.bias_concat

    def forward(self, x):
        if self.training:
            pad = 1
            x_pad = F.pad(x, (pad, pad, pad, pad), "constant", 0)
            out = sum(branch(x_pad) for branch in self.branches) + self.sk(x)
        else:
            # self.update_params()
            out = self.eval_conv(x)

        return out


class TenInOneBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,):
        super(TenInOneBlock, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = TenInOneConv(in_channels, mid_channels, s=1)
        self.c2_r = TenInOneConv(mid_channels, mid_channels, s=1)
        self.c3_r = TenInOneConv(mid_channels, out_channels, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        self.attn = LocalAttention(out_channels)


    def forward(self, x):
        x = self.attn(x)
        res = x.clone()
        out1 = self.c1_r(x)
        out1_act = self.act1(out1)
        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)
        out3 = self.c3_r(out2_act)
        return out3 + res, out1

    
# @ARCH_REGISTRY.register()
class TenInOneSR(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=48,
                 upscale=4,                 
                 num_blocks=4): 
        super(TenInOneSR, self).__init__()
        self.num_blocks = num_blocks

        self.conv_1 = TenInOneConv(num_in_ch, feature_channels, s=1)
        self.blocks = nn.ModuleList([TenInOneBlock(feature_channels) for _ in range(num_blocks)])
        self.conv_2 = TenInOneConv(feature_channels, feature_channels, s=1)
        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True)
        self.tail = TenInOneConv(feature_channels,
                              num_out_ch * (upscale ** 2),
                              gains=[1, 2, 3],
                              s=1,
                              bias=True)
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        shortcut = torch.repeat_interleave(x, 16, dim=1)
        out_feature = self.conv_1(x)
        
        block_out1 = out_feature
        for i, block in enumerate(self.blocks):
            block_out1, block_out2 = block(block_out1)
            if i == self.num_blocks - 1:  
                concat1 = block_out1
                concat2 = block_out2

        out = block_out1 + out_feature
        out = self.conv_2(block_out1)
        out = self.conv_cat(torch.cat([out_feature, concat1, concat2, out], 1))
        out = self.tail(out) + shortcut
        output = self.upsampler(out)
        return output