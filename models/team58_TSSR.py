from collections import OrderedDict
import torch
from torch import nn as nn

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

class Conv(nn.Module):
    def __init__(self, c_in, c_out, s=1, bias=True):
        super(Conv, self).__init__()
        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)            
    def forward(self, x):
        out = self.eval_conv(x)
        return out

class REECB(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False,
                 extern_conv=0,):
        super(REECB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels

        self._extern_conv = extern_conv
        self._extern_conv_block = nn.ModuleList()
        if self._extern_conv > 0:
            for i in range(self._extern_conv):
                self._extern_conv_block.append(Conv(mid_channels, mid_channels, s=1, bias=bias))

        self.c1_r = Conv(in_channels, mid_channels, s=1, bias=bias)
        self.c2_r = Conv(mid_channels, mid_channels, s=1, bias=bias)
        self.c3_r = Conv(mid_channels, out_channels, s=1, bias=bias)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)
        out2_act = self.act1(self.c2_r(out1_act))

        for extern_conv in self._extern_conv_block:
            out2_act = self.act1(extern_conv(out2_act))

        out3 = (self.c3_r(out2_act))
        out = self.act1(out3) + x
        return out
    
class TSSR(nn.Module):
    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=32,
                 upscale=4,
                 bias=True,
                 num_block=4
                 ):
        super(TSSR, self).__init__()

        num_in_ch = num_in_ch
        upscale = upscale
        feature_channels = feature_channels
        num_block = 4
        extern_conv = 1
        use_bias = True
           
        in_channels = num_in_ch
        out_channels = num_out_ch
        out_channels = in_channels

        self.conv_1 = Conv(in_channels, feature_channels, s=1, bias=use_bias)

        layers = []
        for i in range(num_block):
            extern_conv_num = extern_conv
            layers.append(REECB(feature_channels, bias=use_bias, extern_conv=extern_conv_num))
        self.blocks = nn.Sequential(*layers)

        self.conv_2 = nn.Conv2d(feature_channels, feature_channels, 1, 1, 0, )
        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

    def to(self, device):
        self.conv_1.to(device)
        self.blocks.to(device)
        self.conv_2.to(device)
        self.upsampler.to(device)
        
        inp = torch.randn(1, 3, 256, 256).to(device)
        for i in range(50):
            out = self(inp)
        return self

    def forward(self, x):
        out_feature = self.conv_1(x)
        res_feat = out_feature

        out_feature = self.blocks(out_feature)
        out = self.conv_2(out_feature)

        out = out + res_feat
        output = self.upsampler(out)

        return output
