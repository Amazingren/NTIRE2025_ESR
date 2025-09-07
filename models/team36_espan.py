from collections import OrderedDict
from torch.cuda.jiterator import _create_jit_fn

import torch
import torch.nn as nn
import torch.nn.functional as F


jiterator_code = """
template <typename T>
T sigmoid_mul_kernel(T out3, T x) {
    return (out3 + x) * (1.0f / (1.0f + ::expf(-out3)) - 0.5f);
}
"""

sigmoid_mul_op_fast = _create_jit_fn(jiterator_code)

def sigmoid_mul_op(out3, x):
    sim_att = torch.sigmoid(out3) - 0.5
    out = (out3 + x) * sim_att
    return out


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value


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




class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, s=1, bias=True):
        super(Conv3XC, self).__init__()
        self.stride = s


        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)


    def forward(self, x):
        out = self.eval_conv(x)
        return out




def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=4,
                       use_rep=False,
                       dep_mul=2):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = Conv3XC(in_channels,
                    out_channels * (upscale_factor ** 2),
                    s=1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)



    

class SPAB1(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 use_fast_op=True
                 ):
        super(SPAB1, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels,  s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        if use_fast_op:
            self.att = sigmoid_mul_op_fast
        else:
            self.att = sigmoid_mul_op

        # self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)


    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        # sim_att = torch.sigmoid(out3) - 0.5
        # out = (out3 + x) * sim_att

        out = self.att(out3, x)

        return out




class SPAB1T2(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 use_fast_op=True
                 ):
        super(SPAB1T2, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels

        self.c1_r = Conv3XC(in_channels, mid_channels, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)
        if use_fast_op:
            self.att = sigmoid_mul_op_fast
        else:
            self.att = sigmoid_mul_op

        # self.act2 = activation('lrelu', neg_slope=0.1, inplace=True)


    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))

        out = self.att(out2, x)

        return out



class Conv9XC(nn.Module):
    def __init__(self, c_in, c_out, s=1, bias=True):
        super(Conv9XC, self).__init__()
        self.stride = s

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=9, padding=4, stride=s, bias=bias)


    def forward(self, x):
        out = self.eval_conv(x)
        return out




class ESPAN(nn.Module):


    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=32,
                 mid_channels=32,
                 upscale=4,
                 bias=True,
                 teacher_feature_channels=32,
                 teacher_extra_depth=1,
                 use_fast_op=True
                 ):
        super(ESPAN, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch

        self.conv_1 = Conv9XC(in_channels, feature_channels, s=1, bias=True)

        self.block_1 = SPAB1(feature_channels, mid_channels = mid_channels, use_fast_op=use_fast_op)
        self.block_2 = SPAB1(feature_channels, mid_channels = mid_channels, use_fast_op=use_fast_op)
        self.block_3 = SPAB1(feature_channels, mid_channels = mid_channels, use_fast_op=use_fast_op)
        self.block_4 = SPAB1(feature_channels, mid_channels = mid_channels, use_fast_op=use_fast_op)
        self.block_5 = SPAB1(feature_channels, mid_channels = mid_channels, use_fast_op=use_fast_op)


        self.teacher_block_seq = SPAB1T2(teacher_feature_channels, mid_channels = teacher_feature_channels, use_fast_op=use_fast_op)

        self.tea_conv_cat = nn.Conv2d(feature_channels * 2 + teacher_feature_channels*2, feature_channels, 1, 1, 0, bias=True)

        self.tea_conv_2 = Conv3XC(teacher_feature_channels, teacher_feature_channels, s=1)

        # self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

        self.tea_upsampler = pixelshuffle_block(teacher_feature_channels, out_channels, upscale_factor=upscale)

        self.cuda()(torch.randn(1, 3, 512, 512).cuda())

    def forward(self, x):

        out_feature = self.conv_1(x)
        out_b1 = self.block_1(out_feature)
        out_b2 = self.block_2(out_b1)

        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)

        out_b6 = self.teacher_block_seq(out_b5)


        out_final = self.tea_conv_2(out_b6)
        out = self.tea_conv_cat(torch.cat([out_feature, out_final, out_b1, out_b6], 1))
        output = self.tea_upsampler(out)

        return output


