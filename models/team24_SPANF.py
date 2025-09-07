from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F

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

class Conv3XC(nn.Module):
    def __init__(self, c_in, c_out, gain1=1, gain2=0, s=1, bias=True, relu=False):
        super(Conv3XC, self).__init__()
        self.weight_concat = None
        self.bias_concat = None
        self.update_params_flag = False
        self.stride = s
        self.has_relu = relu
        gain = gain1

        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
        self.eval_conv.weight.requires_grad = False
        self.eval_conv.bias.requires_grad = False

    def forward(self, x):
        out = self.eval_conv(x)

        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out
    

class SPAB1(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False):
        super(SPAB1, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.mid_channels = mid_channels
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.c1_r = Conv3XC(in_channels, mid_channels, gain1=2, s=1)
        self.c2_r = Conv3XC(mid_channels, mid_channels, gain1=2, s=1)
        self.c3_r = Conv3XC(mid_channels, out_channels, gain1=2, s=1)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        if self.in_channels == self.out_channels:
            sim_att = torch.sigmoid(out3) - 0.5
            out = (out3 + x) * sim_att
        else:
            out = out3

        return out


class NearestNeighborUpsampler(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int = 2):
        """
        使用 3x3 普通卷积 (分组卷积) + nn.PixelShuffle 来实现与最近邻插值等效的上采样。
        :param in_channels: 输入通道数
        :param scale_factor: 放大倍数
        """
        super().__init__()
        self.scale_factor = scale_factor

        # 1) 升维到 in_channels * (scale_factor^2)
        #    分组卷积(groups=in_channels)，确保每个输入通道单独对应一组卷积核。
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * (scale_factor ** 2),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # 2) 初始化卷积核，使每个通道组都是“取中心像素”，其余位置为 0
        #    卷积核 shape: [out_channels, 1, 3, 3]  (因为是分组卷积，每组只有1个 in_channel)
        with torch.no_grad():
            w = torch.zeros_like(self.conv.weight)  # 与 self.conv.weight 相同形状的 0 张量
            out_channels_total = in_channels * (scale_factor ** 2)
            for c in range(in_channels):
                # 对应第 c 个输入通道的卷积核在输出维度上的范围：
                # [c*(scale_factor^2), (c+1)*(scale_factor^2))
                start_idx = c * (scale_factor ** 2)
                end_idx = (c + 1) * (scale_factor ** 2)
                # 令这些输出通道的卷积核中心为 1，其他位置为 0
                # 即实现“直接复制中心像素”
                for out_c in range(start_idx, end_idx):
                    w[out_c, 0, 1, 1] = 1.0

            self.conv.weight.copy_(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3) 通过 PixelShuffle 将通道重新排列到空间维度
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x

class SPANF(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=48,
                 upscale=4,
                 bias=True,
                 ):
        super(SPANF, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        scale_factor = upscale

        self.conv_near = nn.Conv2d(
            in_channels,
            in_channels * (scale_factor ** 2),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False
        )
        # # self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # # 2) 初始化卷积核，使每个通道组都是“取中心像素”，其余位置为 0
        # #    卷积核 shape: [out_channels, 1, 3, 3]  (因为是分组卷积，每组只有1个 in_channel)
        # with torch.no_grad():
        #     w = torch.zeros_like(self.conv_near.weight)  # 与 self.conv.weight 相同形状的 0 张量
        #     out_channels_total = in_channels * (scale_factor ** 2)
        #     for c in range(in_channels):
        #         # 对应第 c 个输入通道的卷积核在输出维度上的范围：
        #         # [c*(scale_factor^2), (c+1)*(scale_factor^2))
        #         start_idx = c * (scale_factor ** 2)
        #         end_idx = (c + 1) * (scale_factor ** 2)
        #         # 令这些输出通道的卷积核中心为 1，其他位置为 0
        #         # 即实现“直接复制中心像素”
        #         for out_c in range(start_idx, end_idx):
        #             w[out_c, 0, 1, 1] = 1.0

        #     self.conv_near.weight.copy_(w)

        self.block_1 = SPAB1(in_channels, feature_channels, feature_channels, bias=bias)
        self.block_2 = SPAB1(feature_channels, bias=bias)
        self.block_3 = SPAB1(feature_channels, bias=bias)
        self.block_4 = SPAB1(feature_channels, bias=bias)
        self.block_5 = SPAB1(feature_channels, bias=bias)

        
        self.conv_cat = conv_layer(int(feature_channels * 2 + in_channels * upscale**2), feature_channels, kernel_size=1, bias=True)
        self.conv_2 = Conv3XC(feature_channels, in_channels * upscale**2, gain1=2, s=1)

        self.depth_to_space = nn.PixelShuffle(upscale)

        self.cuda()(torch.randn(1, 3, 256, 256).cuda())

    def forward(self, x):

        out_feature = self.conv_near(x)
        out_b1 = self.block_1(x)
        
        out_b2 = self.block_2(out_b1)
        out_b3 = self.block_3(out_b2)
        out_b4 = self.block_4(out_b3)
        out_b5 = self.block_5(out_b4)


        out = self.conv_cat(torch.cat([out_feature, out_b5, out_b1], 1))
        out = self.conv_2(out)
        
        output = self.depth_to_space(out)

        return output


if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    import time
    model = SPANF(3, 3, upscale=4, feature_channels=32).cuda()
    model.eval()
    inputs = (torch.rand(1, 3, 256, 256).cuda(),)
    print(flop_count_table(FlopCountAnalysis(model, inputs)))

    total_time = 0
    input_x = torch.rand(1, 3, 512, 512).cuda()
    for i in range(100):
        torch.cuda.empty_cache()
        sta_time = time.time()
        model(input_x)
        one_time = time.time() - sta_time
        total_time += one_time * 1000
        print("idx: {} one time: {:.4f} ms".format(i, one_time))
    print("Avg time: {:.4f}".format(total_time / 100.))

