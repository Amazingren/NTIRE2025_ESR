import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_pair(value):
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
               out_channels,
               kernel_size,
               stride=1,
               padding=None,
               bias=False):
    
    kernel_size = _make_pair(kernel_size)
    if padding is None:
        padding = (int((kernel_size[0] - 1) / 2), int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias)


class ESA(nn.Module):

    def __init__(self, esa_channels, n_feats, c_out, bias):
        super(ESA, self).__init__()
        f = esa_channels
        self.conv1 = conv_layer(n_feats, f, kernel_size=1, bias=bias)
        self.conv_f = conv_layer(f, f, kernel_size=1, bias=bias)
        self.conv2 = conv_layer(f, f, kernel_size=3, stride=2, padding=0, bias=bias)
        self.conv3 = conv_layer(f, f, kernel_size=3, padding=1, bias=bias)
        self.conv4 = conv_layer(f, c_out, kernel_size=1, bias=bias)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(v_max)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return m


class BasicBlock(nn.Module):
    def __init__(self, num_features, mid_channels, bias):
        super(BasicBlock, self).__init__()
        self.mid_channels = mid_channels
        self.conv1 = conv_layer(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, bias=bias)
        self.conv2 = conv_layer(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, bias=bias)
        self.conv3 = conv_layer(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, bias=bias)
        self.conv4 = conv_layer(self.mid_channels, self.mid_channels, kernel_size=3, padding=1, bias=bias)

        self.act = nn.LeakyReLU(inplace=True)
        
        self.c5 = conv_layer(num_features+self.mid_channels, num_features, kernel_size=1, padding=0, bias=bias)
        self.esa = ESA(16, num_features, num_features, bias=bias)
    
    def forward(self, x):
        x1 = x[:, :self.mid_channels, :, :]
        x2 = x[:, self.mid_channels:, :, :]
        out1 = self.conv1(x1)
        out1 = self.act(out1)
        out1 = self.conv2(out1)
        out1 = self.act(out1)
        
        out2 = self.conv3(out1)
        out2 = self.act(out2)
        out2 = self.conv4(out2)
        out2 = self.act(out2)
        
        out = torch.cat([x2, out1, out2], dim=1)
        out = self.c5(out)
        out = self.esa(out) * out
        
        return out


class RCUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, upscale=4):
        super(RCUNet, self).__init__()
        
        num_features = 48
        mid_channels = 30

        self.conv_first = conv_layer(in_channels, num_features, kernel_size=3, padding=1, bias=False)

        self.b1 = BasicBlock(num_features, mid_channels, bias=False)
        self.b2 = BasicBlock(num_features, mid_channels, bias=False)
        self.b3 = BasicBlock(num_features, mid_channels, bias=False)
        self.b4 = BasicBlock(num_features, mid_channels, bias=False)
        
        self.conv_tail = conv_layer(num_features, num_features, kernel_size=3, padding=1, bias=False)
        self.conv_last = conv_layer(num_features, out_channels*upscale*upscale, kernel_size=3, padding=1, bias=False)

        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out_feat = self.conv_first(x)

        out1 = self.b1(out_feat)
        out2 = self.b2(out1)
        out3 = self.b3(out2) + out2
        out4 = self.b4(out3) + out1
        
        out = self.conv_tail(out4) + out_feat
        out = self.conv_last(out)
        out = self.upsampler(out)

        return out
