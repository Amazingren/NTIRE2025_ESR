import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class AddOp(nn.Module):
    def forward(self, x1, x2):
        return x1 + x2

class AnchorOp(nn.Module):

    def __init__(self, scaling_factor, in_channels=3, init_weights=True, freeze_weights=True, kernel_size=1, **kwargs):

        super().__init__()

        self.net = nn.Conv2d(in_channels=in_channels,
                             out_channels=(in_channels * scaling_factor**2),
                             kernel_size=kernel_size,
                             **kwargs)

        if init_weights:
            num_channels_per_group = in_channels // self.net.groups
            weight = torch.zeros(in_channels * scaling_factor**2, num_channels_per_group, kernel_size, kernel_size)

            bias = torch.zeros(weight.shape[0])
            for ii in range(in_channels):
                weight[ii * scaling_factor**2: (ii + 1) * scaling_factor**2, ii % num_channels_per_group,
                kernel_size // 2, kernel_size // 2] = 1.

            new_state_dict = OrderedDict({'weight': weight, 'bias': bias})
            self.net.load_state_dict(new_state_dict)

            if freeze_weights:
                for param in self.net.parameters():
                    param.requires_grad = False

    def forward(self, input):
        return self.net(input)

def mean_channels(F):
    assert (F.dim() == 4) 
    return F.sum(dim=(2, 3), keepdim=True) / (F.size(2) * F.size(3))  

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    return ((F - F_mean).pow(2).sum(dim=(2, 3), keepdim=True) / (F.size(2) * F.size(3))).sqrt()

class CCALayer(nn.Module):

    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()
        self.contrast = stdv_channels  
        self.avg_pool = nn.AdaptiveAvgPool2d(1) 
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=True), 
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=True), 
            nn.Sigmoid() 
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x) 
        y = self.conv_du(y) 
        return x * y  

class ESA(nn.Module):

    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4  
        self.conv1 = conv(n_feats, f, kernel_size=1)  
        self.conv_f = conv(f, f, kernel_size=1)  
        self.conv_max = conv(f, f, kernel_size=3, padding=1)  
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0) 
        self.conv3 = conv(f, f, kernel_size=3, padding=1) 
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)  
        self.conv4 = conv(f, n_feats, kernel_size=1)  
        self.sigmoid = nn.Sigmoid()  
        self.relu = nn.ReLU(inplace=True) 

    def forward(self, x):
        c1_ = self.conv1(x)  
        c1 = self.conv2(c1_)  
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  
        v_range = self.relu(self.conv_max(v_max)) 
        c3 = self.relu(self.conv3(v_range)) 
        c3 = self.conv3_(c3) 
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)  
        cf = self.conv_f(c1_)  
        c4 = self.conv4(c3 + cf)  
        m = self.sigmoid(c4)  

        return x * m  

class block(nn.Module):

    def __init__(self, channels, reduction=4, bias=False):
        super(block, self).__init__()
        
        self.c1_r = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//2, kernel_size=(3, 1), padding=(1, 0), bias=bias),
            nn.Conv2d(in_channels=channels//2, out_channels=channels, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        )

        self.c2_r = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//2, kernel_size=(3, 1), padding=(1, 0), bias=bias),
            nn.Conv2d(in_channels=channels//2, out_channels=channels, kernel_size=(1, 3), padding=(0, 1), bias=bias)
        )

        self.esa = ESA(channels, nn.Conv2d)
        self.cca = CCALayer(channels)

        self.conv_merge = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x):

        out1 = self.c1_r(x)
        out1_act = torch.relu(out1)
        out2 = self.c2_r(out1_act)

        esa_out = self.esa(out2)
        cca_out = self.cca(out2)

        merged_out = torch.cat([esa_out, cca_out], dim=1)

        out = self.conv_merge(merged_out)

        return x + out

class ESA_CCA(nn.Module):

    def __init__(self,
                 scaling_factor=4,
                 num_channels=48,
                 num_intermediate_layers=9,
                 use_ito_connection=True,
                 in_channels=3,
                 out_channels=3):

        super(ESA_CCA, self).__init__()

        self.out_channels = out_channels
        self._use_ito_connection = use_ito_connection
        self._has_integer_scaling_factor = float(scaling_factor).is_integer()

        if self._has_integer_scaling_factor:
            self.scaling_factor = int(scaling_factor)

        else:
            raise NotImplementedError(f'1.5 is the only supported non-integer scaling factor. '
                                      f'Received {scaling_factor}.')

        intermediate_layers = []
        for _ in range(num_intermediate_layers):
            intermediate_layers.extend([
                block(num_channels),
            ])

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=(3, 3), padding=1),
            nn.Hardtanh(min_val=0., max_val=1.),
            *intermediate_layers,
        )

        if scaling_factor == 1.5:
            cl_in_channels = num_channels * (2 ** 2)
            cl_out_channels = out_channels * (3 ** 2)
            cl_kernel_size = (1, 1)
            cl_padding = 0
        else:
            cl_in_channels = num_channels
            cl_out_channels = out_channels * (self.scaling_factor ** 2)
            cl_kernel_size = (3, 3)
            cl_padding = 1

        self.conv_last = nn.Conv2d(in_channels=cl_in_channels, out_channels=cl_out_channels, kernel_size=cl_kernel_size, padding=cl_padding)

        if use_ito_connection:
            self.add_op = AddOp()
            self.anchor = AnchorOp(scaling_factor=self.scaling_factor, freeze_weights=False)

        self.depth_to_space = nn.PixelShuffle(self.scaling_factor)

        self.clip_output = nn.Hardtanh(min_val=0., max_val=1.)

    def forward(self, input):
        x = self.cnn(input)

        if not self._has_integer_scaling_factor:
            x = self.space_to_depth(x)

        if self._use_ito_connection:
            residual = self.conv_last(x)
            input_convolved = self.anchor(input)
            x = self.add_op(input_convolved, residual)
        else:
            x = self.conv_last(x)

        x = self.clip_output(x)

        return self.depth_to_space(x)
