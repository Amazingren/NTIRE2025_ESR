import torch
import torch.nn as nn
import torch.nn.functional as F

# from copy import deepcopy
from collections import OrderedDict
# from einops.layers.torch import Rearrange, Reduce
# from basicsr.utils.registry import ARCH_REGISTRY

def pad_tensor(t, pattern):
    pattern = pattern.view(1, -1, 1, 1)
    t = F.pad(t, (1, 1, 1, 1), 'constant', 0)
    t[:, :, 0:1, :] = pattern
    t[:, :, -1:, :] = pattern
    t[:, :, :, 0:1] = pattern
    t[:, :, :, -1:] = pattern

    return t

# class ASR(nn.Module):
#     def __init__(self, n_feat, ratio=2):
#         super().__init__()
#         self.n_feat = n_feat
#         self.tensor = nn.Parameter(
#             0.1*torch.ones((1, n_feat, 1, 1)),
#             requires_grad=True
#         )
#         self.se = nn.Sequential(
#             Reduce('b c 1 1 -> b c', 'mean'),
#             nn.Linear(n_feat, n_feat//4, bias = False),
#             nn.SiLU(),
#             nn.Linear(n_feat//4, n_feat, bias = False),
#             nn.Sigmoid(),
#             Rearrange('b c -> b c 1 1')
#         )
#         self.init_weights()

#     def init_weights(self): 
#         # to make sure the inital [0.5,0.5,...,0.5]
#         self.se[1].weight.data.fill_(1)    
#         self.se[3].weight.data.fill_(1)
        
#     def forward(self, x):
#         attn = self.se(self.tensor)
#         x = attn*x 
#         return x

class RepMBConvSE(nn.Module):
    '''Reparameterized MobileNet conv with modified squeeze and excitation'''
    def __init__(self, n_feat, ratio=2):
        super().__init__()
        # i_feat = n_feat*ratio
        # self.expand_conv = nn.Conv2d(n_feat,i_feat,1,1,0)
        # self.fea_conv = nn.Conv2d(i_feat,i_feat,3,1,0)
        # self.reduce_conv = nn.Conv2d(i_feat,n_feat,1,1,0)
        # self.se = ASR(i_feat)

        self.conv = nn.Conv2d(n_feat,n_feat,3,1,1)

    def forward(self, x):
        if not hasattr(self, 'expand_conv'):
            return self.conv(x)
        
        out = self.expand_conv(x)
        out_identity = out
        
        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)
        out = self.fea_conv(out) 
        out = self.se(out) + out_identity
        out = self.reduce_conv(out)
        out = out + x

        return out

    def switch_to_deploy(self):
        n_feat, _, _, _ = self.reduce_conv.weight.data.shape
        self.conv = nn.Conv2d(n_feat,n_feat,3,1,1)

        k0 = self.expand_conv.weight.data
        b0 = self.expand_conv.bias.data

        k1 = self.fea_conv.weight.data
        b1 = self.fea_conv.bias.data

        k2 = self.reduce_conv.weight.data
        b2 = self.reduce_conv.bias.data

        # first step: remove the ASR
        a = self.se.se(self.se.tensor)

        k1 = k1*(a.permute(1,0,2,3))
        b1 = b1*(a.view(-1))

        # second step: remove the middle identity
        for i in range(2*n_feat):
            k1[i,i,1,1] += 1.0 

        # third step: merge the first 1x1 convolution and the next 3x3 convolution
        merge_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
        merge_b0b1 = b0.view(1, -1, 1, 1) * torch.ones(1, 2*n_feat, 3, 3) #.cuda()
        merge_b0b1 = F.conv2d(input=merge_b0b1, weight=k1, bias=b1)       

        # third step: merge the remain 1x1 convolution
        merge_k0k1k2 = F.conv2d(input=merge_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
        merge_b0b1b2 = F.conv2d(input=merge_b0b1, weight=k2, bias=b2).view(-1)

        # last step: remove the global identity
        for i in range(n_feat):
            merge_k0k1k2[i, i, 1, 1] += 1.0

        self.conv.weight.data = merge_k0k1k2.float()
        self.conv.bias.data = merge_b0b1b2.float()   

        for para in self.parameters():
            para.detach_()

        self.__delattr__('expand_conv')
        self.__delattr__('fea_conv')
        self.__delattr__('reduce_conv')
        self.__delattr__('se')

class RepMBConv(nn.Module):
    '''Reparameterized MobileNet conv without SE'''
    def __init__(self, c_in, c_out, ratio=2):
        super().__init__()
        # self.identity_conv = nn.Conv2d(c_in, c_out, 1, 1,0)
        # self.expand_conv = nn.Conv2d(c_in, c_in * ratio, 1, 1, 0)
        # self.fea_conv = nn.Conv2d(c_in * ratio, c_out * ratio, 3, 1, 0)
        # self.reduce_conv = nn.Conv2d(c_out * ratio, c_out, 1, 1, 0)

        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)


    def forward(self, x):
        if not hasattr(self, 'expand_conv'):
            return self.conv(x)
        
        out = self.expand_conv(x)

        # padding for reparameterizing
        b0 = self.expand_conv.bias
        out = pad_tensor(out, b0)

        out = self.fea_conv(out)
        out = self.reduce_conv(out)
        out = out + self.identity_conv(x)

        return out


    def switch_to_deploy(self):
        c_out, c_in, _, _ = self.identity_conv.weight.data.shape
        self.conv = nn.Conv2d(c_in, c_out, 3, 1, 1)

        w1 = self.expand_conv.weight.data.clone().detach()
        b1 = self.expand_conv.bias.data.clone().detach()
        w2 = self.fea_conv.weight.data.clone().detach()
        b2 = self.fea_conv.bias.data.clone().detach()
        w3 = self.reduce_conv.weight.data.clone().detach()
        b3 = self.reduce_conv.bias.data.clone().detach()

        w = F.conv2d(w1.flip(2, 3).permute(1, 0, 2, 3), w2, padding=2, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = (w2 * b1.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b2

        self.weight_concat = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), w3, padding=0, stride=1).flip(2, 3).permute(1, 0, 2, 3)
        self.bias_concat = (w3 * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + b3

        sk_w = self.identity_conv.weight.data.clone().detach() # sk_w: short-cut conv weight
        sk_b = self.identity_conv.bias.data.clone().detach()
        target_kernel_size = 3

        H_pixels_to_pad = (target_kernel_size - 1) // 2
        W_pixels_to_pad = (target_kernel_size - 1) // 2
        sk_w = F.pad(sk_w, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])

        self.weight_concat = self.weight_concat + sk_w
        self.bias_concat = self.bias_concat + sk_b

        self.conv.weight.data = self.weight_concat
        self.conv.bias.data = self.bias_concat

        for para in self.parameters():
            para.detach_()

        self.__delattr__('expand_conv')
        self.__delattr__('fea_conv')
        self.__delattr__('reduce_conv')
        self.__delattr__('identity_conv')

class RepPixelAttn(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 ratio=2):
        super().__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.conv_1 = RepMBConv(in_channels, mid_channels, ratio=ratio)
        self.conv_2 = RepMBConv(mid_channels, mid_channels, ratio=ratio)
        self.conv_3 = RepMBConv(mid_channels, out_channels, ratio=ratio)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = (self.conv_1(x))
        out1_act = self.act1(out1)

        out2 = (self.conv_2(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.conv_3(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att

class RepBlock(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.feature_extractor = RepMBConvSE(n_feat, ratio=2)
        self.pixel_attention = RepPixelAttn(n_feat, ratio=2)

    def forward(self, x):
        x = self.feature_extractor(x)
        out, out1, sim_att = self.pixel_attention(x) 
        return out, out1, sim_att


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

# comment this line when converting the model to deployment mode, and then run command: python nanosr_train_arch.py
# @ARCH_REGISTRY.register()
class NanoSR_inference(nn.Module):
    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=48,
                 upscale=4,
                 img_range=255.,
                 rgb_mean=(0.4488, 0.4371, 0.4040)
                 ):
        super().__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.conv_1 = RepMBConv(in_channels, feature_channels, ratio=2) # shallow feature extractor
        self.block_1 = RepBlock(feature_channels) # deep feature extractor
        self.block_2 = RepBlock(feature_channels)
        self.block_3 = RepBlock(feature_channels)
        self.block_4 = RepBlock(feature_channels)
        self.block_5 = RepBlock(feature_channels)
        self.block_6 = RepBlock(feature_channels)
        self.conv_2 = RepMBConv(feature_channels, feature_channels, ratio=2)

        self.conv_cat = conv_layer(feature_channels * 4, feature_channels, kernel_size=1, bias=True) # fc layer downscale feature channels
        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale) # conv3×3 upscale feature channels + pixel shuffle

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
        out = self.conv_cat(torch.cat([out_feature, out_b6, out_b1, out_b5_2], 1))
        output = self.upsampler(out)

        return output

