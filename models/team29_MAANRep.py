import os
import torch
import torch.nn as nn
import torch.nn.functional as F



class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
    """


    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )


    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x



def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)



class MultiscaleLSK(nn.Module):
    def __init__(self, in_channels, branch_ratio=0.25):
        super().__init__()
        
        gc = int(in_channels * branch_ratio) # channel numbers of a convolution branch
        self.dwconv_LSKA_11 = nn.Sequential( # k_size = 11
                                            nn.Conv2d(gc, gc, kernel_size=(1, 3), stride=(1, 1), padding=(0,(3-1)//2), groups=gc),
                                            nn.Conv2d(gc, gc, kernel_size=(3, 1), stride=(1, 1), padding=((3-1)//2,0), groups=gc),
                                            nn.Conv2d(gc, gc, kernel_size=(1, 5), stride=(1, 1), padding=(0, 4), groups=gc, dilation=2),
                                            nn.Conv2d(gc, gc, kernel_size=(5, 1), stride=(1, 1), padding=(4, 0), groups=gc, dilation=2),
                                        )
        self.dwconv_LSKA_23 = nn.Sequential( # k_size = 23
                                            nn.Conv2d(gc, gc, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=gc),
                                            nn.Conv2d(gc, gc, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=gc),
                                            nn.Conv2d(gc, gc, kernel_size=(1, 7), stride=(1,1), padding=(0, 9), groups=gc, dilation=3),
                                            nn.Conv2d(gc, gc, kernel_size=(7, 1), stride=(1,1), padding=(9, 0), groups=gc, dilation=3),
                                        )
        self.dwconv_LSKA_35 = nn.Sequential( # k_size = 35
                                            nn.Conv2d(gc, gc, kernel_size=(1, 5), stride=(1,1), padding=(0,(5-1)//2), groups=gc),
                                            nn.Conv2d(gc, gc, kernel_size=(5, 1), stride=(1,1), padding=((5-1)//2,0), groups=gc),
                                            nn.Conv2d(gc, gc, kernel_size=(1, 11), stride=(1,1), padding=(0, 15), groups=gc, dilation=3),
                                            nn.Conv2d(gc, gc, kernel_size=(11, 1), stride=(1,1), padding=(15, 0), groups=gc, dilation=3),
                                         )
        self.split_indexes = (in_channels - 3 * gc, gc, gc, gc)
        
    def forward(self, x):
        x_id, x_1, x_2, x_3 = torch.split(x, self.split_indexes, dim=1)
        return torch.cat(
            (x_id, self.dwconv_LSKA_11(x_1), self.dwconv_LSKA_23(x_2), self.dwconv_LSKA_35(x_3)), 
            dim=1,
        )




class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()


        self.out_channels = dim * mlp_ratio


        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")


        self.fc1 = nn.Conv2d(dim, dim * mlp_ratio, 1)
        self.pos = nn.Conv2d(dim * mlp_ratio, dim * mlp_ratio,3, padding=1, groups=dim * mlp_ratio)
        self.fc2 = nn.Conv2d(dim * mlp_ratio, dim, 1)
        self.act = nn.GELU()


    def forward(self, x):


        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        # x = x + self.act(self.pos(x))
        out = self.pos(x)
        x = x + self.act(out)
        x = self.fc2(x)


        return x


class LSKAT(nn.Module):
    def __init__(self, n_feats):
        super().__init__()
       
        #self.norm = LayerNorm(n_feats, data_format='channels_first')
        #self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)
        
        self.conv0 = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 1, 1, 0),
            nn.GELU()
        )
        
        self.att = nn.Sequential(
                                nn.Conv2d(n_feats, n_feats, kernel_size=(1, 7), stride=(1,1), padding=(0,(7-1)//2), groups=n_feats),
                                nn.Conv2d(n_feats, n_feats, kernel_size=(7, 1), stride=(1,1), padding=((7-1)//2,0), groups=n_feats),
                                nn.Conv2d(n_feats, n_feats, kernel_size=(1, 7), stride=(1,1), padding=(0, 9), groups=n_feats, dilation=3),
                                nn.Conv2d(n_feats, n_feats, kernel_size=(7, 1), stride=(1,1), padding=(9, 0), groups=n_feats, dilation=3),
                                nn.Conv2d(n_feats, n_feats, 1, 1, 0)
                            )  



        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)
        
    def forward(self, x):
        x = self.conv0(x)
        x = x*self.att(x) 
        x = self.conv1(x) 
        return x
    
class ConvMod(nn.Module):
    def __init__(self, dim):
        super().__init__()


        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.a = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.GELU(),
            MultiscaleLSK(dim)
        )
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)


    def forward(self, x):


        x = self.norm(x)
        a = self.a(x)
        x = a * self.v(x)
        x = self.proj(x)


        return x



class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=2):
        super().__init__()


        self.attn = ConvMod(dim)
        self.mlp = MLP(dim, mlp_ratio)
        layer_scale_init_value = 1e-6
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x):
        x = x + self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(x)
        x = x + self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(x)
        return x



class Layer(nn.Module):
    def __init__(self, dim, depth, mlp_ratio=2):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim, mlp_ratio) for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 1)


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.conv(x) + x



class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.


    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.


    """


    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)


    def flops(self):
        h, w = self.input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops



class MAAN_rep(nn.Module):
    def __init__(self, in_chans=3, embed_dim=48, depths=(2, 3, 2), mlp_ratio=2, scale=4, img_range=1., upsampler='pixelshuffledirect'):
        super(MAAN_rep, self).__init__()
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = scale
        self.upsampler = upsampler
        self.num_layers = len(depths)


        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = Layer(
                embed_dim, depths[i_layer],  mlp_ratio)
            self.layers.append(layer)
        self.norm = LayerNorm(embed_dim, eps=1e-6,
                              data_format="channels_first")
        self.conv_after_body = LSKAT(embed_dim)
        if self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(scale, embed_dim, in_chans)
        else:
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)


    def forward_features(self, x):
        for layer in self.layers:
            x = layer(x)


        x = self.norm(x)
        return x


    def forward(self, x):
        # if not self.merged_inf:
        #     self.merge_all()


        h, w = x.shape[2:]
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range


        if self.upsampler == 'pixelshuffledirect':
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        else:
            x_first = self.conv_first(x)
            res = self.conv_after_body(
                self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)


        x = x / self.img_range + self.mean


        return x
