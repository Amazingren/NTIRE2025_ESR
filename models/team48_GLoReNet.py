import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import imp
import torch.nn as nn
import torch.nn.functional as F

class Conv3X3(nn.Module):
    def __init__(self, inp_planes, out_planes, act_type='prelu', bias=True, dtype=torch.float32):
        super(Conv3X3, self).__init__()

        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        self.conv3x3 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1, bias=bias, dtype=dtype)
        self.act  = None

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        elif self.act_type == 'silu':
            self.act = nn.SiLU(inplace=True)
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        y = self.conv3x3(x)
        if self.act_type != 'linear':
            y = self.act(y)
        return y
    
class ESA(nn.Module):
    def __init__(self, n_feats, conv, bias = True, dtype=torch.float32):
        super(ESA, self).__init__()
        f = 16
        self.conv1 = conv(n_feats, f, kernel_size=1, bias=bias, dtype=dtype)
        self.conv_f = conv(f, f, kernel_size=1, bias=bias, dtype=dtype)
        self.conv_max = conv(f, f, kernel_size=3, padding=1, bias=bias, dtype=dtype)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0, bias=bias, dtype=dtype)
        self.conv3 = conv(f, f, kernel_size=3, padding=1, bias=bias, dtype=dtype)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1, bias=bias, dtype=dtype)
        self.conv4 = conv(f, n_feats, kernel_size=1, bias=bias, dtype=dtype)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False) 
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3+cf)
        m = self.sigmoid(c4)

        return x * m
    
class SPABRep(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels=None,
                 out_channels=None,
                 bias=False, 
                 dtype=torch.float32):
        super(SPABRep, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = Conv3X3(inp_planes=in_channels, act_type='linear', out_planes=mid_channels, bias=bias, dtype=dtype)
        self.c2_r = Conv3X3(inp_planes=in_channels, act_type='linear', out_planes=mid_channels, bias=bias, dtype=dtype)
        self.c3_r = Conv3X3(inp_planes=in_channels, act_type='linear', out_planes=mid_channels, bias=bias, dtype=dtype)
        self.act1 = torch.nn.SiLU(inplace=True)

    def forward(self, x):
        out1 = (self.c1_r(x))
        out1_act = self.act1(out1)

        out2 = (self.c2_r(out1_act))
        out2_act = self.act1(out2)

        out3 = (self.c3_r(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out, out1, sim_att
    
class CASRv016_deploy(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, down_scale, colors, block_type=None, bias=True, dtype=torch.float16):
        super(CASRv016_deploy, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.bias = bias
        self.repBlk = block_type
        self.dtype = dtype

        backbone = []
        self.head = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type, bias=self.bias, dtype=dtype)
        
        for i in range(self.module_nums//2):
            backbone += [nn.Sequential(SPABRep(in_channels=self.channel_nums, out_channels=self.channel_nums, bias=self.bias, dtype=dtype))]
        
        self.backbone = nn.Sequential(*backbone)

        self.transition = nn.Sequential(nn.Conv2d(self.channel_nums*4, self.channel_nums, 1, padding=0, bias=True, dtype=dtype),
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=act_type, 
                                                bias=self.bias, dtype=dtype)
                                        )                                 

        self.input_conv2 = Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear', 
                                   bias=self.bias, dtype=dtype)
        
        self.upsampler1 = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        if self.dtype == torch.float16:
            x = x.half()
        y0 = self.head(x)
        
        y1 = self.backbone[0](y0)
        y2 = self.backbone[1](y1[0])
        y3 = self.backbone[2](y2[0])
        y4 = self.backbone[3](y3[0])
        
        y = self.transition(torch.cat([y0,y1[0],y4[0],y4[1]], 1))
        
        y = self.input_conv2(y)
        
        y = self.upsampler1(y)
        return y
     
class CASRv016_hybrid_deploy(nn.Module):
    def __init__(self, module_nums, channel_nums, act_type, scale, down_scale, colors, block_type=None, bias=True, dtype=torch.float16):
        super(CASRv016_hybrid_deploy, self).__init__()
        self.module_nums = module_nums
        self.channel_nums = channel_nums
        self.scale = scale
        self.colors = colors
        self.act_type = act_type
        self.backbone = None
        self.upsampler = None
        self.bias = bias
        self.repBlk = block_type
        self.dtype = dtype

        backbone_span = []
        backbone_esa = []
        self.head = Conv3X3(inp_planes=self.colors, out_planes=self.channel_nums, act_type=self.act_type, bias=self.bias, dtype=dtype)
        
        for i in range(2):
            backbone_span += [nn.Sequential(SPABRep(in_channels=self.channel_nums, out_channels=self.channel_nums, bias=self.bias, dtype=dtype))]
        
        self.backbone_span = nn.Sequential(*backbone_span)

        self.transition = nn.Sequential(nn.Conv2d(self.channel_nums*4, self.channel_nums, 1, padding=0, bias=True, dtype=dtype),
                                        Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=act_type, bias=self.bias, dtype=dtype)
                                        )                                 

        self.input_conv1 = Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=act_type, bias=self.bias, dtype=dtype)

        for i in range(2):
            backbone_esa += [nn.Sequential(Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=act_type, bias=self.bias, dtype=dtype),
                                       Conv3X3(inp_planes=self.channel_nums, out_planes=self.channel_nums, act_type=act_type, bias=self.bias, dtype=dtype),
                                       ESA(self.channel_nums, nn.Conv2d, bias = self.bias, dtype=dtype))]

        self.backbone_esa = nn.Sequential(*backbone_esa)

        self.input_conv2 = Conv3X3(inp_planes=self.channel_nums, out_planes=self.colors*self.scale*self.scale, act_type='linear', bias=self.bias, dtype=dtype)

        self.upsampler1 = nn.PixelShuffle(self.scale)
    
    def forward(self, x):
        if self.dtype == torch.float16:
            x = x.half()
        y0 = self.head(x)
        
        y1 = self.backbone_span[0](y0)
        y2 = self.backbone_span[1](y1[0])
    
        y = self.transition(torch.cat([y0,y1[0],y2[0],y2[1]], 1))

        y = self.input_conv1(y+y0)

        y3 = self.backbone_esa[0](y)
        y4 = self.backbone_esa[1](y3)

        y = self.input_conv2(y4+y)

        y = self.upsampler1(y)

        return y
       
if __name__ == "__main__":
    x = torch.rand(1,3,384,384).cuda()
            
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    import time
    model = CASRv016_hybrid_deploy(module_nums=8, channel_nums=48, act_type='relu', scale=3, down_scale=1, colors=3).cuda().eval()
    
    inputs = (torch.rand(1, 3, 256, 256).cuda(),)
    
    print('Deploy model: ')
    print(flop_count_table(FlopCountAnalysis(model, inputs)))
    
