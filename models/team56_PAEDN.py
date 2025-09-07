# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class Cell(nn.Module):
    def __init__(self, n_feats=48, dynamic = True, deploy = False, L= None, with_13=False):
        super(Cell, self).__init__()
        
        self.conv0 = nn.Conv2d(n_feats,n_feats,1,1,0)
        self.conv2 = EDBB_deploy(n_feats,n_feats)
        self.conv3 = EDBB_deploy(n_feats,n_feats)

        self.fuse = nn.Conv2d(int(n_feats*(6/4)), n_feats, 1, 1, 0)

        self.branch = nn.ModuleList([nn.Conv2d(n_feats, n_feats//2, 1, 1, 0) for _ in range(3)])

    def forward(self, x):

        out1 = self.conv0(x)

        out2 = self.conv2(out1)
        attn_1 = torch.sigmoid(out2) - 0.5
        out2_attn = out2 * attn_1

        out3 = self.conv3(out2)
        attn_2 = torch.sigmoid(out3) - 0.5
        out3_attn = out3 * attn_2

        out = self.fuse(torch.cat([self.branch[0](out1), self.branch[1](out2_attn), self.branch[2](out3_attn)], dim=1))

        #out = F.pad(out,(1,1,1,1))
        out += x

        return out

class PAEDN(nn.Module):
    def __init__(self, scale=4, in_channels=3, n_feats=48, out_channels=3):
        super(PAEDN, self).__init__()
        self.head = nn.Conv2d(in_channels, n_feats, 3, 1, 1)
        # body cells
        self.cells = nn.ModuleList([Cell(n_feats) for _ in range(4)])
        
        # fusion
        self.local_fuse = nn.ModuleList([nn.Conv2d(n_feats*2, n_feats, 1, 1, 0) for _ in range(3)])

        self.tail = nn.Sequential(
            EDBB_deploy(n_feats, out_channels*(scale**2)),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        # head
        out0 = self.head(x)

        # body cells
        out1 = self.cells[0](out0)
        out2 = self.cells[1](out1)
        out2_fuse = self.local_fuse[0](torch.cat([out1, out2], dim=1))
        out3 = self.cells[2](out2_fuse)
        out3_fuse = self.local_fuse[1](torch.cat([out2, out3], dim=1))
        out4 = self.cells[3](out3_fuse)
        out4_fuse = self.local_fuse[2](torch.cat([out2, out4], dim=1))

        out = out4_fuse + out0      

        # tail
        out = self.tail(out)

        return out.clamp(0,1)
   

# -------------------------------------------------
#This part code based on DBB(https://github.com/DingXiaoH/DiverseBranchBlock) and ECB(https://github.com/xindongzhang/ECBSR)
def multiscale(kernel, target_kernel_size):
    H_pixels_to_pad = (target_kernel_size - kernel.size(2)) // 2
    W_pixels_to_pad = (target_kernel_size - kernel.size(3)) // 2
    return F.pad(kernel, [H_pixels_to_pad, H_pixels_to_pad, W_pixels_to_pad, W_pixels_to_pad])


class SeqConv3x3(nn.Module):
    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier):
        super(SeqConv3x3, self).__init__()

        self.type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias
            
        elif self.type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias

            # init scale & bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            # bias = 0.0
            # bias = [bias for c in range(self.out_planes)]
            # bias = torch.FloatTensor(bias)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes,))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('the type of seqconv is not supported!')

    def forward(self, x):
        if self.type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1
    
    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.type == 'conv1x1-conv3x3':
            # re-param conv kernel
            RK = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=self.k1).view(-1,) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            RK = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            RB = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            RB = F.conv2d(input=RB, weight=k1).view(-1,) + b1
        return RK, RB


class EDBB(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=None, act_type='prelu', with_idt = False, deploy=False, with_13=False, gv=False):
        super(EDBB, self).__init__()
        
        self.deploy = deploy
        self.act_type = act_type
        
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        self.gv = gv          

        if depth_multiplier is None:
            self.depth_multiplier = 1.0
        else: 
            self.depth_multiplier = depth_multiplier   # For mobilenet, it is better to have 2X internal channels
        
        if deploy:
            self.rep_conv = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes, kernel_size=3, stride=1,
                                      padding=1, bias=True)
        else: 
            self.with_13 = with_13
            if with_idt and (self.inp_planes == self.out_planes):
                self.with_idt = True
            else:
                self.with_idt = False

            self.rep_conv = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
            self.conv1x1 = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
            self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes, -1)
            self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes, -1)
            self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes, -1)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.deploy:
            y = self.rep_conv(x)
        elif self.gv:
            y = self.rep_conv(x)     + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x) + x 
        else:
            y = self.rep_conv(x)     + \
                self.conv1x1(x)     + \
                self.conv1x1_sbx(x) + \
                self.conv1x1_sby(x) + \
                self.conv1x1_lpl(x)            
                #self.conv1x1_3x3(x) + \
            if self.with_idt:
                y += x
            if self.with_13:
                y += self.conv1x1_3x3(x)

        if self.act_type != 'linear':
            y = self.act(y)
        return y
    
    def switch_to_gv(self):
        if self.gv:
            return
        self.gv = True
        
        K0, B0 = self.rep_conv.weight, self.rep_conv.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K5, B5 = multiscale(self.conv1x1.weight,3), self.conv1x1.bias
        RK, RB = (K0+K5), (B0+B5) 
        if self.with_13:
            RK, RB = RK + K1, RB + B1

        self.rep_conv.weight.data = RK
        self.rep_conv.bias.data = RB
        
        for para in self.parameters():
            para.detach_()
      
    
    def switch_to_deploy(self):

        if self.deploy:
            return
        self.deploy = True
        
        K0, B0 = self.rep_conv.weight, self.rep_conv.bias
        K1, B1 = self.conv1x1_3x3.rep_params()
        K2, B2 = self.conv1x1_sbx.rep_params()
        K3, B3 = self.conv1x1_sby.rep_params()
        K4, B4 = self.conv1x1_lpl.rep_params()
        K5, B5 = multiscale(self.conv1x1.weight,3), self.conv1x1.bias
        if self.gv:
            RK, RB = (K0+K2+K3+K4), (B0+B2+B3+B4) 
        else:
            RK, RB = (K0+K2+K3+K4+K5), (B0+B2+B3+B4+B5) 
            if self.with_13:
                RK, RB = RK + K1, RB + B1
        if self.with_idt:
            device = RK.get_device()
            if device < 0:
                device = None
            K_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                K_idt[i, i, 1, 1] = 1.0
            B_idt = 0.0
            RK, RB = RK + K_idt, RB + B_idt        
            

        self.rep_conv = nn.Conv2d(in_channels=self.inp_planes, out_channels=self.out_planes, kernel_size=3, stride=1,
                                      padding=1, bias=True)
        self.rep_conv.weight.data = RK
        self.rep_conv.bias.data = RB
        
        for para in self.parameters():
            para.detach_()
            
        #self.__delattr__('conv3x3')
        self.__delattr__('conv1x1_3x3')
        self.__delattr__('conv1x1')
        self.__delattr__('conv1x1_sbx')
        self.__delattr__('conv1x1_sby')
        self.__delattr__('conv1x1_lpl')


class EDBB_deploy(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(EDBB_deploy, self).__init__()
               
        self.rep_conv = nn.Conv2d(in_channels=inp_planes, out_channels=out_planes, kernel_size=3, stride=1,
                                      padding=1, bias=True)
 
        self.act = nn.PReLU(num_parameters=out_planes)

    def forward(self, x):
        y = self.rep_conv(x)
        y = self.act(y)
        
        return y
# -------------------------------------------------

# 25-03-07 11:15:11.319 :       Max Memory : 625.452 [M]
# 25-03-07 11:15:11.319 : ------> Average runtime of (valid) is : 16.582070 milliseconds
# 25-03-07 11:15:11.319 : ------> Average PSNR of (valid) is : 20.622159 dB
# 25-03-07 11:15:11.329 :     #Activations : 91.2261 [M]
# 25-03-07 11:15:11.329 :          #Conv2d : 37
# Unsupported operator aten::prelu encountered 12 time(s)
# Unsupported operator aten::sigmoid encountered 4 time(s)
# Unsupported operator aten::sub encountered 4 time(s)
# Unsupported operator aten::mul encountered 4 time(s)
# Unsupported operator aten::add encountered 5 time(s)
# Unsupported operator aten::pixel_shuffle encountered 1 time(s)
# The following submodules of the model were never called during the trace of the graph. They may be unused, or they were accessed by direct calls to .forward() or via other python methods. In the latter case they will have zeros for statistics, though their statistics will still contribute to their parent calling module.
# cells.0.att, cells.0.att.conv1, cells.0.att.conv2, cells.0.att.conv3, cells.0.att.conv3_, cells.0.att.conv4, cells.0.att.conv_f, cells.0.att.conv_max, cells.0.att.relu, cells.0.att.sigmoid, cells.1.att, cells.1.att.conv1, cells.1.att.conv2, cells.1.att.conv3, cells.1.att.conv3_, cells.1.att.conv4, cells.1.att.conv_f, cells.1.att.conv_max, cells.1.att.relu, cells.1.att.sigmoid, cells.2.att, cells.2.att.conv1, cells.2.att.conv2, cells.2.att.conv3, cells.2.att.conv3_, cells.2.att.conv4, cells.2.att.conv_f, cells.2.att.conv_max, cells.2.att.relu, cells.2.att.sigmoid, cells.3.att, cells.3.att.conv1, cells.3.att.conv2, cells.3.att.conv3, cells.3.att.conv3_, cells.3.att.conv4, cells.3.att.conv_f, cells.3.att.conv_max, cells.3.att.relu, cells.3.att.sigmoid
# 25-03-07 11:15:11.388 :            FLOPs : 16.2414 [G]
# 25-03-07 11:15:11.388 :          #Params : 0.2762 [M]