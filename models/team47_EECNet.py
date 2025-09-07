import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from einops import rearrange


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = 16    
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        c1 = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv3(c1)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + c1_)
        m = self.sigmoid(c4)
        return x * m


# edge enhanced block
class EEB(nn.Module):

    def __init__(self, dim=32):
        super(EEB, self).__init__()
        #self.eec1 = EEC(n_feats=dim, ratio=2, act_type='prelu', with_idt=True, depth_multiplier=2, with_subBlur=True, with_13=True, deploy=False)
        #self.eec2 = EEC(n_feats=dim, ratio=2, act_type='linear', with_idt=True, depth_multiplier=2, with_subBlur=True, with_13=True, deploy=False)
        self.eec1 = EEC_deploy(n_feats=dim, act_type='prelu')
        self.eec2 = EEC_deploy(n_feats=dim, act_type='linear')
        self.esa = ESA(dim, nn.Conv2d)

    def forward(self, x):
        out = self.eec1(x)
        out = self.eec2(out)
        out = self.esa(out)
        return out 

class EECNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, dim = 32, n_blocks = 8, upscaling_factor=4):
        super(EECNet, self).__init__()
        
        
        self.head = nn.Conv2d(in_nc, dim, 3, 1, 1, bias=True)
        self.blocks = nn.Sequential(*[EEB(dim) for _ in range(n_blocks)])
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0, bias=True)

        self.tail = nn.Conv2d(dim, out_nc * upscaling_factor * upscaling_factor, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(upscaling_factor)
        self.cuda()(torch.randn(1, 3, 256, 256).cuda())

    def forward(self, x):
        fea = self.head(x)
        out = fea
        fea = self.blocks(fea)
        out = self.conv1x1(fea) + out
        out = self.pixel_shuffle(self.tail(out))
        return out


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
        elif self.type == 'conv1x1-subBlur':
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
                self.mask[i, 0, 0, 0] = -0.0625
                self.mask[i, 0, 0, 1] = -0.125
                self.mask[i, 0, 0, 2] = -0.0625
                self.mask[i, 0, 1, 0] = -0.125
                self.mask[i, 0, 1, 1] = 0.75
                self.mask[i, 0, 1, 2] = -0.125
                self.mask[i, 0, 2, 0] = -0.0625
                self.mask[i, 0, 2, 1] = -0.125
                self.mask[i, 0, 2, 2] = -0.0625
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

# -------------------------------------------------
class subEEC(nn.Module):
    def __init__(self, inp_planes, out_planes, depth_multiplier=None, act_type='prelu', with_idt = False, deploy=False, with_13=False, gv=False, with_subBlur=False):
        super(subEEC, self).__init__()
        
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
            self.with_subBlur = with_subBlur
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
            self.conv1x1_subBlur = SeqConv3x3('conv1x1-subBlur', self.inp_planes, self.out_planes, -1)
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
            if self.with_subBlur:
                y += self.conv1x1_subBlur(x)

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
        K6, B6 = self.conv1x1_subBlur.rep_params()
        if self.gv:
            RK, RB = (K0+K2+K3+K4), (B0+B2+B3+B4) 
        else:
            RK, RB = (K0+K2+K3+K4+K5), (B0+B2+B3+B4+B5) 
            if self.with_13:
                RK, RB = RK + K1, RB + B1
            if self.with_subBlur:
                RK, RB = RK + K6, RB + B6
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
        self.__delattr__('conv1x1_subBlur')

        print('subEEC switch to deploy mode')

        return RK, RB  


class EEC(nn.Module):
    def __init__(self, n_feats,ratio=2, act_type='prelu',with_idt=True, depth_multiplier=2, with_subBlur=True, with_13=True, deploy=False):
        super(EEC, self).__init__()
        self.n_feats = n_feats
        self.deploy = deploy
        self.act_type = act_type

        self.expand_conv = nn.Conv2d(n_feats, ratio*n_feats, kernel_size=1, stride=1, padding=0, bias=False)
        self.fea_conv = subEEC(inp_planes=ratio*n_feats, out_planes=ratio*n_feats, depth_multiplier=depth_multiplier, act_type='linear',with_idt=with_idt, with_13=with_13, with_subBlur=with_subBlur)
        self.reduce_conv = nn.Conv2d(ratio*n_feats, n_feats, kernel_size=1, stride=1, padding=0)

        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.n_feats)
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
            out = self.rep_conv(x)
        else:
            out = self.expand_conv(x)
            out = self.fea_conv(out)
            out = self.reduce_conv(out)
            out = out + x
        
        if self.act_type != 'linear':
            out = self.act(out)

        return out

    def switch_to_deploy(self):
        if self.deploy:
            return
        self.deploy = True

        k0 = self.expand_conv.weight.data
        #b0 = self.expand_conv.bias.data
        b0 = torch.tensor(0.0)

        k1,b1 = self.fea_conv.switch_to_deploy()

        k2 = self.reduce_conv.weight.data
        b2 = self.reduce_conv.bias.data

        mid_feats, n_feats = k0.shape[:2]
    
        # first step: merge the first 1x1 convolution and the next 3x3 convolution
        merge_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
        merge_b0b1 = (b0.view(1, -1, 1, 1) * torch.ones(1, mid_feats, 3, 3)).cuda()
        merge_b0b1 = F.conv2d(input=merge_b0b1, weight=k1, bias=b1)

        # second step: merge the remain 1x1 convolution
        merge_k0k1k2 = F.conv2d(input=merge_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
        merge_b0b1b2 = F.conv2d(input=merge_b0b1, weight=k2, bias=b2).view(-1)

        # last step: remove the global identity
        for i in range(n_feats):
            merge_k0k1k2[i, i, 1, 1] += 1.0

        self.rep_conv.weight.data = merge_k0k1k2.float()
        self.rep_conv.bias.data = merge_b0b1b2.float()
            
        self.__delattr__('expand_conv')
        self.__delattr__('fea_conv')
        self.__delattr__('reduce_conv')
        print('EEC Switch to deploy mode!')


class EEC_deploy(nn.Module):
    def __init__(self, n_feats,act_type='prelu'):
        super(EEC_deploy, self).__init__()

        self.act_type = act_type

        self.rep_conv = nn.Conv2d(n_feats, n_feats, 3, 1, 1)
               
        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=n_feats)
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
        y = self.rep_conv(x)
        if self.act_type != 'linear':
            y = self.act(y)

        return y