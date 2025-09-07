import torch
import torch.nn as nn
import torch.nn.functional as F
class FRAM(nn.Module):
    def __init__(self, dim, ratio=3):
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // ratio

        self.proj = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        self.dwconv = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1, groups=self.chunk_dim, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        
        self.conv5_proj=nn.Conv2d(dim, dim,1,padding=0,bias=False)
        self.conv6_proj=nn.Conv2d(dim, dim,3,padding=1,bias=False)

        self.conv_fuse1=nn.Conv2d(dim,dim,3,padding=1,bias=False)
        self.conv_fuse2=nn.Conv2d(dim,dim,3,padding=1,bias=False)
        self.initflag=False
        self.act = nn.GELU()

    #串行卷积
    def transIII_conv_sequential(self,conv1, conv2):
        weight=F.conv2d(conv2.weight.data,conv1.weight.data.permute(1,0,2,3))
        return weight
    
    #并行卷积
    def transII_conv_branch(self,conv1, conv2):
        weight=conv1.weight.data+conv2.weight.data
        return weight 

    def forward(self, x):
        h, w = x.size()[-2:]
        # x=self.proj(x) + self.conv6_proj(self.conv5_proj(x))
        if  self.initflag==False:
            self.conv_fuse1.weight.data=self.transIII_conv_sequential(self.conv5_proj,self.conv6_proj)
            self.conv_fuse2.weight.data=self.transII_conv_branch(self.proj,self.conv_fuse1)
            self.initflag=True
            del self.proj
            del self.conv5_proj
            del self.conv6_proj
            del self.conv_fuse1

        x=self.conv_fuse2(x)

        x0, x1 = x.split([self.chunk_dim, self.dim-self.chunk_dim], dim=1)

        x2 = F.adaptive_max_pool2d(x0, (h//16, w//16))
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
        x2 = self.act(x2) * x0

        x = torch.cat([x1, x2], dim=1)
        x = self.act(x)
        x = self.out(x)
        return x

class CFM(nn.Module):
    def __init__(self, dim, ffn_scale, use_se=False, idx=None):
        super().__init__()
        self.use_se = use_se
        hidden_dim = int(dim*ffn_scale)

        self.conv1 = nn.Conv2d(dim, hidden_dim, 3, 1, 1, bias=False)
        
        if idx <6:
            self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        else:
            self.conv5 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
            
        self.act = nn.GELU()
        self.idx = idx

    def forward(self, x):
        x = self.act(self.conv1(x))
        if self.idx <6:
            x = self.conv2(x)
        else:
            x = self.conv5(x)
        return x

class FRABlock(nn.Module):
    def __init__(self, dim, ffn_scale, use_se=False, idx=None):
        super().__init__()

        self.conv1 = FRAM(dim, ratio=3)

        self.conv2 = CFM(dim, ffn_scale, use_se, idx=idx)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

class FRnet(nn.Module):
    def __init__(self, dim=36, n_blocks=6, ffn_scale=1.5, use_se=False, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1, bias=False)

        self.feats = nn.Sequential(*[FRABlock(dim, ffn_scale, use_se, idx) for idx in range(n_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)
        )
        
    def forward(self, x):
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.to_feat(x)
        x = self.feats(x)
        return self.to_img(x) + res



