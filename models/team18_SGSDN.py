import torch
import torch.nn as nn
import torch.nn.functional as F

class SDC(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv_0 = nn.Conv2d(in_dim, in_dim,3,1,1,groups=in_dim)
        self.GELU =nn.GELU()
        self.conv_1 = nn.Conv2d(in_dim, out_dim // 2, 1, 1, 0)

        self.pw1 = nn.Conv2d(in_dim, in_dim * 2, 1, 1, bias=False)
        self.pw2 = nn.Conv2d(in_dim, out_dim // 2, 1, 1, bias=False)

        # k_size == 35
        self.LKA35 = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=(1, 5), stride=(1, 1), padding=(0, 2), groups=in_dim),
            nn.Conv2d(in_dim, in_dim, kernel_size=(1, 11), stride=(1, 1), padding=(0, 15), groups=in_dim, dilation=3),
            nn.Conv2d(in_dim, in_dim, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0), groups=in_dim),
            nn.Conv2d(in_dim, in_dim, kernel_size=(11, 1), stride=(1, 1), padding=(15, 0), groups=in_dim, dilation=3)
        )

    def forward(self, input):
        fea = self.GELU(self.pw1(input))
        a_1, a_2 = torch.chunk(fea, 2, dim=1)

        y = self.LKA35(a_1) * a_2
        y = self.pw2(y)

        x = self.conv_0(a_1) * a_2
        x = self.GELU(self.conv_1(x))
        out = torch.cat([x, y], dim=1)
        return out

class VSD(nn.Module):
    def __init__(self, in_dim):
        super(VSD, self).__init__()
        self.conv_0 = nn.Conv2d(in_dim, in_dim * 2, 3, 1, 1, groups=in_dim)
        self.GELU = nn.GELU()

        self.conv_1 = nn.Conv2d(in_dim * 2, in_dim, 1, 1, 0)

    def forward(self, input):

        x = self.conv_1(self.GELU(self.conv_0(input)))

        return x

class LPF(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.VA1 = SDC(dim, dim)
        self.VA2 = SDC(dim, dim)
        self.VA3 = SDC(dim, dim)
        self.VSD = VSD(dim)
    def forward(self, x):
        x1 = self.VA1(F.normalize(x))
        x2 = self.VA2(F.normalize(x1))
        x3 = self.VA3(F.normalize(x2)) + x
        out = self.VSD(x3) + x3
        return out

class SGSDN(nn.Module):
    def __init__(self, num_feat=24, num_block=8, upscale=4):
        super(SGSDN, self).__init__()

        self.fea_conv = nn.Conv2d(3 , num_feat, 3,1,1)
        self.feats = nn.Sequential(*[LPF(num_feat) for _ in range(num_block)])
        self.upsampler = nn.Sequential(
            nn.Conv2d(num_feat, 3 * upscale ** 2, 3, 1, 1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, input):
        out_fea = self.fea_conv(input)
        out_lr = self.feats(out_fea) + out_fea
        output = self.upsampler(out_lr)
        return output
