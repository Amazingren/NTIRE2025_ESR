import torch
import torch.nn as nn
import torch.nn.functional as F
    
class Conv3X3_deploy(nn.Module):
    def __init__(self, c_in, c_out, ratio=1, s=1, bias=True, relu=False, deploy=False):
        super(Conv3X3_deploy, self).__init__()
        self.has_relu = relu
        self.eval_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, padding=1, stride=s, bias=bias)
    def forward(self, x):
        out = self.eval_conv(x)
        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out

class Conv1X1_deploy(nn.Module):
    def __init__(self, c_in, c_out, ratio=1, s=1, bias=True, relu=False, deploy=False):
        super(Conv1X1_deploy, self).__init__()
        self.has_relu = relu
        self.eval_conv = nn.Conv2d(c_in, c_out, 1, stride=s, bias=bias)
    def forward(self, x):
        out = self.eval_conv(x)
        if self.has_relu:
            out = F.leaky_relu(out, negative_slope=0.05)
        return out
    
class Cell(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.dim = dim
        self.chunk_dim = dim // ratio
        self.proj = Conv3X3_deploy(dim, dim)
        self.dwconv = nn.Conv2d(self.chunk_dim, self.chunk_dim, 3, 1, 1, groups=self.chunk_dim, bias=False)
        self.out = Conv1X1_deploy(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        x0, x1 = self.proj(x).split([self.chunk_dim, self.dim-self.chunk_dim], dim=1)

        x2 = F.adaptive_max_pool2d(x0, (h//8, w//8))
        x2 = self.dwconv(x2)
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
        x2 = self.act(x2) * x0

        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x


class CCM(nn.Module):
    def __init__(self, dim, ffn_scale=1.5, use_se=False):
        super().__init__()
        self.use_se = use_se
        hidden_dim = int(dim*ffn_scale)
        self.conv1 = Conv3X3_deploy(dim, hidden_dim)
        self.conv2 = Conv1X1_deploy(hidden_dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x

class REB(nn.Module):
    def __init__(self, dim, ffn_scale=1.5, use_se=False):
        super().__init__()

        self.conv1 = Cell(dim, ratio=3)
        self.conv2 = CCM(dim, ffn_scale)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x

class ExpandRepNet(nn.Module):
    def __init__(self, dim, n_blocks=8, ffn_scale=2.0, use_se=False, upscaling_factor=4):
        super().__init__()
        self.scale = upscaling_factor

        self.to_feat = Conv3X3_deploy(3, dim)

        self.feats = nn.Sequential(*[REB(dim, ffn_scale, use_se) for _ in range(n_blocks)])

        self.to_img = nn.Sequential(
            Conv3X3_deploy(dim, 3 * upscaling_factor**2),
            nn.PixelShuffle(upscaling_factor)
        )
        
    def forward(self, x):
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        x = self.to_feat(x)
        x = self.feats(x)
        return self.to_img(x) + res

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    model = ExpandRepNet(dim=36, ffn_scale=1.5, n_blocks=6).cuda()
    model.eval()
    print(model)
    state_dict = torch.load('./model_zoo/ExpandRepNet.pth')
    model.load_state_dict(state_dict['params_ema'],True)
    
    #test
    inputs = (torch.rand(1, 3, 256, 256).cuda(),)
    print(flop_count_table(FlopCountAnalysis(model, inputs)))

