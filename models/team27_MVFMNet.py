import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops

from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def get_local_weights(residual, ksize, padding):
    pad = padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
    unfolded_residual = residual_pad.unfold(2, ksize, 3).unfold(3, ksize, 3)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)
    return pixel_level_weight

def adaptive_get_local_weights(residual, out_size):
    _, _, h, w = residual.shape
    ph, pw = out_size
    stride_h = h // ph
    stride_w = w // pw
    kernel_h = h - (ph - 1) * stride_h
    kernel_w = w - (pw - 1) * stride_w
    
    unfolded_residual = residual.unfold(2, kernel_h, stride_h).unfold(3, kernel_w, stride_w)
    pixel_level_weight = (torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1))
    return pixel_level_weight


class SGFN(nn.Module):
    """ symmetrical gated feed-forward network """
    def __init__(self, dim, growth_rate=4.0, bias=False):
        super().__init__()

        hidden_dim = int(dim * growth_rate)
        
        self.project_in = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1,
                                groups=hidden_dim, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2, x3, x4 = self.dwconv(x).chunk(4, dim=1)
        y1 = F.gelu(x2) * x1
        y2 = F.gelu(x3) * x4
        x = self.project_out(torch.cat([y1, y2], dim=1))
        return x
    


class LVSA(nn.Module):
    """ local variance-aware spatial attention """
    def __init__(self, dim=36, down_scale=8):
        super().__init__()
        
        self.conv = nn.Conv2d(dim, dim, 1, 1, 0)
        self.dw_conv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        
        self.down_scale = down_scale

        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))
        
        self.gelu = nn.GELU()
    
    def forward(self, x):
        _, _, h, w = x.shape
        x_s = self.dw_conv(F.adaptive_max_pool2d(x, (h // self.down_scale, w // self.down_scale)))
        # x_v = torch.var(x, dim=(-2, -1), keepdim=True)
        x_v = adaptive_get_local_weights(x, (h // self.down_scale, w // self.down_scale))
        x_l = x * F.interpolate(self.gelu(self.conv(x_s * self.alpha + x_v * self.belt)),size=(h, w),mode="nearest",)
        return x_l
    

class MVFMB(nn.Module):
    """ multi-level variance feature modulation block """

    def __init__(self, dim=36) -> None:
        super().__init__()
        self.project_in = nn.Conv2d(dim, dim * 3, 1, 1, 0)
        
        self.sa1 = LVSA(dim=dim, down_scale=7)
        self.sa2 = LVSA(dim=dim, down_scale=5)
        
        self.linear = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, 1, 0),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim * 2, 3, 1, 1, groups=dim * 2),
            nn.GELU()
        )
        
        self.project_out = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        
    def forward(self, f):
        x_l, y, x_s = self.project_in(f).chunk(3, dim=1)
        y_l, y_s = self.linear(y).chunk(2, dim=1)
        x_l = self.sa1(x_l)
        x_s = self.sa2(x_s)
        f = self.project_out(torch.cat([y_l + x_l, y_s + x_s], dim=1))
        return f


class FMM(nn.Module):
    """ feature modulation module """

    def __init__(self, dim, ffn_scale=2.0) -> None:
        super().__init__()

        self.mvfmb = MVFMB(dim)
        self.sgfn = SGFN(dim)
        
        self.pixel_norm = nn.LayerNorm(dim)  # channel-wise
        default_init_weights([self.pixel_norm], 0.1)

    def forward(self, x):
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.pixel_norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        x = self.mvfmb(x_norm) + x
        
        x_norm = x.permute(0, 2, 3, 1)
        x_norm = self.pixel_norm(x_norm)
        x_norm = x_norm.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        
        x = self.sgfn(x_norm) + x
        return x


class MVFMNet(nn.Module):
    """ multi-level variance feature modulation network """
    def __init__(self, dim=26, n_blocks=6, ffn_scale=2, upscaling_factor=4) -> None:
        super().__init__()
        self.scale = upscaling_factor
        
        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1)
        self.feats = nn.Sequential(*[FMM(dim, ffn_scale) for _ in range(n_blocks)])
        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1),
            nn.PixelShuffle(upscaling_factor),
        )

    def forward(self, x):
        x = self.to_feat(x)
        x = self.feats(x) + x
        x = self.to_img(x)
        return x

if __name__ == "__main__":
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    # x = torch.randn(1, 3, 640, 360)  # x2
    # x = torch.randn(1, 3, 427, 240)  # x3
    x = torch.randn(1, 3, 320, 180)  # x4
    # x = torch.randn(1, 3, 256, 256)

    model = MVFMNet(dim=26, n_blocks=6, upscaling_factor=4)

    print(model)
    print(f"params: {sum(map(lambda x: x.numel(), model.parameters()))}")
    print(
        flop_count_table(
            FlopCountAnalysis(model, x), activations=ActivationCountAnalysis(model, x)
        )
    )
    output = model(x)
    print(output.shape)