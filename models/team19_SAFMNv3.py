import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSAFM(nn.Module):
    def __init__(self, dim, split_dim=12):
        super().__init__()

        dim2 = dim - split_dim
        self.conv1 = nn.Conv2d(dim, split_dim, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(dim, dim2, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv2d(split_dim, split_dim, 1, 1, 0, bias=False)
        
        self.dwconv = nn.Conv2d(split_dim, split_dim, 3, 1, 1, groups=split_dim, bias=False)
        self.out = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        h, w = x.size()[-2:]

        x0 = self.conv1(x)
        x1 = self.conv2(x)

        x2 = F.adaptive_max_pool2d(x0, (h//8, w//8))
        x2 = self.dwconv(x2)
        x2 = self.conv3(x2 + torch.var(x0, dim=(-2, -1), keepdim=True))
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear')
        x2 = self.act(x2) * x0

        x = torch.cat([x1, x2], dim=1)
        x = self.out(self.act(x))
        return x


# Convolutional Channel Mixer
class FFN(nn.Module):
    def __init__(self, dim, ffn_scale):
        super().__init__()
        hidden_dim = int(dim*ffn_scale)

        self.conv1 = nn.Conv2d(dim, hidden_dim, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x


class AttBlock(nn.Module):
    def __init__(self, dim, ffn_scale):
        super().__init__()

        self.conv1 = SimpleSAFM(dim, split_dim=12)
        self.conv2 = FFN(dim, ffn_scale)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


class SAFMN_NTIRE25(nn.Module):
    def __init__(self, dim, num_blocks=8, ffn_scale=2.0, upscaling_factor=4):
        super().__init__()
        self.num_blocks = num_blocks
        self.scale = upscaling_factor

        self.to_feat = nn.Conv2d(3, dim, 3, 1, 1, bias=False)

        self.feats = nn.Sequential(*[AttBlock(dim, ffn_scale) for _ in range(num_blocks)])

        self.to_img = nn.Sequential(
            nn.Conv2d(dim, 3 * upscaling_factor**2, 3, 1, 1, bias=False),
            nn.PixelShuffle(upscaling_factor)
        )

    def forward(self, x):
        res = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        x = self.to_feat(x)
        x = self.feats(x)
        return self.to_img(x) + res


if __name__== '__main__':
    #############Test Model Complexity #############
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    img = torch.randn(1, 3, 256, 256).cuda()
    model = SAFMN_NTIRE25(dim=40, num_blocks=6, ffn_scale=1.5, upscaling_factor=4).cuda()
    print(model)

    model.train()
    output_train = model(img)

    model = model.eval()
    with torch.no_grad():
        output_eval = model(img)

    assert torch.allclose(output_train[-1], output_eval, rtol=1e-5, atol=1e-5)
    
    print(flop_count_table(FlopCountAnalysis(model, img), activations=ActivationCountAnalysis(model, img)))
    if isinstance(output_eval, list):
        for idx, out in enumerate(output_eval):
            print(f'idx: {idx}, out: {out.shape}')
    else:
        print(output_eval.shape)
