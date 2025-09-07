import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange

class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=requires_grad)

    def forward(self, x):
        return x * self.scale

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

class BSConvU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
                 padding_mode="zeros"):
        super().__init__()

        self.pw = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                            stride=1, padding=0, dilation=1, groups=1, bias=False)
        self.dw = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation, groups=out_channels, bias=bias,
                            padding_mode='reflect')

    def forward(self, fea):
        fea = self.pw(fea)
        fea = self.dw(fea)
        return fea

class CCALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.contrast(x) + self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class RCCA(nn.Module):
    def __init__(self, embed_dims, feedforward_channels, kernel_size=3):
        super(RCCA, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        self.dwconv = nn.Conv2d(in_channels=self.feedforward_channels, out_channels=self.feedforward_channels,
                                kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True,
                                groups=self.feedforward_channels)

        self.decompose = nn.Conv2d(in_channels=self.feedforward_channels, out_channels=1, kernel_size=1)
        self.sigma = ElementScale(self.feedforward_channels, init_value=1e-5, requires_grad=True)

        self.cca = CCALayer(self.feedforward_channels, self.feedforward_channels // 4)

        self.act = nn.GELU()
        self.decompose_act = nn.GELU()

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.act(x)
        x1 = self.feat_decompose(x)
        x2 = self.cca(x)
        x = x1 + x2
        return x + input

def get_local_weights(residual, ksize, padding):
    pad = padding
    residual_pad = F.pad(residual, pad=[pad, pad, pad, pad], mode='reflect')
    unfolded_residual = residual_pad.unfold(2, ksize, 3).unfold(3, ksize, 3)
    pixel_level_weight = torch.var(unfolded_residual, dim=(-1, -2), unbiased=True, keepdim=True).squeeze(-1).squeeze(-1)

    return pixel_level_weight

class ESA(nn.Module):
    def __init__(self, num_feat=24, conv=nn.Conv2d):
        super(ESA, self).__init__()
        f = num_feat // 4
        self.conv1 = nn.Conv2d(num_feat, f, 1)
        self.conv_f = nn.Conv2d(f, f, 1)
        self.conv2_1 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_2 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_3 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.conv2_4 = conv(f, f, 3, 2, 1, padding_mode='reflect')
        self.maxPooling_1 = nn.MaxPool2d(kernel_size=5, stride=3, padding=1)
        self.maxPooling_2 = nn.MaxPool2d(kernel_size=7, stride=3, padding=1)
        self.conv_max_1 = MRConv(f, f, 3, "same", input)
        self.conv_max_2 = MRConv(f, f, 3, "same", input)
        self.var_3 = get_local_weights
        self.var_4 = get_local_weights

        self.conv3_1 = MRConv(f, f, 3, "same", input)
        self.conv3_2 = MRConv(f, f, 3, "same", input)
        self.conv4 = nn.Conv2d(f, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()

    def forward(self, input):
        c1_ = self.conv1(input)  # channel squeeze
        c1_1 = self.maxPooling_1(self.conv2_1(c1_))  # strided conv 5
        c1_2 = self.maxPooling_2(self.conv2_2(c1_))  # strided conv 7
        c1_3 = self.var_3(self.conv2_3(c1_), 7, padding=1)  # strided local-var 7
        c1_4 = self.var_4(self.conv2_4(c1_), 5, padding=1)  # strided local-var 5

        v_range_1 = self.conv3_1(self.ReLU(self.conv_max_1(c1_1 + c1_4)))
        v_range_2 = self.conv3_2(self.ReLU(self.conv_max_2(c1_2 + c1_3)))

        c3_1 = F.interpolate(v_range_1, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)
        c3_2 = F.interpolate(v_range_2, (input.size(2), input.size(3)), mode='bilinear', align_corners=False)

        cf = self.conv_f(c1_)
        c4 = self.conv4((c3_1 + c3_2 + cf))
        m = self.sigmoid(c4)

        return input * m

class HVSA(nn.Module):
    def __init__(self, c_dim, conv):
        super().__init__()
        self.body = nn.Sequential(ESA(c_dim, conv))

    def forward(self, x):
        sa_x = self.body(x)
        sa_x += x
        return sa_x

# Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

class AFFN(nn.Module):

    def __init__(self, dim, ffn_expansion_factor, bias, input_resolution=None):
        super(AFFN, self).__init__()

        self.input_resolution = input_resolution
        self.dim = dim
        self.ffn_expansion_factor = ffn_expansion_factor

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=(5,2), stride=1, padding="same", groups=hidden_features * 2, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=(2,5), stride=1, padding="same", groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1 = self.dwconv1(x)
        x2 = self.dwconv2(x1)
        y1, y2 = x2.chunk(2, dim=1)
        y = F.gelu(y1) * y2
        out = self.project_out(y)
        return out

class AFFNBlock(nn.Module):
    def __init__(self, dim, restormer_num_heads=3, restormer_ffn_expansion_factor=2., tlc_flag=True, tlc_kernel=96,
                 activation='relu', input_resolution=None):
        super(AFFNBlock, self).__init__()

        self.input_resolution = input_resolution

        self.norm4 = LayerNorm(dim, LayerNorm_type='WithBias')
        self.restormer_ffn = AFFN(dim, ffn_expansion_factor=restormer_ffn_expansion_factor, bias=False,
                                         input_resolution=input_resolution)

    def forward(self, x):
        x = self.restormer_ffn(self.norm4(x)) + x
        return x

class MRConv_basic(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, input):
        super().__init__()
        self.dim = in_dim // 4
        self.input = input
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.conv2_1 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_2 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size * 2 - 1, kernel_size), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_3 = nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size * 2 - 1), padding=padding,
                                 groups=in_dim // 4)
        self.conv2_4 = [
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4),
            nn.Conv2d(in_dim // 4, out_dim // 4, (kernel_size, kernel_size), padding=padding, groups=in_dim // 4)]
        self.conv2_4 = nn.Sequential(*self.conv2_4)
        self.act = nn.GELU()

    def forward(self, input):
        out = self.conv1(input)
        out = torch.chunk(out, 4, dim=1)
        s1 = self.act(self.conv2_1(out[0]))
        s2 = self.act(self.conv2_2(out[1] + s1))
        s3 = self.act(self.conv2_3(out[2] + s2))
        s4 = self.act(self.conv2_4(out[3] + s3))
        out = torch.cat([s1, s2, s3, s4], dim=1) + input
        out = self.act(out)
        return out

class MRConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, padding, input):
        super().__init__()
        self.dim = in_dim // 3
        self.input = input
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.conv1 = nn.Conv2d(in_dim, in_dim, (1, 1))
        self.conv2_1 = nn.Conv2d(in_dim // 3, out_dim // 3, (kernel_size, kernel_size), padding=padding,
                                 groups=in_dim // 3)
        self.conv2_2 = nn.Conv2d(in_dim // 3, out_dim // 3, (kernel_size * 2 - 1, kernel_size), padding=padding,
                                 groups=in_dim // 3)
        self.conv2_3 = nn.Conv2d(in_dim // 3, out_dim // 3, (kernel_size, kernel_size * 2 - 1), padding=padding,
                                 groups=in_dim // 3)
        self.act = nn.GELU()

    def forward(self, input):
        out = self.conv1(input)
        out = torch.chunk(out, 3, dim=1)
        s1 = self.act(self.conv2_1(out[0]))
        s2 = self.act(self.conv2_2(out[1] + s1))
        s3 = self.act(self.conv2_3(out[2] + s2))
        out = torch.cat([s1, s2, s3], dim=1) + input
        out = self.act(out)
        return out

def UpsampleOneStep(in_channels, out_channels, upscale_factor=4):
    conv = nn.Conv2d(in_channels, out_channels * (upscale_factor ** 2), 3, 1, 1)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return nn.Sequential(*[conv, pixel_shuffle])

class FRB(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.act = nn.GELU()

        self.linear_0 = nn.Conv2d(dim, dim * 2, 1, 1, 0)
        self.conv0 = nn.Conv2d(self.dim, self.dim, 1)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding="same", groups=dim, bias=True)
        self.conv2_1 = nn.Conv2d(dim // 3, dim // 3, (3, 3), padding="same",
                                 groups=dim // 3)
        self.conv2_2 = nn.Conv2d(dim // 3, dim // 3, (5, 3), padding="same",
                                 groups=dim // 3)
        self.conv2_3 = nn.Conv2d(dim // 3, dim // 3, (3, 5), padding="same",
                                 groups=dim // 3)
        self.linear_2 = nn.Conv2d(dim, dim, 1, 1, 0)
        self.alpha = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.belt = nn.Parameter(torch.zeros((1, dim, 1, 1)))

    def forward(self, x):
        h, w = x.shape[2:]
        short = x
        x = F.normalize(x)
        x1 = self.linear_0(x).chunk(2, dim=1)
        x2 = torch.chunk(x1[1], 3, dim=1)

        x_v = torch.var(x1[0], dim=(-2, -1), keepdim=True)
        x_s = self.dwconv(F.adaptive_max_pool2d(x1[0], (h // 8, w // 8)))

        hfe = x1[0] * F.interpolate(self.act(self.conv0(x_s * self.alpha + x_v * self.belt)), size=(h, w),
                                mode='nearest')
        s1 = self.act(self.conv2_1(x2[0]))
        s2 = self.act(self.conv2_2(x2[1] + s1))
        s3 = self.act(self.conv2_3(x2[2] + s2))
        lfe = torch.cat([s1, s2, s3], dim=1) + x1[1]

        return self.linear_2(lfe + hfe) + short

class FMDB_basic(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d, input=input):
        super(FMDB_basic, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.input = input
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = MRConv_basic(in_channels, in_channels, 3, "same", input)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = MRConv_basic(in_channels, in_channels, 3, "same", input)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = MRConv_basic(in_channels, in_channels, 3, "same", input)
        self.c4 = BSConvU(self.remaining_channels, self.dc)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = HVSA(in_channels, conv)
        self.cca = FRB(in_channels)
        self.ea = AFFNBlock(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        out_fused = self.ea(out_fused)

        return out_fused + input

class FMDB(nn.Module):
    def __init__(self, in_channels, conv=nn.Conv2d, input=input):
        super(FMDB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.input = input
        self.c1_d = nn.Conv2d(in_channels, self.dc, 1)
        self.c1_r = MRConv_basic(in_channels, in_channels, 3, "same", input)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c2_r = MRConv_basic(in_channels, in_channels, 3, "same", input)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, 1)
        self.c3_r = MRConv_basic(in_channels, in_channels, 3, "same", input)

        self.c4 = BSConvU(self.remaining_channels, self.dc)
        self.act = nn.GELU()

        self.c5 = nn.Conv2d(self.dc * 4, in_channels, 1)
        self.esa = HVSA(in_channels, conv)
        self.cca = RCCA(in_channels, int(in_channels * 1))
        self.ea = AFFNBlock(in_channels)

    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.c1_r(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out = self.c5(out)

        out_fused = self.esa(out)
        out_fused = self.cca(out_fused)
        out_fused = self.ea(out_fused)

        return out_fused + input

class FMDN(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=24, num_block=3, num_out_ch=3, upscale=4,
                 rgb_mean=(0.4488, 0.4371, 0.4040), input=(64, 64), **kwargs):
        super(FMDN, self).__init__()
        self.conv = BSConvU
        self.scale = upscale
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.B1 = FMDB_basic(in_channels=num_feat, conv=self.conv, input=input)
        self.B2 = FMDB_basic(in_channels=num_feat, conv=self.conv, input=input)
        self.B3 = FMDB_basic(in_channels=num_feat, conv=self.conv, input=input)
        self.B4 = FMDB(in_channels=num_feat, conv=self.conv, input=input)

        self.c1 = nn.Conv2d(num_feat * num_block, num_feat, 1)
        self.GELU = nn.GELU()
        self.c2 = self.conv(num_feat, num_feat, 3, 1, 1)

        self.upsampler = UpsampleOneStep(num_feat, num_out_ch, upscale_factor=upscale)

    def forward(self, input):
        self.mean = self.mean.type_as(input)
        input = input - self.mean

        out_fea = self.fea_conv(input)

        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)

        out = self.c1(torch.cat([out_B1, out_B2, out_B3], dim=1))
        out = self.GELU(self.B4(out))
        out = self.upsampler(
            self.c2(out) + out_fea) + self.mean
        return out

if __name__ == '__main__':
    import torch

    model = FMDN(num_feat=24, upscale=4).cuda()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    x = torch.randn((1, 3, 256, 256))
    from fvcore.nn import FlopCountAnalysis

    device = torch.device("cuda:0")
    input_fake = torch.rand(1, 3, 256, 256).to(device)
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    print(flop_count_table(FlopCountAnalysis(model, input_fake)))
    flops = FlopCountAnalysis(model, input_fake).total()
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))

    model.cuda()
    out = model(x.cuda())
    print(out.shape)

    iterations = 100  # 重复计算的轮次

    device = torch.device("cuda:0")
    model.to(device)

    random_input = x.to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # GPU预热
    for _ in range(50):
        with torch.no_grad():
            _ = model(random_input)

    # 测速
    times = torch.zeros(iterations)  # 存储每轮iteration的时间
    with torch.no_grad():
        for iter in range(iterations):
            starter.record()
            _ = model(random_input)
            ender.record()
            # 同步GPU时间
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)  # 计算时间
            times[iter] = curr_time
            # print(curr_time)

    mean_time = times.mean().item()
    print("Inference time: {:.6f}, FPS: {} ".format(mean_time, 1000 / mean_time))

    print('最大显存', torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2)

    from fvcore.nn import FlopCountAnalysis

    input_fake = torch.rand(1, 3, 256, 256).to(device)
    flops = FlopCountAnalysis(model, input_fake).total()
    flops = flops / 10 ** 9
    print("{:>16s} : {:<.4f} [G]".format("FLOPs", flops))
