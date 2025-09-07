from collections import OrderedDict
import torch
from torch import nn as nn
import torch.nn.functional as F

def _make_pair(value):  
    if isinstance(value, int):
        value = (value,) * 2
    return value

def conv_layer(in_channels,
               out_channels,
               kernel_size,
               bias=True):
    """
    Re-write convolution layer for adaptive `padding`.
    """
    kernel_size = _make_pair(kernel_size)
    padding = (int((kernel_size[0] - 1) / 2),
               int((kernel_size[1] - 1) / 2))
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=padding,
                     bias=bias)

def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    """
    Activation functions for ['relu', 'lrelu', 'prelu'].
    Parameters
    ----------
    act_type: str
        one of ['relu', 'lrelu', 'prelu'].
    inplace: bool
        whether to use inplace operator.
    neg_slope: float
        slope of negative region for `lrelu` or `prelu`.
    n_prelu: int
        `num_parameters` for `prelu`.
    ----------
    """
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer

def sequential(*args):
    """
    Modules will be added to the a Sequential Container in the order they
    are passed.

    Parameters
    ----------
    args: Definition of Modules in order.
    -------
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError(
                'sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2,
                       kernel_size=3):
    """
    Upsample features according to `upscale_factor`.
    """
    conv = conv_layer(in_channels,
                      out_channels * (upscale_factor ** 2),
                      kernel_size)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class Partial_conv3(nn.Module):
    def __init__(self, dim, n_div):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x

    
class Sobel(nn.Module):
    def __init__(self):
        super(Sobel, self).__init__()
        # Definir los kernels de Sobel para los gradientes horizontal (x) y vertical (y)
        kernel_x = torch.tensor([[-1., 0., 1.],
                                  [-2., 0., 2.],
                                  [-1., 0., 1.]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1., -2., -1.],
                                  [ 0.,  0.,  0.],
                                  [ 1.,  2.,  1.]], dtype=torch.float32)
        # Guardamos los kernels como buffers para que no sean actualizados durante el entrenamiento
        self.register_buffer('kernel_x', kernel_x.view(1, 1, 3, 3))
        self.register_buffer('kernel_y', kernel_y.view(1, 1, 3, 3))
    
    def forward(self, x):
        # Suponiendo que x tiene forma (N, C, H, W)
        N, C, H, W = x.shape
        # Aplicamos convolución por canal usando convoluciones en grupos
        grad_x = F.conv2d(x, self.kernel_x.repeat(C, 1, 1, 1), padding=1, groups=C)
        grad_y = F.conv2d(x, self.kernel_y.repeat(C, 1, 1, 1), padding=1, groups=C)
        # Calcular la magnitud del gradiente
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        # Devolvemos tres tensores para mantener la compatibilidad: 
        # (por ejemplo, gradiente horizontal, vertical y la magnitud)
        return grad_x, grad_y, grad_mag

class EnhancedSimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)  # Divide los canales en dos mitades
        return x1 * x2  # Multiplicación elemento a elemento

class Enhanced_IRB(nn.Module):
    def __init__(self, in_channels, expansion_factor=2, groups=2):
        super(Enhanced_IRB, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.groups = groups  # Número de grupos para el Channel Shuffle

        self.DWConv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
        self.PWConv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.shuffle = self.channel_shuffle  # Agregamos la función de shuffle
        self.PConv1 = Partial_conv3(mid_channels, n_div=4)
        self.SGate = EnhancedSimpleGate()
        self.Pconv2 = Partial_conv3(in_channels, n_div=4)
        self.Pconv3 = Partial_conv3(in_channels, n_div=4)
        self.alpha = nn.Parameter(torch.ones(1, in_channels, 1, 1), requires_grad=True)

    def channel_shuffle(self, x):
        N, C, H, W = x.shape
        assert C % self.groups == 0, "El número de canales debe ser divisible por los grupos"
        x = x.view(N, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
        return x

    def forward(self, x):
        
        res1 = self.Pconv3(x)
        f1 = self.DWConv1(x)
        f2 = self.PWConv1(f1)
        f2s = self.shuffle(f2)
        f3 = self.PConv1(f2s)
        f4 = self.SGate(f3)
        f5 = self.Pconv2(f4)
        output = self.alpha*f5 + res1

        return output
    
class EnhancedESA(nn.Module):

    def __init__(self, esa_channels, n_feats, conv, bias=True, groups = 2):
        super(EnhancedESA, self).__init__()
        f = esa_channels
        self.groups = groups  # Número de grupos para el Channel Shuffle

        self.PWConv1 = conv(n_feats, f, kernel_size=1, bias=bias)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0, bias=bias)
        self.Mpool = nn.MaxPool2d(kernel_size=7, stride=3)
        self.PConv1 = Partial_conv3(f, n_div=4)
        self.PConv2 = Partial_conv3(f, n_div=4)
        self.PWConv2 = conv(f, n_feats, kernel_size=1, bias=bias)       
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1, f, 1, 1), requires_grad=True)
        self.shuffle = self.channel_shuffle

    def channel_shuffle(self, x):
        N, C, H, W = x.shape
        assert C % self.groups == 0, "El número de canales debe ser divisible por los grupos"
        x = x.view(N, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
        return x

    def forward(self, x):

        f1 = self.PWConv1(x)
        f2 = self.conv2(f1)
        res1 = self.PConv2(f1)
        f2s = self.shuffle(f2)
        f3 = self.Mpool(f2s)
        f5 = self.PConv1(f3)
        f6 = F.interpolate(f5, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        f7 = self.alpha*f6 + res1
        f8 = self.PWConv2(f7)
        f9 = self.sigmoid(f8)
        output = f9*x

        return output
    
class EnhancedRRFB(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super(EnhancedRRFB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels

        self.IRB1 = Enhanced_IRB(mid_channels, expansion_factor=2, groups=mid_channels)
        self.act1 = nn.LeakyReLU(negative_slope=0.05)
        self.IRB2 = Enhanced_IRB(mid_channels, expansion_factor=2, groups=mid_channels)
        self.Eesa = EnhancedESA(esa_channels=32, n_feats=mid_channels, conv=conv_layer, bias=False, groups=16)

    def forward(self, x):

        f1 = self.IRB1(x)
        f2 = self.act1(f1)
        f3 = self.IRB2(f2)
        output = self.Eesa(f3)

        return output

class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, expansion_factor=2, groups=2):
        super(InvertedResidualBlock, self).__init__()
        mid_channels = in_channels * expansion_factor
        self.groups = groups  # Número de grupos para el Channel Shuffle

        self.expand = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False)
        self.shuffle = self.channel_shuffle  # Agregamos la función de shuffle
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, groups=mid_channels, bias=False)
        self.project = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, bias=False)

        # Solo agregar residual si los canales coinciden
        self.use_residual = (in_channels == in_channels)

    def channel_shuffle(self, x):
        N, C, H, W = x.shape
        assert C % self.groups == 0, "El número de canales debe ser divisible por los grupos"
        x = x.view(N, self.groups, C // self.groups, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(N, C, H, W)
        return x

    def forward(self, x):
        out = self.expand(x)  
        out = self.shuffle(out)  
        out = self.depthwise(out)  
        out = self.project(out)  
        
        if self.use_residual:
            out += x  # 🔹 Conexión residual
        return out

class FinalSWRB(nn.Module):
    def __init__(self, in_channels, mid_channels=None, out_channels=None, bias=False):
        super(FinalSWRB, self).__init__()
        if mid_channels is None:
            mid_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.c1_r = InvertedResidualBlock(mid_channels, expansion_factor=2)
        self.c2_r = InvertedResidualBlock(mid_channels, expansion_factor=2)
        self.act1 = torch.nn.SiLU()

        self.c3_r = conv_layer(out_channels, out_channels, kernel_size=1)
        self.swt = Sobel()
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

        self.concatandproyect = conv_layer(out_channels*3, out_channels, kernel_size=1)

        self.alpha = nn.Parameter(torch.ones(1, mid_channels, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1, mid_channels, 1, 1), requires_grad=True) 
        self.gamma = nn.Parameter(torch.ones(1, mid_channels, 1, 1), requires_grad=True)
        self.theta = nn.Parameter(torch.zeros(1, mid_channels, 1, 1), requires_grad=True)
        self.omega = nn.Parameter(torch.ones(1, mid_channels, 1, 1), requires_grad=True)

        self.pconv1 = Partial_conv3(mid_channels, n_div=4)
        self.pconv2 = Partial_conv3(mid_channels, n_div=4)
        self.pconv3 = Partial_conv3(mid_channels, n_div=4)
        self.pconv4 = Partial_conv3(mid_channels, n_div=4)


    def forward(self, x):

        #print('entradax', x.shape)

        out1 = self.c1_r(x)
        out1_act = self.act1(out1)

        out2 = self.c2_r(out1_act)
        out2_act = self.act1(out2)

        skip1 = x + self.omega*out2_act  # Salto residual

        #print('skip1', skip1.shape)

        lh, hl, hh = self.swt(skip1)

        lh = self.pconv1(lh)
        hl = self.pconv2(hl)
        hh = self.pconv3(hh)

        enhanced = self.alpha*lh + self.beta*hl + self.gamma*hh

        enhanced = self.pconv4(enhanced)

        #print('enhanced', enhanced.shape)

        #print('enhanced', enhanced.shape)

        attention = enhanced*skip1

        out3 = self.c3_r(attention)

        #print('out3', out3.shape)
        #print('skip1', skip1.shape)

        skip2 = torch.cat([out3, skip1, x], dim=1)

        output = self.concatandproyect(skip2)

        #print('skip2', skip2.shape)

        return output    

class SWAVE(nn.Module):
    """
    Swift Parameter-free Attention Network for Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch,
                 num_out_ch,
                 feature_channels=16,
                 upscale=4,
                 bias=False,
                 ):
        super(SWAVE, self).__init__()

        in_channels = num_in_ch
        out_channels = num_out_ch

        self.conv_init = conv_layer(in_channels,feature_channels, kernel_size=3, bias=bias) #Conv3XC(in_channels, feature_channels, gain1=2, s=1)
        self.eswrb = FinalSWRB(feature_channels, bias=bias)
        self.block_1 = EnhancedRRFB(feature_channels, bias=bias)
        #self.eswrb2 = FinalSWRB(feature_channels, bias=bias)
        self.block_2 = EnhancedRRFB(feature_channels, bias=bias)

        #self.conv_cat1 = nn.Conv2d(feature_channels * 3, feature_channels, bias=bias, kernel_size=1) #Conv1XC(feature_channels * 3, feature_channels, bias=bias)
        #self.conv_cat2 =  nn.Conv2d(feature_channels * 4, feature_channels, bias=bias, kernel_size=1) #Conv1XC(feature_channels * 4, feature_channels, bias=bias)

        self.upsampler = pixelshuffle_block(feature_channels, out_channels, upscale_factor=upscale)

        self.omega = nn.Parameter(torch.ones(1, feature_channels, 1, 1), requires_grad=True)

    def forward(self, x):
            
            f1 = self.conv_init(x)
            f2 = self.eswrb(f1)
            f3 = self.block_1(f2)
            f4 = self.block_2(f3)
            f7 = self.omega*f4 + f1
            out = self.upsampler(f7)

            return out #{"img":out}


def SWave(cfg):
    num_channels= cfg["in_ch"]
    num_feats   = cfg["num_feat"]
    scale       = cfg["scale"]

    model = SWAVE(num_in_ch=num_channels, num_out_ch=num_channels, upscale=scale, feature_channels=num_feats)
    return model

