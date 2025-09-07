from typing import Tuple, List
from torch import Tensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

# from basicsr.utils.registry import ARCH_REGISTRY



# non local attentive block
class DeepFeature(nn.Module):
    def __init__(self, dim=48):
        super(DeepFeature, self).__init__()
        
        self.proj1 = nn.Conv2d(dim,dim*2,1,1,0)
        self.dw_conv1 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.dw_conv2 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
    
        self.dwconv3 = nn.Conv2d(dim,dim,3,1,1,groups=dim)
        self.pwconv = nn.Conv2d(dim,dim,1,1,0)
        
        self.gelu = nn.GELU()
        self.down_scale = 8
        
        self.pwconv_xa = nn.Conv2d(dim,dim,1,1,0)
        
    def forward(self, x):
        b,c,h,w = x.shape
        x = self.proj1(x)
        x_a, x_b = x.chunk(2, dim=1)
        
        x_a_avg = F.adaptive_avg_pool2d(x_a, (h//self.down_scale, w//self.down_scale))
        x_a_max = F.adaptive_max_pool2d(x_a, (h//self.down_scale, w//self.down_scale))
        x_a_comb = self.dw_conv1(x_a_avg) + self.dw_conv2(x_a_max)
        x_a_comb = self.gelu(self.pwconv_xa(x_a_comb))
        x_a_comb = x_a * F.interpolate(x_a_comb, (h,w), mode='nearest')

        x_b = self.gelu(self.pwconv(self.dwconv3(x_b)))
        
        return x_a_comb, x_b
        

class MoEBlock(nn.Module):
    def __init__(self, dim=48, num_experts=4, topk=2):
        super(MoEBlock, self).__init__()
        
        self.growth = lambda i: 2**(i+1)
        
        self.deep_feature = DeepFeature(dim)
        self.moe = MoELayer(experts=[Experts(dim=dim, squeezed_dim=self.growth(i)) for i in range(num_experts)], 
                            gate=Router(dim=dim, num_experts=num_experts), 
                            num_expert=topk,
                            )
        

        self.proj_xa = nn.Conv2d(dim,dim,1,1,0)
        self.proj_xb = nn.Conv2d(dim,dim,1,1,0)
        self.proj_out = nn.Conv2d(dim,dim,1,1,0)
        
    def forward(self, x):
        x_a, x_b = self.deep_feature(x)
        x_a = self.proj_xa(x_a)
        x_b = self.proj_xb(x_b)
        
        x_moe = self.moe(x_a, x_b)
        output = self.proj_out(x_moe)
        
        return output

class MoELayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_expert: int = 2):
        super().__init__()
        
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_expert = num_expert
        
    def forward(self, x: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        gate = self.gate(x)
        logits = F.softmax(gate, dim=1)
        topk_weights, topk_experts_indices = torch.topk(logits, self.num_expert)
        
        output = x.clone()
        if self.training:
            experts_weights = torch.zeros_like(logits)
            experts_weights.scatter_(1, topk_experts_indices, logits.gather(1,topk_experts_indices))
            
            for idx, expert in enumerate(self.experts):
                output += expert(x, K) * experts_weights[:, idx:idx+1, None, None]
        else:
            selected_experts = [self.experts[idx] for idx in topk_experts_indices.squeeze(dim=0)]
            for i, exp in enumerate(selected_experts):
                output += exp(x, K) * topk_weights[:, i:i+1, None, None]

        return output

class Router(nn.Module):
    def __init__(self,
                 dim: int,
                 num_experts: int):
        super().__init__()
        
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(dim, num_experts, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)
    
class Experts(nn.Module):
    def __init__(self,dim,squeezed_dim,expand_rate=1):
        super().__init__()
        self.expand_rate = expand_rate
        self.linear1 = nn.Conv2d(dim,squeezed_dim,1,1,0)
        self.linear2 = nn.Conv2d(dim,squeezed_dim,1,1,0)
        self.linear_final = nn.Conv2d(squeezed_dim * self.expand_rate,dim,1,1,0 )
    

    def forward(self,x_a,x_b):
        x = self.linear1(x_a)
        y = self.linear2(x_b)
        # add? multiply? 
        x_y = x * y
        final = self.linear_final(x_y) 
        
        return final
        
        
'''
Second Module
'''
class DepthFFN(nn.Module):
    def __init__(self, dim, kernel_size=3, expand_rate=2, depthwise=True):
        super(DepthFFN, self).__init__()
        
        self.expand_dim = int(dim * expand_rate)
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
        
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(self.kernel_size,1), padding=(self.padding,0),  groups=dim if depthwise else 1),
            nn.Conv2d(dim, dim, kernel_size=(1,self.kernel_size), padding=(0,self.padding), groups=dim if depthwise else 1),
        )
        
        self.act = nn.GELU()
        self.proj1 = nn.Conv2d(dim, self.expand_dim, 1, 1, 0)
        self.proj2 = nn.Conv2d(dim, dim, 1, 1, 0)
        
        
    def forward(self,x):
        x = self.proj1(x)
        x = self.act(x)
        x_a, x_b = x.chunk(2, dim=1)
        x_a = self.act(self.conv(x_a))
        
        x_final = self.proj2(x_b * x_a)
        return x_final
    
    

class SpatialFocusBlock(nn.Module):
    def __init__(self, dim, num_experts=4, topk=2):
        super(SpatialFocusBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim, data_format='channels_first')
        self.block = MoEBlock(dim=dim, num_experts=num_experts, topk=topk)
        
    def forward(self, x):
        x = self.block(self.norm1(x)) + x
        return x
    
    
class DepthFocusBlock(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(DepthFocusBlock, self).__init__()
        
        self.norm1 = LayerNorm(dim, data_format='channels_first')
        self.block1 = DepthFFN(dim, kernel_size=kernel_size, expand_rate=2, depthwise=True)
        

    def forward(self, x):
        x = self.block1(self.norm1(x)) + x
        return x
    
      
# Feature Representation Module
class FRM(nn.Module):
    def __init__(self, dim, num_experts=4, topk=2, kernel_size=3):
        super().__init__()
        
        self.spatial = SpatialFocusBlock(dim, num_experts=num_experts, topk=topk)
        self.depth = DepthFocusBlock(dim, kernel_size=kernel_size)
        
    def forward(self, x):
        x = self.spatial(x)
        x = self.depth(x)
        return x
    
class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x 
        

# @ARCH_REGISTRY.register()
class MixtureofAttention_Multiply(nn.Module):
    def __init__(self, dim=36, kernel_size=7, num_out_channels=3, num_experts=3, topk=1, scale=4, num_blocks=12):
        super(MixtureofAttention_Multiply, self).__init__()
    
        self.num_out_channels = num_out_channels
        self.scale = scale
        self.img_range = 1.0
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)

        self.shallow = nn.Conv2d(3,dim,3,1,1)
        
        self.feature_extractor = nn.Sequential(*[FRM(dim, num_experts=num_experts, topk=topk, kernel_size=kernel_size) 
                                                 for _ in range(num_blocks)])
        
        
        self.conv3x3 = nn.Conv2d(dim,dim,3,1,1)
        self.norm = LayerNorm(dim, data_format="channels_first")
        self.upsampler = nn.Sequential(
            nn.Conv2d(dim, (scale**2) * self.num_out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale)
        )
        
    def forward(self, x):
        # shallow feature extractor
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        # print(x.shape, 'xshape')
        x = self.shallow(x)
        identity = x
        
        # deep feature extractor
        x = self.feature_extractor(x)
        
        # reconstruction 
        x = self.norm(x) + identity
        x = self.conv3x3(x)
        
        x = self.upsampler(x)
        
        x = x / self.img_range + self.mean
        return x 
    
