import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

class RRRB(nn.Module):
    """
    Residual in residual reparameterizable block (RRRB)
    @inproceedings{du2022fast,
        title={Fast and memory-efficient network towards efficient image super-resolution},
        author={Du, Zongcai and Liu, Ding and Liu, Jie and Tang, Jie and Wu, Gangshan and Fu, Lean},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        pages={853--862},
        year={2022}
    }
    """

    def __init__(self, n_feats, ratio=2):

        super(RRRB, self).__init__()
        mid_ch = n_feats * ratio

        # Define the layers
        self.conv_expand = nn.Conv2d(n_feats, mid_ch, 1, 1, 0)
        self.conv_feat = nn.Conv2d(mid_ch, mid_ch, 3, 1, 0)
        self.conv_reduce = nn.Conv2d(mid_ch, n_feats, 1, 1, 0)

        # Store reparameterized weights (used in inference)
        self.conv_fused = None
        self.register_buffer("fuse_ones", torch.ones(1, mid_ch, 3, 3))  # A helper for fusion

    def forward(self, x):

        if self.conv_fused is not None:
            return self.conv_fused(x)

        out = self.conv_expand(x)
        out_identity = out

        # explicitly padding with bias for reparameterizing in the test phase
        b0 = self.conv_expand.bias.view(1, -1, 1, 1)
        out = F.pad(out, (1, 1, 1, 1), "constant", 0)
        out[:, :, 0:1, :] = b0
        out[:, :, -1:, :] = b0
        out[:, :, :, 0:1] = b0
        out[:, :, :, -1:] = b0

        out = self.conv_feat(out) + out_identity
        out = self.conv_reduce(out)
        out += x

        return out

    def fuse_model(self):
        """Merges the 1×1, 3×3, and 1×1 convolutions into a single equivalent 3×3 conv."""

        if self.conv_fused is not None:
            return  # Already fused

        k0, b0 = self.conv_expand.weight.data, self.conv_expand.bias.data
        k1, b1 = self.conv_feat.weight.data, self.conv_feat.bias.data
        k2, b2 = self.conv_reduce.weight.data, self.conv_reduce.bias.data

        mid_c, in_c = k0.shape[:2]

        # first step: remove the middle identity
        for i in range(mid_c):
            k1[i, i, 1, 1] += 1.0

        # second step: merge the first 1x1 convolution and the next 3x3 convolution
        merge_k0k1 = F.conv2d(input=k1, weight=k0.permute(1, 0, 2, 3))
        merge_b0b1 = b0.view(1, -1, 1, 1) * self.fuse_ones
        merge_b0b1 = F.conv2d(input=merge_b0b1, weight=k1, bias=b1)

        # third step: merge the remain 1x1 convolution
        merge_k0k1k2 = F.conv2d(input=merge_k0k1.permute(1, 0, 2, 3), weight=k2).permute(1, 0, 2, 3)
        merge_b0b1b2 = F.conv2d(input=merge_b0b1, weight=k2, bias=b2).view(-1)

        # last step: remove the global identity
        for i in range(in_c):
            merge_k0k1k2[i, i, 1, 1] += 1.0

        # Create the new fused 3x3 convolution layer
        self.conv_fused = nn.Conv2d(in_c, in_c, 3, 1, 1, bias=True)
        self.conv_fused.weight.data = merge_k0k1k2
        self.conv_fused.bias.data = merge_b0b1b2

        # Delete old layers to save memory
        del self.conv_expand, self.conv_feat, self.conv_reduce, self.fuse_ones


class ESA(nn.Module):

    def __init__(self, in_c, mid_c):
        super(ESA, self).__init__()
        f = mid_c
        self.conv1 = nn.Conv2d(in_c, f, kernel_size=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = RRRB(f, ratio=2)
        self.conv3_fused = None
        self.conv4 = nn.Conv2d(f, in_c, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        c1_ = self.conv1(x)
        c1 = self.conv2(c1_)
        c1 = F.max_pool2d(c1, kernel_size=7, stride=3)
        c1 = self.conv3(c1)
        c1 = F.interpolate(c1, (x.size(2), x.size(3)), mode="bilinear", align_corners=False)
        c1 = self.conv4(c1 + c1_)
        m = self.sigmoid(c1)
        return x * m

    def fuse_model(self):
        self.conv3.fuse_model()


class RLFB(nn.Module):
    """
    Residual Local Feature Block (RLFB).
    """

    def __init__(self, n_feats, n_rrrb, esa_c=16, ratio=2):
        super(RLFB, self).__init__()
        self.RRRBs = nn.ModuleList([RRRB(n_feats, ratio) for _ in range(n_rrrb)])
        self.act = nn.LeakyReLU(negative_slope=0.05, inplace=True)
        self.conv = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.esa = ESA(n_feats, esa_c)

    def forward(self, x):
        shortcut = x.clone()
        for rrrb in self.RRRBs:
            x = self.act(rrrb(x))
        x = x + shortcut
        x = self.esa(self.conv(x))
        return x

    def fuse_model(self):
        for rrrb in self.RRRBs:
            rrrb.fuse_model()
        self.esa.fuse_model()


class RLFN(nn.Module):

    def __init__(
        self,
        in_c=3,     # Input channels
        n_feats=32, # Number of features in the RRFB
        n_rrrb=4,   # Number of RRRB blocks in a RLFB
        n_blocks=4, # Number of RLFB blocks in the network
        esa_c=16,   # ESA channels
        ratio=2,    # Ratio for the RRRB (conv3x3 expand)
        scale=2,    # Upscale factor
    ):  

        super(RLFN, self).__init__()

        self.conv1 = nn.Conv2d(in_c, n_feats, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([RLFB(n_feats, n_rrrb, esa_c, ratio) for _ in range(n_blocks)])
        self.conv2 = nn.Conv2d(n_feats, n_feats, kernel_size=1)
        self.conv_up = nn.Conv2d(n_feats, in_c * scale**2, kernel_size=3, padding=1)
        self.upsample = nn.PixelShuffle(scale)

    def forward(self, x):

        out = self.conv1(x)
        shortcut = out.clone()

        for block in self.blocks:
            out = block(out)

        out = self.conv2(out) + shortcut
        out = self.conv_up(out)
        out = self.upsample(out)
        return out

    def fuse_model(self):
        for block in self.blocks:
            block.fuse_model()


class ARRLFN(nn.Module):

    def __init__(
        self,
        in_c=3,     # Input channels
        n_feats=12, # Number of features in the RRFB
        n_rrrb=3,   # Number of RRRB blocks in a RLFB
        n_blocks=3, # Number of RLFB blocks in the network
        esa_c=32,   # ESA channels
        ratio=2,    # Ratio for the RRRB (conv3x3 expand)
        scale=4,    # Upscale factor
    ):

        super(ARRLFN, self).__init__()

        # Restrict the image upscaling factor to 2 each time
        self.model = RLFN(in_c, n_feats, n_rrrb, n_blocks, esa_c, ratio, scale=2)
        self.x4 = scale == 4

    def forward(self, x):
        out = self.model(x)
        return self.model(out) if self.x4 else out

    def fuse_model(self):
        self.model.fuse_model()
        
    def load_state_dict(self, state_dict, strict=True, fused=True):
        # Override the load_state_dict method to handle the fused model
        if fused:
            self.model.fuse_model()
        super().load_state_dict(state_dict, strict)
        
    def save(self, path: Path):
        
        path = Path(path) if isinstance(path, str) else path
        print(f"Saving model to {path}")
        import copy
        save_path = path / f"checkpoint.pt"
        torch.save(self.state_dict(), save_path)
        
        fused_path = path / f"checkpoint_fused.pt"
        model_copy = copy.deepcopy(self)
        model_copy.load_state_dict(self.state_dict(), fused=False)
        model_copy.fuse_model()
        torch.save(model_copy.state_dict(), fused_path)
