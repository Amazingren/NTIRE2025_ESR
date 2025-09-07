import torch
from torch import nn as nn


def pixelshuffle_block(in_channels,
                       out_channels,
                       upscale_factor=2):
    """
    Upsample features according to `upscale_factor`.
    """
    out = nn.Sequential(nn.Conv2d(in_channels,out_channels*(upscale_factor**2),3,1,1),
              nn.PixelShuffle(upscale_factor))
    return out


class TD_block(nn.Module):
    def __init__(self,feature_channels):
        super(TD_block, self).__init__()

        self.td_1 = nn.Sequential(nn.Conv2d(feature_channels, feature_channels // 2, (3, 1), (1, 1), (1, 0), bias=False),
                                  nn.Conv2d(feature_channels // 2, feature_channels, (1, 3), (1, 1), (0, 1)))
        self.td_2 = nn.Sequential(nn.Conv2d(feature_channels, feature_channels // 2, (3, 1), (1, 1), (1, 0), bias=False),
                                  nn.Conv2d(feature_channels // 2, feature_channels, (1, 3), (1, 1), (0, 1)))
        self.td_3 = nn.Sequential(nn.Conv2d(feature_channels, feature_channels // 2, (3, 1), (1, 1), (1, 0), bias=False),
                                  nn.Conv2d(feature_channels // 2, feature_channels, (1, 3), (1, 1), (0, 1)))
        self.act = torch.nn.SiLU(inplace=True)



    def forward(self, x):
        out1 = (self.td_1(x))
        out1_act = self.act(out1)

        out2 = (self.td_2(out1_act))
        out2_act = self.act(out2)

        out3 = (self.td_3(out2_act))

        sim_att = torch.sigmoid(out3) - 0.5
        out = (out3 + x) * sim_att

        return out


class TDESR(nn.Module):
    """
    Tensor Decomposed Efficient Super-Resolution
    """

    def __init__(self,
                 num_in_ch=3,
                 num_out_ch=3,
                 feature_channels=64,
                 num_block=6
                 ):
        super(TDESR, self).__init__()

        self.head = nn.Conv2d(num_in_ch, feature_channels,3,1,1)

        body=[]
        for _ in range(5):
            body += [TD_block(feature_channels)]
        self.body = nn.Sequential(*body)
        self.fuse = nn.Conv2d(feature_channels,feature_channels,3,1,1)

        self.tail = pixelshuffle_block(feature_channels, out_channels=num_out_ch, upscale_factor=4)


    def forward(self, x):

        out0 = self.head(x)
        out_fuse = self.fuse(self.body(out0)) + out0
        output = self.tail(out_fuse)

        return output

if __name__ == "__main__":
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    model = TDESR()
    model.eval()
    inputs = (torch.rand(1, 3, 256, 256),)
    print(flop_count_table(FlopCountAnalysis(model, inputs)))

