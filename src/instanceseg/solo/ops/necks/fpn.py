# -*- coding: utf-8 -*-

from torch import nn
import torch.nn.functional as F

from ..blocks.convs import DeformableConv

################################################################################
# Modified FPN based on official SOLO implementation
################################################################################

class FPN(nn.Module):
    """Feature Pyramid Network for object detection.

    Parameters:
    -----------
    in_channels : list
        list of input channels
    out_channels : int
        number of out channels
    use_deform_conv : bool
        if True, use deformable convolution
    extra_maxpool : bool
        if True, adds an additionnal output, as a 2D maxpooling operation on P5.
    extra_p6p7 : bool
        if True, adds 2 addition outputs P6 and P7, computed with convolutional
        layers from P5.
    """

    def __init__(self, in_channels, out_channels, use_deform_conv=False,
                 extra_maxpool=False, extra_p6p7=False):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        if extra_maxpool and extra_p6p7:
            raise ValueError('Only one extra layer expected')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_deform_conv = use_deform_conv
        self.extra_maxpool = extra_maxpool
        self.extra_p6p7 = extra_p6p7

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for c in range(len(self.in_channels)):
            self.lateral_convs.append(nn.Conv2d(self.in_channels[c],
                                                self.out_channels, 1))
            self.fpn_convs.append(self.get_conv())

        if self.extra_p6p7:
            self.fpn_convs.extend([self.get_conv(), self.get_conv()])

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def get_conv(self):
        if self.use_deform_conv:
            conv = DeformableConv(in_channels=self.out_channels,
                                  out_channels=self.out_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  dilation=1,
                                  groups=1,
                                  bias=True)
        else:
            conv = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)
        return conv

    def forward(self, x):
        assert isinstance(x, tuple) # expects tuple of tensors
        assert len(x) == len(self.in_channels), 'x & in_channels are different'

        # build laterals
        laterals = [l_conv(x[i]) for i, l_conv in enumerate(self.lateral_convs)]

        # build top-down path
        n_levels = len(laterals)
        for i in range(n_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i], scale_factor=2, mode='nearest')

        # build outputs
        outs = [self.fpn_convs[i](laterals[i]) for i in range(n_levels)]

        # build extra layers
        if self.extra_maxpool:
            outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        elif self.extra_p6p7:
            outs.append(self.fpn_convs[n_levels](outs[-1])) # p6
            outs.append(self.fpn_convs[n_levels+1](F.relu(outs[-1]))) # p7

        return tuple(outs)


if __name__ == '__main__':
    pass
