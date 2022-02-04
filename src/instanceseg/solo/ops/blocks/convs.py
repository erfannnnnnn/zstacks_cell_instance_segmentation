import torch
import torch.nn as nn
from torchvision.ops.deform_conv import DeformConv2d


class ConvBlock(nn.Module):

    def __init__(self, conv_params, norm_params):
        super(ConvBlock, self).__init__()
        self.conv = self.get_conv(conv_params)
        if norm_params is not None:
            self.norm = self.get_norm(norm_params)

    def get_conv(self, conv_params):
        if conv_params['name'] == 'conv2d':
            conv = nn.Conv2d(in_channels=conv_params['in_channels'],
                             out_channels=conv_params['out_channels'],
                             kernel_size=conv_params['kernel_size'],
                             stride=conv_params['stride'],
                             padding=conv_params['padding'],
                             dilation=conv_params['dilation'],
                             groups=conv_params['groups'],
                             bias=conv_params['bias'])
            nn.init.kaiming_normal_(
                conv.weight, mode='fan_out',nonlinearity='relu')

        elif conv_params['name'] == 'coord_conv':
            conv = CoordConv(in_channels=conv_params['in_channels'],
                             out_channels=conv_params['out_channels'],
                             kernel_size=conv_params['kernel_size'],
                             stride=conv_params['stride'],
                             padding=conv_params['padding'],
                             dilation=conv_params['dilation'],
                             groups=conv_params['groups'],
                             bias=conv_params['bias'],
                             boundaries=conv_params['boundaries'])
            nn.init.kaiming_normal_(
                conv.conv.weight, mode='fan_out', nonlinearity='relu')

        elif conv_params['name'] == 'deform_conv2d':
            conv = DeformableConv(in_channels=conv_params['in_channels'],
                                  out_channels=conv_params['out_channels'],
                                  kernel_size=conv_params['kernel_size'],
                                  stride=conv_params['stride'],
                                  padding=conv_params['padding'],
                                  dilation=conv_params['dilation'],
                                  groups=conv_params['groups'],
                                  bias=conv_params['bias'])
        return conv

    def get_norm(self, norm_params):
        if norm_params['name'] == 'BN':
            norm = nn.BatchNorm2d(num_features=norm_params['num_channels'])
        elif norm_params['name'] == 'GN':
            norm = nn.GroupNorm(num_groups=norm_params['num_groups'],
                                num_channels=norm_params['num_channels'])
        nn.init.ones_(norm.weight)
        nn.init.zeros_(norm.bias)
        return norm

    def forward(self, x):
        x = self.conv(x)
        if hasattr(self, 'norm'):
            x = self.norm(x)
        x = nn.ReLU()(x)
        return x


class CoordConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, boundaries=None):
        super(CoordConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels+2,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.boundaries = boundaries

    def get_grid_coords(self, x):
        N, C, H, W = x.shape
        if self.boundaries is None:
            h = torch.range(
                start=0, end=H, dtype=torch.float32, device=x.device)
            w = torch.range(
                start=0, end=W, dtype=torch.float32, device=x.device)
        else:
            mini, maxi = self.boundaries
            h = torch.linspace(start=mini, end=maxi, steps=H, device=x.device)
            w = torch.linspace(start=mini, end=maxi, steps=W, device=x.device)

        grid_h, grid_w = torch.meshgrid(h, w)
        coords = torch.stack((grid_h, grid_w), dim=0)
        coords = coords.repeat(N, 1, 1, 1)
        return coords

    def forward(self, x):
        coords = self.get_grid_coords(x)
        x = torch.cat((x, coords), dim=1)
        x = self.conv(x)
        return x


class BasicResidualBlock(nn.Module):
    def __init__(self, conv_params, norm_params):
        super(BasicResidualBlock, self).__init__()

        self.conv_block1 = nn.Sequential(self.get_conv(conv_params),
                                         self.get_norm(norm_params),
                                         nn.ReLU())
        self.conv_block2 = nn.Sequential(self.get_conv(conv_params),
                                         self.get_norm(norm_params))
        self.relu = nn.ReLU()

    def get_conv(self, conv_params):
        if conv_params['name'] == 'conv2d':
            conv = nn.Conv2d(in_channels=conv_params['in_channels'],
                             out_channels=conv_params['out_channels'],
                             kernel_size=conv_params['kernel_size'],
                             stride=conv_params['stride'],
                             padding=conv_params['padding'],
                             dilation=conv_params['dilation'],
                             groups=conv_params['groups'],
                             bias=conv_params['bias'])
            nn.init.kaiming_normal_(
                conv.weight, mode='fan_out', nonlinearity='relu')

        elif conv_params['name'] == 'coord_conv':
            conv = CoordConv(in_channels=conv_params['in_channels'],
                             out_channels=conv_params['out_channels'],
                             kernel_size=conv_params['kernel_size'],
                             stride=conv_params['stride'],
                             padding=conv_params['padding'],
                             dilation=conv_params['dilation'],
                             groups=conv_params['groups'],
                             bias=conv_params['bias'],
                             boundaries=conv_params['boundaries'])
            nn.init.kaiming_normal_(
                conv.conv.weight, mode='fan_out', nonlinearity='relu')
        return conv

    def get_norm(self, norm_params):
        if norm_params['name'] == 'BN':
            norm = nn.BatchNorm2d(
                num_features=norm_params['num_channels'])
        elif norm_params['name'] == 'GN':
            norm = nn.GroupNorm(
                num_groups=norm_params['num_groups'],
                num_channels=norm_params['num_channels'])
        nn.init.ones_(norm.weight)
        nn.init.zeros_(norm.bias)
        return norm

    def forward(self, x):
        residual = x
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x + residual
        out = self.relu(x)
        return out


class DeformableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, groups, bias):
        super(DeformableConv, self).__init__()
        self.deform_conv = DeformConv2d(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        padding=padding,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias)
        self.offsets = nn.Conv2d(in_channels=in_channels,
                                 out_channels=2*int(kernel_size**2),
                                 kernel_size=kernel_size,
                                 stride=stride,
                                 padding=padding,
                                 dilation=dilation,
                                 groups=groups,
                                 bias=bias)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.deform_conv.weight,
                                mode='fan_out',
                                nonlinearity='relu')
        nn.init.zeros_(self.offsets.weight)

    def forward(self, x):
        offsets = self.offsets(x)
        x = self.deform_conv(x, offsets)
        return x


if __name__ == '__main__':
    pass
