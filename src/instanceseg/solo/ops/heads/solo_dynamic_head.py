import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..blocks.convs import ConvBlock, BasicResidualBlock
from .solo_vanilla_head import SOLOCategoryBranch


class SOLOMaskKernelBranch(nn.Module):

    def __init__(self, in_channels, num_groups, grid_number, num_convs,
                 out_channels):
        super(SOLOMaskKernelBranch, self).__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.grid_number = grid_number
        self.num_convs = num_convs
        self.out_channels = out_channels
        self.convs = self.get_convs()
        self.last_conv = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channels,
                                   kernel_size=1,
                                   stride=1,
                                   bias=True)

    def get_convs(self):
        convs = [ConvBlock(conv_params={'name': 'coord_conv',
                                        'in_channels': self.in_channels,
                                        'out_channels': self.in_channels,
                                        'kernel_size': 3,
                                        'stride': 1,
                                        'padding': 1,
                                        'dilation': 1,
                                        'groups': 1,
                                        'bias': True,
                                        'boundaries': [-1, 1]},
                           norm_params={'name': 'GN',
                                        'num_groups': self.num_groups,
                                        'num_channels': self.in_channels})]

        convs += [ConvBlock(conv_params={'name': 'conv2d',
                                         'in_channels': self.in_channels,
                                         'out_channels': self.in_channels,
                                         'kernel_size': 3,
                                         'stride': 1,
                                         'padding': 1,
                                         'dilation': 1,
                                         'groups': 1,
                                         'bias': True},
                            norm_params={'name': 'GN',
                                         'num_groups': self.num_groups,
                                         'num_channels': self.in_channels})
                  for _ in range(self.num_convs - 1)]
        convs = nn.ModuleList(convs)
        return convs

    def forward(self, x):
        x = F.interpolate(
            input=x, size=self.grid_number, mode='bilinear', align_corners=True)
        for conv in self.convs:
            x = conv(x)
        x = self.last_conv(x)
        return x


class SOLOMaskFeatureBranch(nn.Module):

    def __init__(self, in_channels, num_groups, num_convs, num_up_conv_blocks,
                 out_features_branch):
        super(SOLOMaskFeatureBranch, self).__init__()
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.num_convs = num_convs
        self.num_up_conv_blocks = num_up_conv_blocks
        self.out_features_branch = out_features_branch
        self.num_groups_out = int(out_features_branch // 8)
        self.convs = self.get_convs()
        self.up_convs = self.get_up_convs()

    def get_convs(self):
        convs = [ConvBlock(conv_params={'name': 'coord_conv',
                                        'in_channels': self.in_channels,
                                        'out_channels': self.in_channels,
                                        'kernel_size': 3,
                                        'stride': 1,
                                        'padding': 1,
                                        'dilation': 1,
                                        'groups': 1,
                                        'bias': True,
                                        'boundaries': [-1, 1]},
                           norm_params={'name': 'GN',
                                        'num_groups': self.num_groups,
                                        'num_channels': self.in_channels})]
        convs += [ConvBlock(conv_params={'name': 'conv2d',
                                         'in_channels': self.in_channels,
                                         'out_channels': self.in_channels,
                                         'kernel_size': 3,
                                         'stride': 1,
                                         'padding': 1,
                                         'dilation': 1,
                                         'groups': 1,
                                         'bias': True},
                            norm_params={'name': 'GN',
                                         'num_groups': self.num_groups,
                                         'num_channels': self.in_channels})
                  for _ in range(self.num_convs - 2)]
        convs += [ConvBlock(conv_params={
                                'name': 'conv2d',
                                'in_channels': self.in_channels,
                                'out_channels': self.out_features_branch,
                                'kernel_size': 3,
                                'stride': 1,
                                'padding': 1,
                                'dilation': 1,
                                'groups': 1,
                                'bias': True,},
                            norm_params={
                                'name': 'GN',
                                'num_groups': self.num_groups_out,
                                'num_channels': self.out_features_branch})]
        return nn.ModuleList(convs)

    def get_up_convs(self):
        up_convs = [ConvBlock(
            conv_params={'name': 'conv2d',
                         'in_channels': self.out_features_branch,
                         'out_channels': self.out_features_branch,
                         'kernel_size': 3,
                         'stride': 1,
                         'padding': 1,
                         'dilation': 1,
                         'groups': 1,
                         'bias': True},
            norm_params={'name': 'GN',
                         'num_groups': self.num_groups_out,
                         'num_channels': self.out_features_branch})
            for _ in range(self.num_up_conv_blocks)]
        return nn.ModuleList(up_convs)

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        for up_conv in self.up_convs:
            x = F.interpolate(
                x, scale_factor=2, mode='bilinear', align_corners=True)
            x = up_conv(x)
        return x


class SOLODynamicMaskBranch(nn.Module):

    def __init__(self, in_channels, num_groups, grid_number, num_convs,
                 num_up_conv_blocks, out_features_branch,
                 out_kernel_branch_filters_size):
        super(SOLODynamicMaskBranch, self).__init__()
        self.grid_number = grid_number
        self.in_channels = in_channels
        self.num_groups = num_groups
        self.num_convs = num_convs
        self.num_up_conv_blocks = num_up_conv_blocks
        self.out_features_branch = out_features_branch
        self.out_kernel_branch_filters_size = out_kernel_branch_filters_size
        self.out_channels_kernel_branch = int(
            np.prod(self.out_kernel_branch_filters_size) * out_features_branch)
        self.mask_kernel_branch = SOLOMaskKernelBranch(
            in_channels=in_channels,
            num_groups=num_groups,
            grid_number=grid_number,
            num_convs=num_convs,
            out_channels=self.out_channels_kernel_branch)
        self.mask_features_branch = SOLOMaskFeatureBranch(
            in_channels=in_channels,
            num_groups=num_groups,
            num_convs=num_convs,
            num_up_conv_blocks=num_up_conv_blocks,
            out_features_branch=out_features_branch)

    def forward(self, x):
        mask_kernels = self.mask_kernel_branch(x)
        mask_features = self.mask_features_branch(x)
        masks = self.combine_branches(mask_features, mask_kernels)
        return masks

    def combine_branches(self, mask_features, mask_kernels):
        batch_size = mask_features.size(0)
        S = mask_kernels.size(-1) # S is grid_size
        k_h, k_w = self.out_kernel_branch_filters_size
        padding = 1 if k_h == 3 else 0
        masks = []
        for n in torch.arange(batch_size):
            m_feat = mask_features[n].unsqueeze(0) # size (1, feat_chan, h, w)
            m_kernels = mask_kernels[n] # size (E, S, S), with E = kernel_chan = k_h * k_w * feat_chan
            m_kernels = m_kernels.permute(1, 2, 0) # (E, S, S) -> (S, S, E)
            m_kernels = m_kernels.reshape(
                [S**2, self.out_features_branch, k_h, k_w]) # (S, S, E) -> (S**2, feat_chan, k_w, k_w)
            masks_single_example = F.conv2d(input=m_feat,
                                            weight=m_kernels,
                                            padding=padding) # size (1, S**2, h, w)
            masks.append(masks_single_example)
        masks = torch.cat(masks)
        return masks.sigmoid()


class SOLODynamicHead(nn.Module):

    def __init__(self, in_channels, num_groups, grid_number, num_convs,
                 num_up_conv_blocks, num_categories, out_features_branch,
                 out_kernel_branch_filters_size):
        super(SOLODynamicHead, self).__init__()
        self.category_branch = SOLOCategoryBranch(
            in_channels=in_channels,
            num_groups=num_groups,
            grid_number=grid_number,
            num_convs=num_convs,
            num_categories=num_categories)
        self.mask_branch = SOLODynamicMaskBranch(
            in_channels=in_channels,
            num_groups=num_groups,
            grid_number=grid_number,
            num_convs=num_convs,
            num_up_conv_blocks=num_up_conv_blocks,
            out_features_branch=out_features_branch,
            out_kernel_branch_filters_size=out_kernel_branch_filters_size)

    def forward(self, x):
        categories = self.category_branch(x)
        masks = self.mask_branch(x)
        return categories, masks


if __name__ == '__main__':
    pass
