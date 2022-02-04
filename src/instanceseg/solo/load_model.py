# -*- coding: utf-8 -*-

from .ops.backbones.resnets import ResNetBackbone
from .ops.necks.fpn import FPN
from .ops.heads.solo_dynamic_head import SOLODynamicHead
from .ops.detectors.solo import SOLO


def load_model(sample_type):
    """load SOLO model"""
    z_depth = 21 if sample_type == 'stack' else 1
    use_se = True if sample_type == 'stack' else False

    # --- cfg backbone
    backbone = ResNetBackbone(model_name='resnet18',
                              n_class=2,
                              z_depth=z_depth,
                              return_layers=[0, 1, 2, 3],
                              use_se=use_se,
                              pretrained=False,
                              freeze=False)

    # --- cfg FPN
    fpn = FPN(in_channels=[64, 128, 256, 512],
              out_channels=64,
              use_deform_conv=False,
              extra_maxpool=False,
              extra_p6p7=False)

    # --- cfg HEAD
    head = SOLODynamicHead(
        in_channels=64,
        num_groups=16,
        grid_number=32,
        num_convs=2,
        num_up_conv_blocks=2,
        num_categories=2,
        out_features_branch=32,
        out_kernel_branch_filters_size=[3, 3])

    # final model
    solo = SOLO(backbone=backbone,
                fpn=fpn,
                head=head,
                fpn_level_to_use=[0])
    return solo


if __name__ == '__main__':
    pass
