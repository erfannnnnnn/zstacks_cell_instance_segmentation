import torch.nn as nn
import torch.nn.functional as F


class SOLO(nn.Module):

    def __init__(self, backbone, fpn, head, fpn_level_to_use):
        super(SOLO, self).__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head = head
        self.fpn_level_to_use = fpn_level_to_use

    def forward(self, x, training=False):
        H, W = x.size(-2), x.size(-1)
        x = self.backbone(x)
        x = self.fpn(x)
        x = [self.head(x[fpn_layer]) for fpn_layer in self.fpn_level_to_use]
        categories = [lvl[0] for lvl in x]
        x = [lvl[1] for lvl in x]

        if not training:
            for pyramid_lvl, m in enumerate(x):
                x[pyramid_lvl] = F.interpolate(
                    m, size=(H,W), mode='bilinear', align_corners=True)

        return tuple(categories), tuple(x)

if __name__ == '__main__':
    pass
