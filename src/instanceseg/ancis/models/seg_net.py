import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from skimage.transform import resize

BACKBONE_CHANNELS = {
    'resnetssd18'  : [64, 64, 128, 256],
    'resnetssd34'  : [64, 64, 128, 256],
    'resnetssd50'  : [64, 256, 512, 1024],
    'resnetssd101' : [64, 256, 512, 1024],
    'resnetssd152' : [64, 256, 512, 1024]
}


class CombinationModule(nn.Module):
    def __init__(self, in_size, out_size, cat_size):
        super(CombinationModule, self).__init__()
        self.up =  nn.Sequential(nn.Conv2d(in_size,
                                           out_size,
                                           kernel_size=3,
                                           padding=1,
                                           stride=1),
                                 nn.ReLU(inplace=True))
        self.cat =  nn.Sequential(nn.Conv2d(cat_size,
                                            out_size,
                                            kernel_size=1,
                                            stride=1),
                                  nn.ReLU(inplace=True))

    def forward(self, x1, x2):
        x2 = self.up(F.interpolate(x2, x1.size()[2:], mode='bilinear',
                                   align_corners=True))
        return self.cat(torch.cat([x1, x2], 1))


class Attention(nn.Module):
    def __init__(self, x_channel, y_channel):
        super(Attention,self).__init__()
        self.conv1 = nn.Conv2d(x_channel, y_channel, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(y_channel, y_channel, kernel_size=1, stride=1)

    def forward(self, x, y):
        x = F.interpolate(self.conv1(x), y.size()[2:], mode='bilinear',
                          align_corners=True)
        x = torch.sigmoid(F.relu(x + y, inplace=True))
        return F.relu(self.conv2(x * y),inplace=True)


def make_attention_layers(backbone_name, with_fpn):
    channels = BACKBONE_CHANNELS[backbone_name]
    if with_fpn:
        channels[3] = 256 # set channels[3] to FPN_out_channels

    layers = [Attention(64, channels[0]),
              Attention(channels[1], channels[0]),
              Attention(channels[2], channels[1]),
              Attention(channels[3], channels[2])]
    return layers


def make_concat_layers(backbone_name, with_fpn):
    channels = BACKBONE_CHANNELS[backbone_name]
    if with_fpn:
        channels[3] = 256 # set channels[3] to FPN_out_channels

    layers = [CombinationModule(channels[0], 64, 128),
              CombinationModule(channels[1], channels[0], 2*channels[0]),
              CombinationModule(channels[2], channels[1], 2*channels[1]),
              CombinationModule(channels[3], channels[2], 2*channels[2])]
    return layers


class SEblock(nn.Module):
    """Squeeze and excitation block applied on a tensor of shape (N, C, H, W)

    Here the squeeze is along the spatial dimensions, and excitation along
    channels (cSE).
    """

    def __init__(self, n_channels, r):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        reduc_c = n_channels // r
        self.fc = nn.Sequential(nn.Linear(n_channels, reduc_c, bias=False),
                                nn.ReLU(inplace=True),
                                nn.Linear(reduc_c, n_channels, bias=False),
                                nn.Sigmoid())

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.global_avg_pool(x).view(n,c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class SEG_NET(nn.Module):
    def __init__(self, backbone_name, with_fpn, num_classes, pad=0.03,
                 z_depth=1):
        super(SEG_NET,self).__init__()
        self.num_classes = num_classes
        self.pad = pad
        self.z_depth = z_depth
        self.layer_att = nn.ModuleList(make_attention_layers(backbone_name,
                                                             with_fpn))
        self.layer_up_concat = nn.ModuleList(make_concat_layers(backbone_name,
                                                                with_fpn))

        if self.z_depth == 1:
            self.layer_c0 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3,
                                                    padding=1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 64, kernel_size=3,
                                                    padding=1),
                                          nn.ReLU(inplace=True))
        else:
            self.layer_c0 = nn.Sequential(nn.Conv2d(3 * self.z_depth, 64,
                                                    kernel_size=3, padding=1),
                                          nn.ReLU(inplace=True),
                                          SEblock(64, r=1),
                                          nn.Conv2d(64, 64, kernel_size=3,
                                                    padding=1),
                                          nn.ReLU(inplace=True),
                                          SEblock(64, r=1))

        self.layer_head = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3,
                                                  padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(64, 1, kernel_size=3,
                                                  padding=1))

    def get_patches(self, box, feat):
        """get feature map patches

        Parameters:
        -----------
        box : torch.Tensor
            tensor containing box [y1, x1, y2, x2], normalized w.r.t image dim.
        feat : torch.Tensor
            feature map
        """
        y1, x1, y2, x2 = box
        h, w = feat.size(-2), feat.size(-1)
        y1 = np.maximum(0, np.int32(np.round(y1 * h)))
        x1 = np.maximum(0, np.int32(np.round(x1 * w)))
        y2 = np.minimum(np.int32(np.round(y2 * h)), h - 1)
        x2 = np.minimum(np.int32(np.round(x2 * w)), w - 1)
        if y2 < y1 or x2 < x1 or y2 - y1 < 2 or x2 - x1 < 2:
            return None
        else:
            return (feat[:, y1:y2+1, x1:x2+1].unsqueeze(0))

    def mask_forward(self, i_x):
        pre = None
        for i in range(len(i_x)-1, -1, -1): # from len(i_x)-1 to 0 included
            if pre is None:
                pre = i_x[i]
            else:
                attents = self.layer_att[i](pre, i_x[i])
                pre = self.layer_up_concat[i](attents, pre)

        x = self.layer_head(pre)
        x = torch.sigmoid(x)
        x = torch.squeeze(x, dim=0) # fake batch size squeezed
        x = torch.squeeze(x, dim=0) # single channel pred squeezed
        return x

    def pad_box(self, box):
        # pad around detected box, percentage of feature map size
        y1, x1, y2, x2 = box
        y1 -= self.pad
        x1 -= self.pad
        y2 += self.pad
        x2 += self.pad
        return y1, x1, y2, x2

    def forward(self, detections, feat_seg):
        """forward call of the segmentation model

        Parameters:
        -----------
        detections : torch.Tensor
            tensor of size (batch, n_class, top_k, 5) of confidence and bboxes
            after nms.
        feat_seg : list of torch.Tensor
            tensors of size: [batch_size, 3, 256, 256] -> c0
                             [batch_size, 64, 128, 128] -> c1
                             [batch_size, 256, 64, 64] -> c2
                             [batch_size, 512, 32, 32] -> c3
                             [batch_size, 1024, 16, 16] -> c4

        Returns:
        --------
        mask_patches : list of tensors
            tensors of size (w, h) representing bilinear masks
        mask_dets : list of tensors
            tensors of size (6) : [y1, x1, y2, x2, conf_score, class]
        """
        if self.z_depth > 1:
            N, C, Z, H, W = feat_seg[0].size()
            feat_seg[0] = feat_seg[0].view(N, C*Z, H, W)
        feat_seg[0] = self.layer_c0(feat_seg[0]) # -> [bs, 64, 512, 512]

        mask_patches = [[] for i in range(detections.size(0))]
        mask_dets = [[] for i in range(detections.size(0))]

        for i in range(detections.size(0)): # iterate through batch
            for j in range(1, detections.size(1)): # iterate through class, skipping background
                dects = detections[i, j, :] # size [top_k, 5]
                mask = dects[:, 0].gt(0.).expand(5, dects.size(0)).t()
                dects = torch.masked_select(dects, mask).view(-1, 5)
                if dects.size(0) == 0:
                    continue
                for box, score in zip(dects[:, 1:], dects[:, 0]):
                    box = self.pad_box(box)
                    i_x = []
                    for i_feat in range(len(feat_seg)):
                        x = self.get_patches(box, feat_seg[i_feat][i, :, :, :])
                        if x is None:
                            break
                        else:
                            i_x.append(x)
                    # --- Decoder
                    if len(i_x) > 0:
                        x = self.mask_forward(i_x)  # up pooled mask patch
                        mask_patches[i].append(x)
                        mask_dets[i].append(torch.Tensor(np.append(box,[score,j])))
        output = (mask_patches, mask_dets)

        return output
