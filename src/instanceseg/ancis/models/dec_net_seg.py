import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch
from torchvision.ops.deform_conv import DeformConv2d


__all__ = ['ResNetSSD',
           'resnetssd18',
           'resnetssd34',
           'resnetssd50',
           'resnetssd101',
           'resnetssd152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


def hw_flattern(x):
    return x.view(x.size()[0],x.size()[1],-1)


class Attention(nn.Module):
    def __init__(self, c):
        super(Attention,self).__init__()
        self.conv1 = nn.Conv2d(c, c//8, kernel_size=1, stride=1, bias=False)
        self.conv2 = nn.Conv2d(c, c//8, kernel_size=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(c, c, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        f = self.conv1(x)   # [bs,c',h,w]
        g = self.conv2(x)   # [bs,c',h,w]
        h = self.conv3(x)   # [bs,c',h,w]
        f = hw_flattern(f)
        f = torch.transpose(f, 1, 2)    # [bs,N,c']
        g = hw_flattern(g)              # [bs,c',N]
        h = hw_flattern(h)              # [bs,c,N]
        h = torch.transpose(h, 1, 2)    # [bs,N,c]
        s = torch.matmul(f,g)           # [bs,N,N]
        beta = F.softmax(s, dim=-1)
        o = torch.matmul(beta,h)        # [bs,N,c]
        o = torch.transpose(o, 1, 2)
        o = o.view(x.shape)
        x = o + x
        return x


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
    """

    def __init__(self, in_channels, out_channels, use_deform_conv=False):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_deform_conv = use_deform_conv

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for c in range(len(self.in_channels)):
            self.lateral_convs.append(nn.Conv2d(self.in_channels[c],
                                                self.out_channels, 1))
            self.fpn_convs.append(self.get_conv())

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

        return tuple(outs)


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


class ResNetSSD(nn.Module):

    def __init__(self, block, layers, num_classes, c4_channels, z_depth=1,
                 with_fpn=False,  with_ancis_fusion=False,use_deform_conv=False,
                 se_block=True):
        super(ResNetSSD, self).__init__()
        self.num_classes = num_classes
        self.c4_channels = c4_channels
        self.z_depth = z_depth
        self.with_fpn = with_fpn
        self.with_ancis_fusion = with_ancis_fusion
        self.use_deform_conv = use_deform_conv
        self.se_block = se_block
        if self.with_fpn and self.with_ancis_fusion:
            raise ValueError('cannot have fpn and ancis fusion at same time')
        if self.use_deform_conv and not self.with_fpn:
            print('[WARNING] setting use_deform_conv to True has no effect',
                  'when with_fpn is False')
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3 * self.z_depth, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        if (self.z_depth > 1) and self.se_block:
            self.seblock1 = SEblock(64, r=1) # reduction = 1
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        self.new_layer1 = nn.Sequential(
            nn.Conv2d(self.c4_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.new_layer2 = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.new_layer3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.new_layer4 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        if self.with_fpn:
            # in the original paper, fusion uses feature maps with 256 channels
            self.fpn = FPN(in_channels=[self.c4_channels, 512, 256, 256, 256],
                           out_channels=256,
                           use_deform_conv=self.use_deform_conv)
            self.conf_c4 = nn.Conv2d(
                256, 4*num_classes, kernel_size=3, padding=1)
            self.conf_c5 = nn.Conv2d(
                256, 6*num_classes, kernel_size=3, padding=1)

            self.locs_c4 = nn.Conv2d(256, 4*4, kernel_size=3, padding=1)
            self.locs_c5 = nn.Conv2d(256, 6*4, kernel_size=3, padding=1)

            self.att_c4 = Attention(256)
            self.att_c5 = Attention(256)

        else:
            if self.with_ancis_fusion:
                self.fusion_c4 = nn.Conv2d(
                    in_channels=self.c4_channels, out_channels=256, kernel_size=1)
                self.fusion_c5 = nn.Conv2d(
                    in_channels=512, out_channels=256, kernel_size=1)
                self.fusion_c6 = nn.Conv2d(
                    in_channels=256, out_channels=256, kernel_size=1)
                self.fusion_end = nn.Sequential(
                    nn.BatchNorm2d(768),
                    nn.Conv2d(768, self.c4_channels, kernel_size=1),
                    nn.ReLU(inplace=True))

            self.conf_c4 = nn.Conv2d(
                self.c4_channels, 4*num_classes, kernel_size=3, padding=1)
            self.conf_c5 = nn.Conv2d(
                512, 6*num_classes, kernel_size=3, padding=1)

            self.locs_c4 = nn.Conv2d(
                self.c4_channels, 4*4, kernel_size=3, padding=1)
            self.locs_c5 = nn.Conv2d(512, 6*4, kernel_size=3, padding=1)

            self.att_c4 = Attention(self.c4_channels)
            self.att_c5 = Attention(512)

        # layers independent of the use of FPN:
        self.conf_c6 = nn.Conv2d(256, 6*num_classes, kernel_size=3, padding=1)
        self.conf_c7 = nn.Conv2d(256, 6*num_classes, kernel_size=3, padding=1)
        self.conf_c8 = nn.Conv2d(256, 4*num_classes, kernel_size=3, padding=1)

        self.locs_c6 = nn.Conv2d(256, 6*4, kernel_size=3, padding=1)
        self.locs_c7 = nn.Conv2d(256, 6*4, kernel_size=3, padding=1)
        self.locs_c8 = nn.Conv2d(256, 4*4, kernel_size=3, padding=1)

        self.att_c6 = Attention(256)
        self.att_c7 = Attention(256)
        self.att_c8 = Attention(256)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def locs_forward(self, c4, c5, c6, c7, c8):
        c4_locs = self.locs_c4(c4).permute(0, 2, 3, 1).contiguous().view(
            c4.shape[0], -1, 4)
        c5_locs = self.locs_c5(c5).permute(0, 2, 3, 1).contiguous().view(
            c5.shape[0], -1, 4)
        c6_locs = self.locs_c6(c6).permute(0, 2, 3, 1).contiguous().view(
            c6.shape[0], -1, 4)
        c7_locs = self.locs_c7(c7).permute(0, 2, 3, 1).contiguous().view(
            c7.shape[0], -1, 4)
        c8_locs = self.locs_c8(c8).permute(0, 2, 3, 1).contiguous().view(
            c8.shape[0], -1, 4)
        return torch.cat([c4_locs, c5_locs, c6_locs, c7_locs, c8_locs], dim=1)

    def conf_forward(self, c4, c5, c6, c7, c8):
        c4_conf = self.conf_c4(c4).permute(0, 2, 3, 1).contiguous().view(
            c4.shape[0], -1, self.num_classes)
        c5_conf = self.conf_c5(c5).permute(0, 2, 3, 1).contiguous().view(
            c5.shape[0], -1, self.num_classes)
        c6_conf = self.conf_c6(c6).permute(0, 2, 3, 1).contiguous().view(
            c6.shape[0], -1, self.num_classes)
        c7_conf = self.conf_c7(c7).permute(0, 2, 3, 1).contiguous().view(
            c6.shape[0], -1, self.num_classes)
        c8_conf = self.conf_c8(c8).permute(0, 2, 3, 1).contiguous().view(
            c6.shape[0], -1, self.num_classes)
        return torch.cat([c4_conf, c5_conf, c6_conf, c7_conf, c8_conf], dim=1)

    def forward(self, x):
        c0 = x
        if self.z_depth == 1:
            x = self.conv1(x)
        else:
            N, C, Z, H, W = x.size()
            x = x.view(N, C*Z, H, W)
            x = self.conv1(x)
            if self.se_block:
                x = self.seblock1(x)
        x = self.relu(self.bn1(x))
        c1 = x
        x = self.layer1(self.maxpool(x))
        c2 = x
        x = self.layer2(x)
        c3 = x
        x = self.layer3(x)
        c4 = x
        x = self.new_layer1(x)
        c5 = x
        x = self.new_layer2(x)
        c6 = x
        x = self.new_layer3(x)
        c7 = x
        x = self.new_layer4(x)
        c8 = x

        if self.with_fpn: # fusion with FPN
            c4, c5, c6, c7, c8 = self.fpn((c4, c5, c6, c7, c8))
        elif self.with_ancis_fusion:
            c4_h, c4_w = c4.size(2), c4.size(3)
            c4 = torch.cat(
                [self.fusion_c4(c4),
                 F.upsample_bilinear(self.fusion_c5(c5), (c4_h, c4_w)),
                 F.upsample_bilinear(self.fusion_c6(c6), (c4_h, c4_w))],
                dim=1)
            c4 = self.fusion_end(c4)

        c4 = self.att_c4(c4)
        c5 = self.att_c5(c5)
        c6 = self.att_c6(c6)
        c7 = self.att_c7(c7)
        c8 = self.att_c8(c8)

        locs = self.locs_forward(c4, c5, c6, c7, c8)
        conf = self.conf_forward(c4, c5, c6, c7, c8)

        return (locs, conf, [c0, c1, c2, c3, c4])


def resnetssd18(pretrained=False, num_classes=2, z_depth=1, with_fpn=False,
                with_ancis_fusion=False, use_deform_conv=False, se_block=True):
    """Constructs a ResNet-18 model.

    Parameters:
    -----------
        pretrained : boolean
            If True, returns a model pre-trained on ImageNet
    """
    c4_channels = 256
    model = ResNetSSD(BasicBlock,
                      [2, 2, 2, 2],
                      num_classes,
                      c4_channels=c4_channels,
                      z_depth=z_depth,
                      with_fpn=with_fpn,
                      with_ancis_fusion=with_ancis_fusion,
                      use_deform_conv=use_deform_conv,
                      se_block=se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],
                                                 model_dir='.'), strict=False)
    return model


def resnetssd34(pretrained=False, num_classes=2, z_depth=1, with_fpn=False,
                with_ancis_fusion=False, use_deform_conv=False, se_block=True):
    """Constructs a ResNet-34 model.

    Parameters:
    -----------
        pretrained : boolean
            If True, returns a model pre-trained on ImageNet
    """
    c4_channels = 256
    model = ResNetSSD(BasicBlock,
                      [3, 4, 6, 3],
                      num_classes,
                      c4_channels=c4_channels,
                      z_depth=z_depth,
                      with_fpn=with_fpn,
                      with_ancis_fusion=with_ancis_fusion,
                      use_deform_conv=use_deform_conv,
                      se_block=se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'],
                                                 model_dir='.'), strict=False)
    return model


def resnetssd50(pretrained=False, num_classes=2, z_depth=1, with_fpn=False,
                with_ancis_fusion=False, use_deform_conv=False, se_block=True):
    """Constructs a ResNet-50 model.

    Parameters:
    -----------
        pretrained : boolean
            If True, returns a model pre-trained on ImageNet
    """
    c4_channels = 1024
    model = ResNetSSD(Bottleneck,
                      [3, 4, 6, 3],
                      num_classes,
                      c4_channels=c4_channels,
                      z_depth=z_depth,
                      with_fpn=with_fpn,
                      with_ancis_fusion=with_ancis_fusion,
                      use_deform_conv=use_deform_conv,
                      se_block=se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'],
                                                 model_dir='.'), strict=False)
    return model


def resnetssd101(pretrained=False, num_classes=2, z_depth=1, with_fpn=False,
                 with_ancis_fusion=False, use_deform_conv=False, se_block=True):
    """Constructs a ResNet-101 model.

    Parameters:
    -----------
        pretrained : boolean
            If True, returns a model pre-trained on ImageNet
    """
    c4_channels = 1024
    model = ResNetSSD(Bottleneck,
                      [3, 4, 23, 3],
                      num_classes,
                      c4_channels=c4_channels,
                      z_depth=z_depth,
                      with_fpn=with_fpn,
                      with_ancis_fusion=with_ancis_fusion,
                      use_deform_conv=use_deform_conv,
                      se_block=se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'],
                                                 model_dir='.'), strict=False)
    return model


def resnetssd152(pretrained=False, num_classes=2, z_depth=1, with_fpn=False,
                 with_ancis_fusion=False, use_deform_conv=False, se_block=True):
    """Constructs a ResNet-152 model.

    Parameters:
    -----------
        pretrained : boolean
            If True, returns a model pre-trained on ImageNet
    """
    c4_channels = 1024
    model = ResNetSSD(Bottleneck,
                      [3, 8, 36, 3],
                      num_classes,
                      c4_channels=c4_channels,
                      z_depth=z_depth,
                      with_fpn=with_fpn,
                      with_ancis_fusion=with_ancis_fusion,
                      use_deform_conv=use_deform_conv,
                      se_block=se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'],
                                                 model_dir='.'), strict=False)
    return model
