# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


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


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, z_depth, out_indices,
                 se_block):
        super(ResNet, self).__init__()
        self.num_classes = num_classes
        self.z_depth = z_depth
        self.se_block = se_block
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
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.res_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        self.out_indices = out_indices

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

    def forward(self, x):
        # with input size: [bs, 3, 256, 256]
        if self.z_depth == 1:
            x = self.conv1(x)
        else:
            N, C, Z, H, W = x.size()
            x = x.view(N, C*Z, H, W)
            x = self.conv1(x)
            if self.se_block:
                x = self.seblock1(x)

        x = self.bn1(x)
        x = self.relu(x) # shape: [bs, 64, 128, 128]
        x = self.maxpool(x) # shape: [bs, 64, 64, 64]

        outs = []
        # For resnet50:
        # - layer1 shape: [bs, 256, 64, 64]
        # - layer2 shape: [bs, 512, 32, 32]
        # - layer3 shape: [bs, 1024, 16, 16]
        # - layer4 shape: [bs, 2048, 8, 8]
        for i, layer in enumerate(self.res_layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)


def resnet18(pretrained=False, num_classes=2, z_depth=1, out_indices=[1, 2, 3],
             se_block=True):
    """Constructs a ResNet-18 model."""
    model = ResNet(
        BasicBlock, [2, 2, 2, 2], num_classes, z_depth, out_indices, se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],
                              model_dir='.'), strict=False)
    return model


def resnet34(pretrained=False, num_classes=2, z_depth=1, out_indices=[1, 2, 3],
             se_block=True):
    """Constructs a ResNet-34 model."""
    model = ResNet(
        BasicBlock, [3, 4, 6, 3], num_classes, z_depth, out_indices, se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'],
                              model_dir='.'), strict=False)
    return model


def resnet50(pretrained=False, num_classes=2, z_depth=1, out_indices=[1, 2, 3],
             se_block=True):
    """Constructs a ResNet-50 model."""
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], num_classes, z_depth, out_indices, se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'],
                              model_dir='.'), strict=False)
    return model


def resnet101(pretrained=False, num_classes=2, z_depth=1, out_indices=[1, 2, 3],
              se_block=True):
    """Constructs a ResNet-101 model."""
    model = ResNet(
        Bottleneck, [3, 4, 23, 3], num_classes, z_depth, out_indices, se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101'],
                              model_dir='.'), strict=False)
    return model


def resnet152(pretrained=False, num_classes=2, z_depth=1, out_indices=[1, 2, 3],
              se_block=True):
    """Constructs a ResNet-152 model."""
    model = ResNet(
        Bottleneck, [3, 8, 36, 3], num_classes, z_depth, out_indices, se_block)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152'],
                              model_dir='.'), strict=False)
    return model


class ResNetBackbone(nn.Module):

    def __init__(self, model_name, n_class, z_depth, return_layers, use_se=True,
                 pretrained=False, freeze=False):
        super(ResNetBackbone, self).__init__()

        if not model_name in __all__:
            raise ValueError('model name expected to be in', __all__)

        args = {'pretrained' : pretrained,
                'num_classes': n_class,
                'z_depth'    : z_depth,
                'out_indices': return_layers,
                'se_block'   : use_se}

        if model_name == 'resnet18':
            self.backbone =  resnet18(**args)
        elif model_name == 'resnet34':
            self.backbone = resnet34(**args)
        elif model_name == 'resnet50':
            self.backbone = resnet50(**args)
        elif model_name == 'resnet101':
            self.backbone = resnet101(**args)
        elif model_name == 'resnet152':
            self.backbone = resnet152(**args)
        if freeze:
            self.backbone = self.freeze_network(self.backbone)

    def freeze_network(self, network):
        for param in network.parameters():
            param.requires_grad = False
        return network

    def forward(self, x):
        return self.backbone(x)


if __name__ == '__main__':
    pass
