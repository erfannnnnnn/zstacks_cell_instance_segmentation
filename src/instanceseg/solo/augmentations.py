# -*- coding: utf-8 -*-

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
import numbers


################################################################################
# Augmentations are applied on: sample, bboxes, labels, masks
# sample can be an image (H, W, C) or a stack (Z, H, W, C)
################################################################################

class ToTensor():

    def __init__(self):
        pass

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            if img.ndim == 4:
                img = StackToTensor()(img)
            else:
                img = TF.to_tensor(img)
            bboxes = TF.Tensor(bboxes)
            labels = TF.Tensor(labels)
            masks = TF.Tensor(masks)
            x = [img, bboxes, labels, masks]
        else:
            if x.ndim == 4:
                x = StackToTensor()(x)
            else:
                x = TF.to_tensor(x)
        return x


class StackToTensor():

    def __init__(self):
        pass

    def __call__(self, x):
        x = torch.from_numpy(x.transpose(3, 0, 1, 2)).contiguous()
        if isinstance(x, torch.ByteTensor):
            x = x.float().div(255)
        return x


class RandomRotation90():

    def __init__(self):
        pass

    def __call__(self, x):
        k = torch.randint(-1, 3, (1,)).item()
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            img = torch.rot90(img, k, [-2, -1])
            masks = torch.rot90(masks, k, [-2, -1])
            bboxes = get_bbox(masks)
            x = [img, bboxes, labels, masks]
        else:
            x = torch.rot90(x, k, [-2, -1])
        return x


class Resize():

    def __init__(self, size):
        self.size = size
        self.h = self.size[0]
        self.w = self.size[1]

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            bboxes[:, 0] *= (self.h / img.size(-2))
            bboxes[:, 1] *= (self.w / img.size(-1))
            bboxes[:, 2] *= (self.h / img.size(-2))
            bboxes[:, 3] *= (self.w / img.size(-1))
            img = TF.resize(img, size=self.size,
                interpolation=transforms.InterpolationMode.BILINEAR)
            masks = TF.resize(masks, size=self.size,
                interpolation=transforms.InterpolationMode.NEAREST)
            x = [img, bboxes, labels, masks]
        else:
            x = TF.resize(x, size=self.size)
        return x


class RandomVFlip():

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            if torch.rand(1) < self.p:
                h = img.size(-2)
                img = TF.vflip(img)
                masks = TF.vflip(masks)
                bboxes[:, 0::2] = h - torch.flip(bboxes[:, 0::2], [1])
            x = [img, bboxes, labels, masks]
        else:
            if torch.rand(1) < self.p:
                x = TF.vflip(x)
        return x


class RandomHFlip():

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            if torch.rand(1) < self.p:
                w = img.size(-1)
                img = TF.hflip(img)
                masks = TF.hflip(masks)
                bboxes[:, 1::2] = w - torch.flip(bboxes[:, 1::2], [1])
            x = [img, bboxes, labels, masks]
        else:
            if torch.rand(1) < self.p:
                x = TF.hflip(x)
        return x


class RandomRescale():

    def __init__(self, ratio, p):
        self.ratio = ratio
        self.p = p

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            if torch.rand(1) < self.p:
                in_h, in_w = img.size(-2), img.size(-1)
                out_h = int(in_h * self.ratio)
                out_w = int(in_w * self.ratio)
                bboxes[:, 0] *= (out_h / in_h)
                bboxes[:, 1] *= (out_w / in_w)
                bboxes[:, 2] *= (out_h / in_h)
                bboxes[:, 3] *= (out_w / in_w)
                img = TF.resize(img, size=(out_h, out_w),
                    interpolation=transforms.InterpolationMode.BILINEAR)
                masks = TF.resize(masks, size=(out_h, out_w),
                    interpolation=transforms.InterpolationMode.NEAREST)
            x = [img, bboxes, labels, masks]
        else:
            if torch.rand(1) < self.p:
                in_h, in_w = x.size(-2), x.size(-1)
                out_h = int(in_h * self.ratio)
                out_w = int(in_w * self.ratio)
                x = TF.resize(x, size=(out_h, out_w),
                    interpolation=transforms.InterpolationMode.BILINEAR)
        return x


class RandomCrop():

    def __init__(self, size, pad_if_needed=True):
        self.size = size
        self.pad_if_needed = pad_if_needed
        self.fill = 0
        self.padding_mode = 'constant'

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            h, w = img.shape[-2], img.shape[-1]

            if self.pad_if_needed and h < self.size[0]:
                padding = [0, self.size[0] - h] # padding along w then h
                img = TF.pad(img, padding, self.fill, self.padding_mode)
                masks = TF.pad(masks, padding, self.fill, self.padding_mode)

            if self.pad_if_needed and w < self.size[1]:
                padding = [self.size[1] - w, 0] # padding along w then h
                img = TF.pad(img, padding, self.fill, self.padding_mode)
                masks = TF.pad(masks, padding, self.fill, self.padding_mode)

            i, j, h, w = transforms.RandomCrop.get_params(img,
                output_size=self.size)
            img = TF.crop(img, i, j, h, w)
            masks = TF.crop(masks, i, j, h, w)
            masks = _filter_empty_mask(masks)
            bboxes = get_bbox(masks)
            x = [img, bboxes, labels, masks]
        else:
            h, w = x.shape[-2], x.shape[-1]

            if self.pad_if_needed and h < self.size[0]:
                padding = [0, self.size[0] - h]
                x = TF.pad(x, padding, self.fill, self.padding_mode)

            if self.pad_if_needed and w < self.size[1]:
                padding = [self.size[1] - w, 0]
                x = TF.pad(x, padding, self.fill, self.padding_mode)

            i, j, h, w = transforms.RandomCrop.get_params(x,
                output_size=self.size)
            x = TF.crop(x, i, j, h, w)
        return x


class ColorJitter():
    """
    Parameters:
    -----------
    brightness: (float or tuple of float (min, max))
        How much to jitter brightness.
        brightness_factor is chosen uniformly from
        [max(0, 1 - brightness), 1 + brightness]
        or the given [min, max]. Should be non negative numbers.
    contrast: (float or tuple of float (min, max))
        How much to jitter contrast.
        contrast_factor is chosen uniformly from
        [max(0, 1 - contrast), 1 + contrast]
        or the given [min, max]. Should be non negative numbers.
    saturation: (float or tuple of float (min, max))
        How much to jitter saturation.
        saturation_factor is chosen uniformly from
        [max(0, 1 - saturation), 1 + saturation]
        or the given [min, max]. Should be non negative numbers.
    hue: (float or tuple of float (min, max))
        How much to jitter hue.
        hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
        Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness)
        self.contrast =  self._check_input(contrast)
        self.saturation =  self._check_input(saturation)
        self.hue =  self._check_input(hue, center=0, bound=(-0.5, 0.5),
                                      clip_first_on_zero=False)

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            img = self._transform(img)
            x = [img, bboxes, labels, masks]
        else:
            x = self._transform(x)
        return x

    def _transform(self, x):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                b_factor = torch.tensor(1.0).uniform_(self.brightness[0],
                    self.brightness[1]).item()
                x = TF.adjust_brightness(x, b_factor)

            if fn_id == 1 and self.contrast is not None:
                c_factor = torch.tensor(1.0).uniform_(self.contrast[0],
                    self.contrast[1]).item()
                x = _adjust_contrast(x, c_factor)

            if fn_id == 2 and self.saturation is not None:
                s_factor = torch.tensor(1.0).uniform_(self.saturation[0],
                    self.saturation[1]).item()
                x = TF.adjust_saturation(x, s_factor)

            if fn_id == 3 and self.hue is not None:
                h_factor = torch.tensor(1.0).uniform_(self.hue[0],
                    self.hue[1]).item()
                x = TF.adjust_hue(x, h_factor)
        return x

    def _check_input(self, value, center=1, bound=(0, float('inf')),
                     clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            assert value >= 0
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, tuple):
            assert bound[0] <= value[0] <= value[1] <= bound[1]
        else:
            raise ValueError('value should be number of tuple')
        if value[0] == value[1] == center:
            value = None
        return value


class RandomGrayscale():

    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        if isinstance(x, (list, tuple)):
            img, bboxes, labels, masks = x
            if torch.rand(1) < self.p:
                img = _rgb_to_grayscale(img, num_output_channels=3)
            x = [img, bboxes, labels, masks]
        else:
            if torch.rand(1) < self.p:
                x = _rgb_to_grayscale(x, num_output_channels=3)
        return x


# --- utils functions

def get_bbox(masks):
    """get bboxes (y1, x1, y2, x2) from masks"""
    bbox = torch.FloatTensor(masks.size(0), 4)
    for i in range(masks.size(0)):
        nnz_idx = torch.nonzero(masks[i])
        if nnz_idx.size(0) == 0:
            raise ValueError('empty mask')
        bbox[i, :2] = nnz_idx.min(dim=0)[0]
        bbox[i, 2:] = nnz_idx.max(dim=0)[0]
    return bbox


def _filter_empty_mask(masks):
    keep = torch.stack([masks[i].sum() > 0 for i in range(masks.size(0))])
    return masks[keep]


def _adjust_contrast(x, contrast_factor):
    if contrast_factor < 0:
        raise ValueError('contrast_factor is not non-negative.')
    assert x.ndim == 3, 'input expected dim is 3: (C, H, W)'
    dtype = x.dtype if torch.is_floating_point(x) else torch.float32
    m = torch.mean(_rgb_to_grayscale(x).to(dtype), dim=(-3,-2,-1), keepdim=True)
    return _blend(x, m, contrast_factor)


def _blend(x1, x2, ratio):
    ratio = float(ratio)
    bound = 1.0 if x1.is_floating_point() else 255.0
    return (ratio * x1 + (1.0 - ratio) * x2).clamp(0, bound).to(x1.dtype)


def _get_num_channels(tensor):
    if tensor.ndim == 2:
        return 1
    elif tensor.ndim > 2:
        return tensor.shape[0]
    else:
        raise TypeError('Input ndim should be 2 or more.')


def _rgb_to_grayscale(tensor, num_output_channels=1):
    c = _get_num_channels(tensor)
    if num_output_channels not in (1, 3):
        raise ValueError('num_output_channels should be either 1 or 3')
    if c == 1:
        grey_tensor = tensor
        if grey_tensor.ndim == 2:
            grey_tensor = grey_tensor.unsqueeze(dim=0)
    elif c == 3:
        r, g, b = tensor.unbind(dim=0)
        grey_tensor = (0.2989 * r + 0.587 * g + 0.114 * b).to(tensor.dtype)
        grey_tensor = grey_tensor.unsqueeze(dim=0)
    else:
        raise TypeError('Input channels should be 1 or 3')
    if num_output_channels == 3:
        return grey_tensor.expand(tensor.shape)

    return grey_tensor


if __name__ == '__main__':
    pass
