# -*- coding: utf-8 -*-

import os
import json
import torch
import random
import imageio
import numpy as np
from torchvision import transforms
from skimage import img_as_float32

from ..shared import volumeio
from . import config, augmentations


class CISDataset(torch.utils.data.Dataset):
    def __init__(self, json_file, img_folder=None, traindevtest='train',
                 augment=False, split=0, use_stack=False):
        super(CISDataset, self).__init__()
        if not traindevtest in ['train', 'dev', 'test']:
            raise ValueError('traindevtest should be either train or test')
        with open(json_file, 'r') as f:
            data = json.load(f)
        self.img_folder = img_folder
        self.traindevtest = traindevtest
        self.split = split
        self.use_stack = use_stack
        if self.img_folder is None:
            self.img_folder = data['info']['img_folder']
            if self.use_stack:
                print('[Warning]: use stack and default img folder')
        if traindevtest in ['train', 'dev']:
            traintest = 'train'
        elif traindevtest == 'test':
            traintest = 'test'
        self.assets = [a for a in data['assets'] if a['set'] == traintest]
        if traintest == 'train':
            self._do_train_dev_split(train_ratio=0.75)
        self.transform = get_transforms(augment)

    def _get_sample(self, asset):
        file_name = asset['file_name']
        if self.use_stack:
            file_name = os.path.splitext(file_name)[0]
        sample_path = os.path.join(self.img_folder, file_name)
        if self.use_stack:
            sample = img_as_float32(volumeio.load(sample_path))
        else:
            sample = img_as_float32(imageio.imread(sample_path))
        return sample

    def _get_annot(self, asset):
        masks, bboxes = [], []
        for obj in asset['annotations'][0]['data']:
            bboxes.append(obj['bbox'])
            masks.append(rle2mask(obj['mask']))
        bboxes = np.asarray(bboxes, dtype=np.float32)
        masks  = np.asarray(masks, dtype=np.float32)
        labels = np.ones(masks.shape[0], dtype=np.int32)
        return bboxes, masks, labels

    def __getitem__(self, idx):
        sample = self._get_sample(self.assets[idx])
        bboxes, masks, labels = self._get_annot(self.assets[idx])
        sample, bboxes, labels, masks = self.transform((sample,
                                                        bboxes,
                                                        labels,
                                                        masks))
        # normalize bboxes w.r.t image dimensions
        boxnorm = torch.Tensor([sample.size(-2), sample.size(-1),
                                sample.size(-2), sample.size(-1)])
        boxnorm = boxnorm.unsqueeze(0).expand_as(bboxes)
        bboxes /= boxnorm
        return sample, bboxes, labels, masks

    def __len__(self):
        return len(self.assets)

    def _do_train_dev_split(self, train_ratio=0.75):
        random.Random(self.split).shuffle(self.assets)
        cut = round(train_ratio * len(self.assets))
        if self.traindevtest == 'train':
            self.assets = self.assets[:cut]
        elif self.traindevtest == 'dev':
            self.assets = self.assets[cut:]


def get_transforms(augment):
    transform = []
    transform.append(augmentations.ToTensor())

    if augment:
        if config.USE_RANDOM_ROT:
            transform.append(augmentations.RandomRotation90())
        if config.USE_RANDOM_FLIP:
            transform.append(augmentations.RandomHFlip(0.5))
            transform.append(augmentations.RandomVFlip(0.5))
        if config.USE_DOWNSIZE:
            transform.append(augmentations.RandomRescale(ratio=0.5, p=0.5))
        if config.USE_COLOR_JITTER:
            transform.append(augmentations.ColorJitter(
                brightness=config.DELTA_BRIGHTNESS,
                contrast=config.DELTA_CONTRAST,
                saturation=config.DELTA_SATURATION,
                hue=config.DELTA_HUE))
        if config.USE_RANDOM_GRAY:
            transform.append(augmentations.RandomGrayscale(0.5))
        if config.USE_RANDOM_CROP:
            transform.append(augmentations.RandomCrop(config.INPUT_SIZE))
        else:
            transform.append(augmentations.Resize(config.INPUT_SIZE))

    else:
        transform.append(augmentations.Resize(config.INPUT_SIZE))

    transform = transforms.Compose(transform)
    return transform


def rle2mask(rle):
    values = [i % 2 for i in range(len(rle['counts']))]
    mask = []
    [mask.extend([x] * y) for x,y in zip(values, rle['counts'])]
    return np.array(mask, bool).reshape(rle['size'])


if __name__ == '__main__':
    pass
