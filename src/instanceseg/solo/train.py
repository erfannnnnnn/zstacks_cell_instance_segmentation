# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from tqdm import tqdm

from .ops.losses.solo import SOLOLoss
from .cisd_dataset import CISDataset
from . import load_model


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def collater(data):
    imgs, bboxes, labels, masks = [], [], [], []
    for sample in data:
        imgs.append(sample[0])
        bboxes.append(sample[1])
        labels.append(sample[2])
        masks.append(sample[3])
    return torch.stack(imgs, 0), bboxes, labels, masks


def train(data_json_file, img_folder, sample_type):
    # --- load model
    model = load_model.load_model(sample_type)
    model.to(device)

    # --- get dataloader
    use_stack = True if sample_type == 'stack' else False
    trainset = CISDataset(data_json_file,
                          img_folder,
                          traindevtest='train',
                          augment=True,
                          split=0,
                          use_stack=use_stack)

    devset = CISDataset(data_json_file,
                        img_folder,
                        traindevtest='dev',
                        augment=False,
                        split=0,
                        use_stack=use_stack)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=1, # TODO:  true value is 8
        shuffle=True,
        collate_fn=collater,
        pin_memory=True)

    devloader = torch.utils.data.DataLoader(
        devset,
        batch_size=1,
        shuffle=False,
        collate_fn=collater,
        pin_memory=True)

    # --- optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[90, 100],
        gamma=0.1)

    # --- criterion
    criterion = SOLOLoss(
        num_classes=2,
        pyramid_levels=[0],
        scale_ranges=((0, 1),),
        epsilon=0.2,
        lambda_mask=5,
        gamma=3)

    # --- training loop
    for epoch in range(100):
        model.train()

        for inputs, bboxes, labels, masks in tqdm(trainloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            output = model(x=inputs, training=True)
            batch_loss, cat_loss, mask_loss = criterion(
                output, bboxes, labels, masks)
            batch_loss.backward()
            optimizer.step()
        scheduler.step()

        model_savepath = f'instanceseg/solo/checkpoints/solo_{sample_type}.pth'
        torch.save(model.state_dict(), model_savepath)


if __name__ == '__main__':
    pass
