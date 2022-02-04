# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from tqdm import tqdm

from . import load_model
from .dec_utils import DecLoss, Detect, Anchors
from .cisd_dataset import CISDataset

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
    """train ANCIS detection model

    Parameters:
    -----------
    data_json_file : str
        path of the json file containing the instance masks
    img_folder : str
        path of the folder containing the images / zstacks
    sample_type : str
        either center_slice, edf or stack
    """
    use_stack = True if sample_type == 'stack' else False
    z_depth = 21 if sample_type == 'stack' else 1

    trainset = CISDataset(data_json_file,
                          img_folder,
                          traindevtest='train',
                          augment=True,
                          split=0,
                          use_stack=use_stack)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=8,
        shuffle=True,
        num_workers=6,
        collate_fn=collater,
        pin_memory=True)

    model = load_model.load_dec_model(z_depth)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=1e-3,
                          momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[50, 55, 60],
        gamma=0.1)

    criterion = DecLoss(img_height=256,
                        img_width=256,
                        num_classes=2,
                        variances=[0.1, 0.2])

    for epoch in range(60):
        model.train()
        for inputs, bboxes, labels, _ in tqdm(trainloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss_locs, loss_conf = criterion(outputs, bboxes, labels)
            loss = loss_locs + loss_conf # alpha = 1 as in SSD paper
            loss.backward()
            optimizer.step()
        scheduler.step()

        m_path = f'instanceseg/ancis/checkpoints/ancis_detect_{sample_type}.pth'
        torch.save(model.state_dict(), m_path)


if __name__ == '__main__':
    pass
