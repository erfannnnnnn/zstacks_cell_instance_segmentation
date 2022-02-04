# -*- coding: utf-8 -*-

import os
import torch
import torch.optim as optim
from tqdm import tqdm

from .seg_utils import SEG_loss
from .models import seg_net
from .cisd_dataset import CISDataset
from .dec_utils import Detect, Anchors
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
    z_depth = 21 if sample_type == 'stack' else 1
    use_stack = True if sample_type == 'stack' else False

    # --- init detection model
    dec_model = load_model.load_dec_seg_model(z_depth)
    dec_weights = f'instanceseg/ancis/checkpoints/ancis_detect_{sample_type}.pth'
    if not os.path.exists(dec_weights):
        raise ValueError('no weights found for the detection model')
    dec_model = load_model.load_dec_weights(dec_model, dec_weights)
    dec_model = dec_model.to(device)
    dec_model.eval()
    for param in dec_model.parameters():
        param.requires_grad = False

    # --- segmentation model
    seg_model = seg_net.SEG_NET('resnetssd18',
                                False,
                                num_classes=2,
                                pad=0.,
                                z_depth=z_depth)
    seg_model = seg_model.to(device)

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
        collate_fn=collater,
        pin_memory=True)

    optimizer = optim.Adam(
        params=filter(lambda p: p.requires_grad, seg_model.parameters()),
        lr=1e-3)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[100, 110, 120],
        gamma=0.1)
    criterion = SEG_loss(loss_function='bcedice')

    detector = Detect(n_class=2,
                      top_k=200,
                      conf_thresh=0.3,
                      nms_thresh=0.3,
                      variance=[0.1, 0.2])
    anchorGen = Anchors(256, 256)
    anchors = anchorGen.forward()

    # --- training loop
    for epoch in range(120):
        seg_model.train()

        for inputs, bboxes, labels, masks in tqdm(trainloader):
            inputs = inputs.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                locs, conf, feat_seg = dec_model(inputs)
                detections = detector(locs, conf, anchors)
                del locs
                del conf

            with torch.enable_grad():
                outputs = seg_model(detections, feat_seg)
                loss = criterion(outputs, bboxes, labels, masks)
                loss.backward()
                optimizer.step()
                del detections
                del feat_seg
                del outputs

        scheduler.step()

        m_path = f'instanceseg/ancis/checkpoints/ancis_seg_{sample_type}.pth'
        torch.save(seg_model.state_dict(), m_path)


if __name__ == '__main__':
    pass
