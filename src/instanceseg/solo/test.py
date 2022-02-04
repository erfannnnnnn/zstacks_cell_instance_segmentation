# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn

from .eval_utils import seg_eval, Postprocess
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


def test(data_json_file, img_folder, sample_type):
    # --- load model
    model = load_model.load_model(sample_type)
    model_weights = f'instanceseg/solo/checkpoints/solo_{sample_type}.pth'
    if os.path.exists(model_weights):
        model.load_state_dict(torch.load(model_weights))
    else:
        raise FileNotFoundError('Model weights are not available')
    model.eval()
    model.to(device)

    # --- get dataloader
    use_stack = True if sample_type == 'stack' else False
    testset = CISDataset(data_json_file,
                         img_folder,
                         traindevtest='test',
                         augment=False,
                         split=0,
                         use_stack=use_stack)

    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=1,
                                             shuffle=False,
                                             collate_fn=collater,
                                             pin_memory=True)

    # --- perform evaluation
    output = evaluation(model, testloader)
    print('IoU@07 :', output[0])
    print('AP@07 :', output[1])
    print('AR@07 :', output[2])
    print('FNR@07 :', output[3])
    print('IoU@05 :', output[4])
    print('AP@05 :', output[5])
    print('AR@05 :', output[6])
    print('FNR@05 :', output[7])


def evaluation(model, dataloader):
    """perform evaluation

    Evaluation is designed to work with a batch size of 1. Other batch size
    might create issues.
    """
    # define postprocessor:
    postprocessor = Postprocess(n_class=2,
                                top_k=200,
                                conf_thresh=0.3,
                                nms_thresh=0.7)

    model.eval()
    with torch.no_grad():
        iou07, ap07, ar07, fnr07 = [], [], [], []
        iou05, ap05, ar05, fnr05 = [], [], [], []
        for inputs, bboxes, labels, masks in dataloader:
            inputs = inputs.to(device)
            pred_cat, pred_masks = model(x=inputs, training=False)

            pos_scores, pos_labels, pos_masks = [], [], []
            # iterate through pyramid levels
            for p_level in range(len(pred_cat)):
                categories = pred_cat[p_level].squeeze(0) # size (n_class, S, S)
                lvl_masks = pred_masks[p_level].squeeze(0) # size (S**2, H, W)
                categories = categories.permute(1, 2, 0) # -> (S, S, n_class)
                categories = categories.reshape([-1, categories.size(-1)]) # -> (S*S, n_class)
                categories = nn.Softmax(dim=-1)(categories)
                scores, labels = torch.max(categories, axis=-1)

                pos = labels > 0 # Select positive instances
                pos_scores.append(scores[pos])
                pos_labels.append(labels[pos])
                pos_masks.append(lvl_masks[pos])

            # Float masks to binary masks, decision threshold is then 0.5
            pos_masks = [torch.round(pos_mask) for pos_mask in pos_masks]

            pos_scores = torch.cat(pos_scores)
            pos_labels = torch.cat(pos_labels)
            pos_masks = torch.cat(pos_masks)

            if pos_scores.numel() > 0: # there is at least one instance detected
                pos_masks, pos_scores, pos_labels = postprocessor(
                    pos_masks, pos_scores, pos_labels)
            else:
                pos_masks = np.zeros(0)
                pos_scores = np.zeros(0)
                pos_labels = np.zeros(0)

            gt = masks[0].cpu().numpy() # batch size is expected to be 1
            j05, p05, r05, fn05 = seg_eval.eval_on_image(
                gt, pos_masks, iou_ratio=0.5)
            j07, p07, r07, fn07 = seg_eval.eval_on_image(
                gt, pos_masks, iou_ratio=0.7)

            iou07.append(j07)
            ap07.append(p07)
            ar07.append(r07)
            fnr07.append(fn07)
            iou05.append(j05)
            ap05.append(p05)
            ar05.append(r05)
            fnr05.append(fn05)

    iou07 = np.mean(iou07)
    ap07 = np.mean(ap07)
    ar07 = np.mean(ar07)
    fnr07 = np.mean(fnr07)
    iou05 = np.mean(iou05)
    ap05 = np.mean(ap05)
    ar05 = np.mean(ar05)
    fnr05 = np.mean(fnr05)

    return [iou07, ap07, ar07, fnr07, iou05, ap05, ar05, fnr05]


if __name__ == '__main__':
    pass
