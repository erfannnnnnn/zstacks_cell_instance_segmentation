# -*- coding: utf-8 -*-

import os
import numpy as np
import torch

from .seg_utils import seg_eval
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


def test(data_json_file, img_folder, sample_type):
    z_depth = 21 if sample_type == 'stack' else 1
    use_stack = True if sample_type == 'stack' else False

    # --- detection model
    dec_model = load_model.load_dec_seg_model(z_depth)
    dec_weights = f'instanceseg/ancis/checkpoints/ancis_detect_{sample_type}.pth'
    if not os.path.exists(dec_weights):
        raise ValueError('no weights found for the detection model')
    dec_model = load_model.load_dec_weights(dec_model, dec_weights)
    dec_model = dec_model.to(device)
    dec_model.eval()

    # --- segmentation model
    seg_model = seg_net.SEG_NET('resnetssd18',
                                False,
                                num_classes=2,
                                pad=0.,
                                z_depth=z_depth)
    seg_weights = f'instanceseg/ancis/checkpoints/ancis_seg_{sample_type}.pth'
    if not os.path.exists(seg_weights):
        raise ValueError('no weights found for the segmentation model')
    seg_model.load_state_dict(torch.load(seg_weights))
    seg_model = seg_model.to(device)
    seg_model.eval()

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

    detector = Detect(n_class=2,
                      top_k=200,
                      conf_thresh=0.3,
                      nms_thresh=0.3,
                      variance=[0.1, 0.2])
    anchorGen = Anchors(256, 256)
    anchors = anchorGen.forward()

    output = evaluation(dec_model, seg_model, testloader, detector, anchors)
    print('IoU@07 :', output[0])
    print('AP@07 :', output[1])
    print('AR@07 :', output[2])
    print('FNR@07 :', output[3])
    print('IoU@05 :', output[4])
    print('AP@05 :', output[5])
    print('AR@05 :', output[6])
    print('FNR@05 :', output[7])


def evaluation(dec_model, seg_model, dataloader, detector, anchors):
    with torch.no_grad():
        iou07, ap07, ar07, fnr07 = [], [], [], []
        iou05, ap05, ar05, fnr05 = [], [], [], []
        for inputs, bboxes, _, masks in dataloader:
            inputs = inputs.to(device)
            locs, conf, feat_seg = dec_model(inputs)
            detections = detector(locs, conf, anchors)
            del locs
            del conf
            outputs = seg_model(detections, feat_seg)
            del detections
            del feat_seg

            for b in range(len(masks)):
                gt = masks[b].cpu().numpy()
                mask_patches, mask_dets = outputs
                out = seg2img(mask_patches[b], mask_dets[b], (256, 256))
                j07, p07, r07, fn07 = seg_eval.eval_on_image(gt, out,
                                                             iou_ratio=0.7)
                j05, p05, r05, fn05 = seg_eval.eval_on_image(gt, out,
                                                             iou_ratio=0.5)
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


def seg2img(masks, dets, img_size):
    """mask are predicted in cropped original input. Replace the predicted masks
    in context.

    Parameters:
    -----------
    mask : list
        list of mask patches, for every detected objects
    dets : list
        list of detect props associated to mask: [y1, x1, y2, x2, score, class]
    """
    n_dets = len(masks)
    h, w = img_size
    out = np.zeros((n_dets, h, w))

    for i in range(n_dets):
        y1, x1, y2, x2 = dets[i][:4]
        y1 = np.maximum(np.round(y1.item() * h), 0).astype(np.int32)
        y2 = np.minimum(np.round(y2.item() * h), h-1).astype(np.int32)
        x1 = np.maximum(np.round(x1.item() * w), 0).astype(np.int32)
        x2 = np.minimum(np.round(x2.item() * w), w-1).astype(np.int32)
        test = out[i, y1:y2+1, x1:x2+1]
        out[i, y1:y2+1, x1:x2+1] = np.round(masks[i].cpu().numpy()).astype(bool)

    return out


if __name__ == '__main__':
    pass
