# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF


class SEG_loss(nn.Module):
    def __init__(self, loss_function='bcedice'):
        super(SEG_loss, self).__init__()
        if loss_function == 'bce':
            self.loss_func = BCELoss()
        elif loss_function == 'dice':
            self.loss_func = SoftDice()
        elif loss_function == 'bcedice':
            self.loss_func = DiceBCELoss()
        else:
            raise ValueError('loss func expected: bce, dice or bcedice, got',
                             loss_function)

    def jaccard(self, a, b):
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        int_h = max(min(a[2], b[2]) - max(a[0], b[0]), 0.)
        int_w = max(min(a[3], b[3]) - max(a[1], b[1]), 0.)
        area_inter = int_h * int_w
        union = area_a + area_b - area_inter
        jaccard = 0. if (union == 0.) else np.divide(area_inter, union)
        return jaccard

    def forward(self, predictions, gt_boxes, gt_classes, gt_masks):
        """forward segmentation loss

        Parameters:
        -----------
        predictions : tuple
            contains mask_patches and mask_dets
        gt_boxes : list of len batch_size
            list of tensors, each tensor of size [n_gt, 4]. boxes are normalized
        gt_classes : list of len batch_size
            list of tensors, each tensor of size [n_gt]
        gt_masks : list of len batch_size
            list of tensors, each tensor of size [n_gt, img_h, img_w]
        """
        mask_patches, mask_dets = predictions
        batch_size = len(mask_patches)
        loss_total = torch.tensor(0,
                                  device=mask_patches[0][0].device,
                                  dtype=torch.float32)
        n_obj = 0

        for b in range(batch_size):
            n_pred = len(mask_patches[b])
            n_gt = gt_boxes[b].shape[0]

            for pred_obj in range(n_pred):
                obj_p_mask = mask_patches[b][pred_obj] # pred mask
                obj_p_box = mask_dets[b][pred_obj][:4] # pred box [y1, x1, y2, x2] -> normalized
                tmp_tab = np.array([self.jaccard(obj_p_box, gt_boxes[b][gt]) for
                                    gt in range(n_gt)])
                max_val = np.amax(tmp_tab)
                if max_val < 0.5:
                    continue # predicted box match no ground truth
                gt_obj = np.argmax(tmp_tab) # gt idx best fitting pred_obj

                h, w = gt_masks[b][gt_obj].size()
                y1, x1, y2, x2 = obj_p_box
                y1 = np.maximum(0, np.int32(np.round(y1*h)))
                x1 = np.maximum(0, np.int32(np.round(x1*w)))
                y2 = np.minimum(np.int32(np.round(y2*h)), h-1)
                x2 = np.minimum(np.int32(np.round(x2*w)), w-1)
                ## Crop the obj_gt_mask from gt_mask
                obj_gt_mask = gt_masks[b][gt_obj][y1:y2+1,x1:x2+1]
                obj_gt_mask = obj_gt_mask.to(obj_p_mask.device)

                loss_obj = self.loss_func.forward(obj_p_mask, obj_gt_mask)

                loss_total += loss_obj
                n_obj += 1

        if n_obj > 0:
            loss_total /= n_obj

        return loss_total


class SoftDice(nn.Module):
    def __init__(self):
        super(SoftDice, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inter = (inputs * targets).sum()
        dice = (2. * inter + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        inter = (inputs * targets).sum()
        dice_loss = 1 - (2.*inter+smooth)/(inputs.sum()+targets.sum()+smooth)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return bce + dice_loss


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, inputs, targets, reduction='mean'):
        bce = F.binary_cross_entropy(inputs, targets, reduction=reduction)
        return bce


class SEG_loss_old(nn.Module):
    def __init__(self):
        super(SEG_loss, self).__init__()

    def jaccard(self, a, b):
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        int_h = max(min(a[2], b[2]) - max(a[0], b[0]), 0.)
        int_w = max(min(a[3], b[3]) - max(a[1], b[1]), 0.)
        area_inter = int_h * int_w
        union = area_a + area_b - area_inter
        jaccard = 0. if (union == 0.) else np.divide(area_inter, union)
        return jaccard

    def forward(self, predictions, gt_boxes, gt_classes, gt_masks):
        """forward segmentation loss

        Parameters:
        -----------
        predictions : tuple
            contains mask_patches and mask_dets
        gt_boxes : list of len batch_size
            list of tensors, each tensor of size [n_gt, 4]. boxes are normalized
        gt_classes : list of len batch_size
            list of tensors, each tensor of size [n_gt]
        gt_masks : list of len batch_size
            list of tensors, each tensor of size [n_gt, img_h, img_w]
        """
        mask_patches, mask_dets = predictions
        batch_size = len(mask_patches)
        loss_total = torch.tensor(0,
                                  device=mask_patches[0][0].device,
                                  dtype=torch.float32)
        n_obj = 0

        for b in range(batch_size):
            n_pred = len(mask_patches[b])
            n_gt = gt_boxes[b].shape[0]

            for pred_obj in range(n_pred):
                obj_p_mask = mask_patches[b][pred_obj] # pred mask
                obj_p_box = mask_dets[b][pred_obj][:4] # pred box [y1, x1, y2, x2] -> normalized

                for gt_obj in range(n_gt):
                    jaccard = self.jaccard(obj_p_box, gt_boxes[b][gt_obj])
                    if jaccard >= 0.5:
                        h, w = gt_masks[b][gt_obj].size()
                        y1, x1, y2, x2 = obj_p_box
                        y1 = np.maximum(0, np.int32(np.round(y1*h)))
                        x1 = np.maximum(0, np.int32(np.round(x1*w)))
                        y2 = np.minimum(np.int32(np.round(y2*h)), h-1)
                        x2 = np.minimum(np.int32(np.round(x2*w)), w-1)
                        ## Crop the obj_gt_mask from gt_mask
                        obj_gt_mask = gt_masks[b][gt_obj][y1:y2+1,x1:x2+1]

                        if obj_p_mask.size() != obj_gt_mask.size():
                            print('[WARNING] resizing is necessary')
                            obj_gt_mask = TF.resize(obj_gt_mask,
                                size=obj_p_mask.size(),
                                interpolation=transforms.InterpolationMode.NEAREST)
                        obj_gt_mask = obj_gt_mask.to(obj_p_mask.device)

                        loss_obj = F.binary_cross_entropy(obj_p_mask,
                                                          obj_gt_mask,
                                                          reduction='mean')
                        loss_total += loss_obj
                        n_obj += 1

        if n_obj > 0:
            loss_total /= n_obj

        return loss_total


if __name__ == '__main__':
    pass
