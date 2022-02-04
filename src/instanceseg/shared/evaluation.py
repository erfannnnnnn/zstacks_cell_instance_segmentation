# -*- coding: utf-8 -*-

import numpy as np


def eval(ground_truths, segmentations):
    """evaluate segmentation on a list of images, as mean iou score for TP,
    precision and recall.

    Parameters:
    -----------
    ground_truths : list of ndarray
        list of instance ground_truths, for every image
    segmentations : list of ndarray
        list of predicted instances masks, for every image
    """
    assert len(ground_truths) == len(segmentations)
    n_image = len(ground_truths)
    iou_07, iou_05 = np.zeros(n_image), np.zeros(n_image)
    precision_07, precision_05 = np.zeros(n_image), np.zeros(n_image)
    recall_07, recall_05 = np.zeros(n_image), np.zeros(n_image)
    fnr_07, fnr_05 = np.zeros(n_image), np.zeros(n_image)
    for i in range(n_image):
        metrics07 = eval_on_image(ground_truths[i],
                                  segmentations[i],
                                  iou_ratio=0.7,
                                  return_idx=False)
        iou_07[i] = metrics07[0]
        precision_07[i] = metrics07[1]
        recall_07[i] = metrics07[2]
        fnr_07[i] = metrics07[3]

        metrics05 = eval_on_image(ground_truths[i],
                                  segmentations[i],
                                  iou_ratio=0.5,
                                  return_idx=False)

        iou_05[i] = metrics05[0]
        precision_05[i] = metrics05[1]
        recall_05[i] = metrics05[2]
        fnr_05[i] = metrics05[3]

    m_iou_07, std_iou_07 = iou_07.mean(), iou_07.std()
    m_ap_07, std_ap_07 = precision_07.mean(), precision_07.std()
    m_ar_07, std_ar_07 = recall_07.mean(), recall_07.std()
    m_fnr_07, std_fnr_07 = fnr_07.mean(), fnr_07.std()

    m_iou_05, std_iou_05 = iou_05.mean(), iou_05.std()
    m_ap_05, std_ap_05 = precision_05.mean(), precision_05.std()
    m_ar_05, std_ar_05 = recall_05.mean(), recall_05.std()
    m_fnr_05, std_fnr_05 = fnr_05.mean(), fnr_05.std()

    out07 = [m_iou_07,
             std_iou_07,
             m_ap_07,
             std_ap_07,
             m_ar_07,
             std_ar_07,
             m_fnr_07,
             std_fnr_07]

    out05 = [m_iou_05,
             std_iou_05,
             m_ap_05,
             std_ap_05,
             m_ar_05,
             std_ar_05,
             m_fnr_05,
             std_fnr_05]

    return out07 + out05


def eval_on_image(ground_truth, segmentation, iou_ratio=0.7, return_idx=False):
    """evaluate instance segmentation on one image, as mean iou score of TP
    instances, precision and recall.

    Parameters:
    -----------
    ground_truth : ndarray
        ndarray of size (n_gt_instance, H, W) with ground truth instance masks.
    segmentation : ndarray
        ndarray of size (n_seg_instance, H, W) with predicted instance masks.
    iou_ratio : float, optional
        segmentation and gt masks are a match when iou score > iou_ratio.
    return_idx : boolean
        if True, return true positives indexes for both ground_truth and
        segmentation.
    """
    ground_truth = ground_truth.astype(bool)
    segmentation = segmentation.astype(bool)
    n_instance_gt = ground_truth.shape[0]
    n_instance_seg = segmentation.shape[0]

    pairwise_iou = np.zeros((n_instance_gt, n_instance_seg))
    for g in range(n_instance_gt):
        for s in range(n_instance_seg):
            pairwise_iou[g,s] = iou(ground_truth[g], segmentation[s])

    min_n_instance = min(n_instance_gt, n_instance_seg)
    idx_gt = np.zeros(min_n_instance)
    idx_seg = np.zeros(min_n_instance)
    maxiou = np.zeros(min_n_instance)
    for i in range(min_n_instance):
        maxiou[i] = pairwise_iou.max()
        r, c = np.unravel_index(np.argmax(pairwise_iou), pairwise_iou.shape)
        idx_gt[i], idx_seg[i] = r, c
        pairwise_iou[r, c] = -1

    # compute the actual evaluation metrics:
    object_iou = maxiou[maxiou > iou_ratio]
    mean_iou = object_iou.mean()
    true_pos_gt = idx_gt[maxiou > iou_ratio] # useful for viz
    true_pos_seg = idx_gt[maxiou > iou_ratio] # useful for viz

    true_pos = object_iou.size
    false_neg = n_instance_gt - true_pos
    false_pos = n_instance_seg - true_pos
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    fnr = false_neg / n_instance_gt

    if return_idx:
        out = (true_pos_gt, true_pos_seg)
    else:
        out = mean_iou, precision, recall, fnr

    return out


def dice(x,y):
    return 2 * (x & y).sum() / (x.sum() + y.sum())


def iou(x,y):
    return (x & y).sum() / (x | y).sum()


if __name__ == '__main__':
    pass
