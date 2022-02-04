# -*- coding: utf-8 -*-

import numpy as np
from skimage import color, morphology, img_as_float64, draw


def instance_viz(img, masks):
    """vizualize predicted instance masks

    Parameters:
    -----------
    img : ndarray
        image for superposition
    masks : list or dict
        list or dict of mask for every detected instances
    """
    out = img_as_float64(img)
    assert isinstance(masks, (dict, list, np.ndarray))
    if isinstance(masks, dict):
        masks = [masks[i] for i in masks]
    if isinstance(masks, np.ndarray):
        masks = [m for m in masks]
    n_masks = len(masks)
    if n_masks == 0:
        return out # directly return img if no mask is detected

    masks = list(map(get_binary_contours, masks))
    masks = [(i+1)*masks[i].astype('int') for i in range(n_masks)]
    mask = np.max(np.stack(masks, axis=-1), axis=-1)
    nnz = mask > 0
    mask = color.label2rgb(mask,
                           colors=get_n_colors(n_masks),
                           bg_label=0,
                           bg_color=(0,0,0))
    if out.ndim == 2:
        out = np.stack([out]*3, axis=-1)
    out[nnz] = 0.5 * out[nnz] + 0.5 * mask[nnz]
    return out


def viz_detection(input, class_bboxes, pred_boxes):
    """create detection vizualisation

    Parameters:
    -----------
    input : torch.Tensor
        input
    class_bboxes : ndarray
        ground truth bboxes for given class, of shape (n_gt, 4)
    pred_boxes : ndarray
        predicted bounding boxes for a given class, of shape (n_pred, 4)
    """
    h, w = input.size(-2), input.size(-1)
    input = input.cpu().numpy().squeeze(0)
    input = np.moveaxis(input, 0, -1) # [Channels, ...] -> [..., Channels]
    if input.ndim == 4: # ensure input is an image
        input = input[input.shape[0] // 2, ...]
    boxnorm = np.array([h, w, h, w], dtype='float')
    boxnorm = boxnorm[np.newaxis, ...]
    class_bboxes *= boxnorm
    class_bboxes = np.round(class_bboxes).astype(int)
    pred_boxes *= boxnorm
    pred_boxes = np.round(pred_boxes).astype(int)
    gt_rec = rectangles_from_bbox(class_bboxes, (h, w))
    pred_rec = rectangles_from_bbox(pred_boxes, (h, w))
    img_with_gt = instance_viz(input, gt_rec)
    img_with_pred = instance_viz(input, pred_rec)
    return np.concatenate([img_with_gt, img_with_pred], axis=1)


################################################################################
# utils
################################################################################

def get_n_colors(n):
    """get n colors

    Parameters:
    -----------
    n : int
        number of colors wanted
    """
    hsv = np.array([(i / n, 1, 0.8) for i in range(n)])
    rgb = color.hsv2rgb(hsv[np.newaxis, ...])[0, ...]
    return rgb


def get_binary_contours(x):
    """get binary contour

    Parameters:
    -----------
    x : ndarray
        binary mask
    """
    x = x.astype(bool) # ensure binary
    se = morphology.disk(4)
    dilate = morphology.binary_dilation(x, selem=se)
    erode = morphology.binary_erosion(x, selem=se)
    return dilate ^ erode


def rectangles_from_bbox(bbox, shape):
    rectangles = []
    for r0, c0, r1, c1 in bbox:
        points = draw.rectangle((r0, c0), (r1, c1), shape=shape)
        rec = np.zeros((shape[0], shape[1]), bool)
        rec[tuple(points)] = True
        rectangles.append(rec)
    return rectangles


if __name__ == '__main__':
    pass
