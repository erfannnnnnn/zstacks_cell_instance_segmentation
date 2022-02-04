# -*- coding: utf-8 -*-

import numpy as np
from skimage import color, morphology, img_as_float64


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


if __name__ == '__main__':
    pass
