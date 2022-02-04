# -*- coding: utf-8 -*-

import numpy as np
from . import segnuclei, segclumps, segcyto


def segment(img, stack=None, cell_info=None, q=None, min_clump_size=None,
            alpha=None, beta=None):
    """segment cytoplasms intances using Phoulady et al method.

    Parameters:
    -----------
    img : ndarray
        image to be segmented.
    stack : ndarray, optional
        z-stack corresponding to img, if available.
    cell_info : dict, optional
        properties for nucleus detection.
    q : float, optional
        parameter for clump segmentation.
    min_clump_size : int, optional
        parameter defining minimal acceptable size for clumps.
    alpha & beta : floats
        parameters used to compute similarity between cell patches.
    """
    if cell_info is None:
        cell_info = segnuclei.CELL_INFO_GITHUB
    if q is None:
        q = segclumps.Q
    if min_clump_size is None:
        min_clump_size = segclumps.MIN_SIZE
    if alpha is None:
        alpha = segcyto.INFO_GITHUB['alpha']
    if beta is None:
        beta = segcyto.INFO_GITHUB['beta']

    nuclei, contour_area, contour_size = segnuclei.segment_nuclei(img,
                                                                  cell_info)
    clumps = {}
    for count in range(10):
        if clumps == {}:
            clumps = segclumps.segment_clumps(img, q, min_clump_size)
        else:
            break

    if clumps == {}: # no clumps detected
        cytoplasms = {1 : np.zeros((img.shape[0], img.shape[1]), bool)}
    else:
        cytoplasms = segcyto.segment_cytoplasm(img,
                                               contour_size,
                                               contour_area,
                                               clumps,
                                               stack)
    return cytoplasms


if __name__ == '__main__':
    pass
