# -*- coding: utf-8 -*-

import numpy as np
import csv
import os
import imageio
from skimage import color, measure, morphology, img_as_float64, img_as_ubyte
from scipy import signal
import scipy.ndimage as ndi


CELL_INFO_ISBI = {'min size': 110,
                  'min mean': 0,
                  'max mean': 200,
                  'min solidity': 0.7}

CELL_INFO_PHOULADY = {'min size' : 150,
                      'min mean' : 10,
                      'max mean' : 120,
                      'min solidity' : 0.88}

CELL_INFO_GITHUB = {'min size' : 110,
                    'min mean' : 60,
                    'max mean' : 150,
                    'min solidity' : 0.9}

CELL_INFO_CISD = {'min size' : 110,
                  'min mean' : 0,
                  'max mean' : 150,
                  'min solidity' : 0.8}

MIN_ACCEPT_BOUND_DIFF = 15
MIN_ACCEPT_BOUND_DIFF_ISBI = 5


def segment_nuclei(img, cell_info=None):
    """nucleus segmentation baseline by Phoulady et al.

    Parameters:
    -----------
    img : ndarray
        input img
    cell_info : dict
        hypothesis on nuclei size, mean and solidity
    """
    img = img_as_ubyte(img)
    if img.ndim == 3:
        img = color.rgb2gray(img)

    if cell_info is None:
        cell_info = CELL_INFO_CISD

    low_n = cell_info['min mean'] // 10 * 10
    high_n = cell_info['max mean'] // 10 * 10

    img = img_as_float64(img) # float64 for wiener filtering
    img = signal.wiener(img, (5,5))

    # clip values
    if img.max() > 1: img[img > 1] = 1
    if img.min() < 0: img[img < 0] = 0
    img = img_as_ubyte(img) # noise filtering

    nuclei = np.zeros_like(img, dtype=bool)
    n_pixels = img.size

    for thresh in range(low_n, high_n+1, 10):
        binary_img = img <= thresh
        if np.sum(binary_img) > 0.2 * n_pixels:
            break

        labels = measure.label(binary_img, connectivity=2)
        reg_props = measure.regionprops(labels, img, cache=False)

        areas = np.array([reg_props[i].area >= cell_info['min size']
                          for i in range(len(reg_props))])
        solids = np.array([reg_props[i].solidity >= cell_info['min solidity']
                               for i in range(len(reg_props))])
        add_these_regions = areas * solids
        if add_these_regions.size == 0:
            continue # no satisfactory connected component

        pixels_already_nuclei = (labels != 0) * (nuclei != 0)
        label_already_nuclei = np.unique(labels[pixels_already_nuclei])

        nuclei_labels = measure.label(nuclei, connectivity=2)
        ncl_reg_props = measure.regionprops(nuclei_labels, img, cache=False)

        if len(label_already_nuclei) != 0:
            for j in label_already_nuclei:
                intersect_with_these = np.unique(nuclei_labels[labels == j])[1:]
                assert len(intersect_with_these) > 0
                current_solid = reg_props[j-1].solidity
                previous_max_solid = max([ncl_reg_props[i-1].solidity for i in
                                          intersect_with_these])
                # tmp
                idx = np.argmax(np.array([ncl_reg_props[i-1].solidity for i in
                                          intersect_with_these]))
                if current_solid < previous_max_solid:
                    add_these_regions[j-1] = False

        for prop in np.arange(1, len(reg_props)+1)[add_these_regions]:
            nuclei[labels==prop] = True

    nuclei = ndi.morphology.binary_fill_holes(nuclei)
    nuclei = morphology.remove_small_objects(nuclei,
                                             min_size=cell_info['min size'],
                                             connectivity=2)
    nuclei = _post_process(img, nuclei)
    label, n_label = measure.label(nuclei, return_num=True)
    contour_area = {}
    contour_size = {}
    for i in range(1, n_label+1):
        contour_area[i] = label == i
        contour_size[i] = np.sum(label == i)

    return nuclei, contour_area, contour_size


def _post_process(img, nuclei, min_dist2bg=MIN_ACCEPT_BOUND_DIFF):
    """apply post processing : nuclei are dilated if dilation does not produce
    fusion, segmented regions with low

    Parameters:
    -----------
    nuclei : ndarray
        segmentation
    min_dist2bg : int
        min acceptable distance between nuclei and surrounding bg
    """

    nuclei = nuclei.astype(bool)
    struct = morphology.disk(1)
    dilated = ndi.morphology.binary_dilation(nuclei, structure=struct)

    nuclei_label, n_nuclei = measure.label(nuclei, return_num=True)
    dilated_label, n_dilated = measure.label(dilated, return_num=True)

    # replace dilated by original when nuclei fusion occurs
    for l in range(1, n_dilated+1):
        if np.unique(nuclei_label[dilated_label == l]).size > 2:
            dilated[dilated_label == l] = nuclei[dilated_label == l]
    nuclei = dilated

    labels, n_labels = measure.label(nuclei, return_num=True)
    mean_diff = np.zeros(n_labels)

    for i in range(1, n_labels + 1):
        struct = morphology.disk(3)
        dilated_nucleus = ndi.morphology.binary_dilation(labels == i,
                                                         structure=struct)
        mean_diff[i-1] = np.mean(img[dilated_nucleus & (labels == 0)]) -\
                                 np.mean(img[labels == i])
    filter_mean_diff = mean_diff < min_dist2bg
    for l in np.arange(1, n_labels+1)[filter_mean_diff]:
        nuclei[labels==l] = 0

    nuclei = _remove_edge_nuclei(nuclei)

    return nuclei


def _remove_edge_nuclei(nuclei):
    """post processing : remove detected nuclei touching the edges"""
    nuclei = nuclei.astype(bool)
    edges = np.zeros_like(nuclei, dtype=bool)
    edges[0, :] = True
    edges[-1, :] = True
    edges[:, 0] = True
    edges[:, -1] = True
    labels = measure.label(nuclei)
    labels_on_edge = np.unique(labels[edges])[1:]
    for l in labels_on_edge:
        nuclei[labels == l] = False
    return nuclei


if __name__ == '__main__':
    pass
