# -*- coding: utf-8 -*-

import numpy as np
from skimage import color, morphology, measure, img_as_ubyte, img_as_float64
from sklearn import mixture
from scipy import signal, stats
import scipy

Q = 0.06
MIN_SIZE = 2000


def segment_clumps(img, q=Q, min_size=MIN_SIZE):
    """segment clumps

    Parameters:
    -----------
    img : 2darray
        image to be segmented
    q : float
        percentile for intensity separation between foreground and background
    min_size : int
        min clump size
    """
    if img.ndim == 3: # RGB 2 grayscale
        img = color.rgb2gray(img)
    img = img_as_ubyte(img)

    w = img_as_float64(img) # float64 for wiener filtering
    w = signal.wiener(w, (5,5))

    # clip values
    if w.max() > 1: w[w > 1] = 1
    if w.min() < 0: w[w < 0] = 0
    w = img_as_ubyte(w) # noise filtering

    i1, i2, mu1, mu2 = 0, 0, 0, 0
    # Gaussian Mixture on Pixel Values
    for count in range(10):
        if (max(i1, i2) < 150) | (mu1 == mu2):
            gm = mixture.GaussianMixture(n_components=2)
            gm.fit(w.reshape(-1, 1).astype(np.float64))
            mu1 = gm.means_[0,0]
            mu2 = gm.means_[1,0]
            i1 = stats.norm.ppf(q, loc=mu1, scale=np.sqrt(gm.covariances_[0,0,0]))
            i2 = stats.norm.ppf(q, loc=mu2, scale=np.sqrt(gm.covariances_[1,0,0]))
        else:
            break

    clump = img <= max(i1, i2)

    # threshold for removing clumps too bright
    v1 = stats.norm.ppf(.0001, loc=mu1, scale=np.sqrt(gm.covariances_[0,0,0]))
    v2 = stats.norm.ppf(.0001, loc=mu2, scale=np.sqrt(gm.covariances_[0,0,0]))
    thresh = max(v1, v2)

    # morpho postprocessing
    selem = morphology.disk(5)
    clump = morphology.binary_opening(clump, selem=selem)
    clump = morphology.remove_small_objects(clump, min_size=min_size)

    labels, n_label = measure.label(clump, return_num=True)
    rprops = measure.regionprops(labels, w)

    rm = np.array([w[labels==l].mean() for l in range(1, n_label+1)]) > thresh
    for i in np.arange(1, n_label+1)[rm]:
        clump[labels==i] = False

    # final postprocessing
    labels, n_label = measure.label(clump, return_num=True)
    clumps = {}
    for i in range(1, n_label+1):
        c = morphology.binary_dilation(labels == i, selem=morphology.disk(1))
        clumps[i] = morphology.remove_small_holes(c, area_threshold=20)

    return clumps


    if __name__ == '__main__':
        pass
