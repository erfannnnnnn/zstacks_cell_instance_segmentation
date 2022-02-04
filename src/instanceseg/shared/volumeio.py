# -*- coding: utf-8 -*-

import numpy as np
import imageio
import os

LIST_EXT = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']


def load(path):
    """load a volume from images present in a folder or a tiff file.

    Returns a volume of images present in a given folder or tiff file. Images
    are sorted in alphanumeric order.

    Parameters:
    -----------
    path : string
        path of the folder

    Returns:
    --------
    volume : ndarray
        volume of images
    """
    if not isinstance(path, str):
        raise TypeError('string expected, not', type(path))
    if os.path.isdir(path):
        volume = _load_from_folder(path)
    elif path.lower().endswith(('tif', 'tiff')):
        volume = imageio.volread(path)
    elif not os.path.exists(path):
        raise FileNotFoundError('file not found', path)
    else:
        raise ValueError('folder expected or tiff file, got:', path)

    return volume


def dump(volume, path):
    """Dump a volume as a list of images in a given folder.

    The volume shape should be passed as (z, x, y[, c]) for each image to
    correspond to one optical section.

    Parameters:
    -----------
    volume : ndarray
        volume to be dumped
    path : string
        save path. Path of the folder where to save the images, or path of the
        tiff file.
    """
    if not isinstance(path, str):
        raise TypeError('string expected, not', type(path))

    ext = os.path.splitext(path)[1]
    if ext in ['.tif', '.tiff']:
        imageio.volwrite(path, volume)
    elif ext == '':
        if not os.path.exists(path):
            print('makedir:', path)
            os.mkdir(path)
        elif not os.path.isdir(path):
            raise ValueError('path already exists but not a folder:', path)
        for index, image in enumerate(volume):
            name = os.path.join(path, f'img_{index:03d}.png')
            imageio.imwrite(name, image)
    else:
        raise ValueError('.tiff or .tif file expected, got', ext)


################################################################################
#####                           PRIVATE FUNCTIONS
################################################################################


def _load_from_folder(folder_path):
    """load a volume from images present in the given folder

    Images are sorted in alphanumeric order.

    Parameters:
    -----------
    folder_path : string
        path of the folder containing the images

    Returns:
    --------
    volume : ndarray
        volume loaded from the images present in the folder
    """
    list_arrays = []
    for file in sorted(os.listdir(folder_path)):
        if not os.path.splitext(file)[1].lower() in LIST_EXT:
            continue
        list_arrays.append(imageio.imread(os.path.join(folder_path, file)))

    return np.stack(list_arrays, axis=0)


if __name__ == "__main__":
    pass
