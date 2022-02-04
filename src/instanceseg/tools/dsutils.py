# -*- coding: utf-8 -*-

import numpy as np
import os
import json
import imageio
from skimage import draw
from itertools import groupby
from datetime import datetime
import random

import matplotlib.pyplot as plt

SKIP_AUTHORS = ['adminsi@vitadx.com', 'a.douibi@vitadx.com']


def dataset_from_kili(json_file, img_folder, save_path, mask_encode='polygon'):
    """
    """
    info = {'dataset_name' : 'miniCISD',
            'dataset_type': 'INSTANCE_SEGMENTATION',
            'version': '1.0.0',
            'time_stamp': str(datetime.now()),
            'description': 'mini Cell Instance Segmentation Dataset',
            'img_folder' : img_folder}

    assets, categories = assets_from_kili(json_file,
                                          img_folder,
                                          mask_encode=mask_encode)

    dataset = {'info': info,
               'assets': assets,
               'categories': categories}

    with open(save_path, 'w') as f:
        json.dump(dataset, f)


def assets_from_kili(json_file, img_folder, mask_encode='polygon'):
    """
    """
    if not mask_encode in ['polygon', 'rle']:
        raise ValueError('mask_encode should be either "polygon" or "rle", got',
                         mask_encode)
    with open(json_file, 'r') as f:
        data = json.load(f)
    assets = []
    categories = []
    for asset_id, d in enumerate(data): # iter through assets
        asset = {'asset_id' : asset_id,
                 'file_name': d['externalId'],
                 'annotations': []}
        img_annot = asset.get('annotations')

        img = imageio.imread(os.path.join(img_folder, asset['file_name']))
        h, w = img.shape[0:2]

        for label_id, label in enumerate(d['labels']): # iter through annotators
            if (not label['isLatestLabelForUser']) | label['skipped']:
                continue
            if label['author']['name'] in SKIP_AUTHORS:
                continue
            try:
                annots = label['jsonResponse']['JOB_0']['annotations']
            except KeyError:
                continue

            d_obj = []
            for annot in annots: # iter through the objects
                # get id of polygon with most vertices
                pid = np.argmax([len(p['normalizedVertices']) for p in
                                 annot['boundingPoly']])
                polygon = annot['boundingPoly'][pid]['normalizedVertices']
                cols = [w * p['x'] for p in polygon]
                rows = [h * p['y'] for p in polygon]
                binary_mask = poly2mask(rows, cols, (h,w))
                if binary_mask.sum() == 0:
                    continue
                nz = binary_mask.nonzero()
                bbox = [nz[0].min(), nz[1].min(), nz[0].max(), nz[1].max()]
                bbox = [int(b) for b in bbox] # for json to recognize dtype
                if mask_encode == 'polygon':
                    mask = {'coord_w' : cols,
                            'coord_h' : rows,
                            'size'    : (h,w)}
                else:
                    mask = mask2rle(binary_mask)

                d_obj.append({'category_name': annot['categories'][0]['name'],
                                    'bbox': bbox,
                                    'mask': mask})
                categories.append(annot['categories'][0]['name'])
            if len(d_obj) == 0:
                print('[WARNING] no valid annotations for', asset['file_name'],
                      'and annotator', label['author']['name'])
                continue
            time_stamp = ' '.join(label['createdAt'].split('T'))[:-1]
            img_annot.append({'annotation_id': label_id,
                              'data': d_obj,
                              'annotator': label['author']['name'],
                              'time_stamp': time_stamp})
        if len(img_annot) == 0:
            print('[WARNING] no valid annotations for', asset['file_name'])
            continue
        assets.append(asset)
    categories = list(set(categories))
    return assets, categories


################################################################################
# filtering functions
################################################################################


def filter_by_celltype(json_file, save_path: str,
                       rm_cell_types=['NEUTROPHIL', 'RED_BLOOD_CELL']):
    if not save_path.endswith('.json'):
        raise ValueError('save_path is expected to be a json file, got',
                         save_path)
    with open(json_file, 'r') as f:
        data = json.load(f)
    for i, asset in enumerate(data['assets']):
        objects = asset['annotations'][0]['data']
        to_remove = []
        for obj in objects:
            cell_type = obj['category_name']
            if cell_type in rm_cell_types:
                to_remove.append(obj)
        for obj in to_remove:
            objects.remove(obj)
    with open(save_path, 'w') as f:
        json.dump(data, f)


def filter_by_bbox_ratio(json_file, save_path, min_ratio=0.1):
    with open(json_file, 'r') as f:
        data = json.load(f)
    img_folder = data['info']['img_folder']
    for asset in data['assets']:
        img = imageio.imread(os.path.join(img_folder, asset['file_name']))
        img_h, img_w = img.shape[0], img.shape[1]
        objects = asset['annotations'][0]['data']
        to_remove = []
        for obj in objects:
            r0, c0, r1, c1 = obj['bbox']
            s = min((r1 - r0) / img_h, (c1 - c0) / img_w)
            if s < min_ratio:
                to_remove.append(obj)
        for obj in to_remove:
            objects.remove(obj)
    with open(save_path, 'w') as f:
        json.dump(data, f)


def _name2colo(x):
    return x.split('_')[1]

def train_test_split(json_file, save_path, train_ratio=0.75):
    if not isinstance(train_ratio, float) and 0. <= train_ratio <= 1.:
        raise ValueError('test ratio should be a float in [0,1], got:',
                         test_ratio)

    with open(json_file, 'r') as f:
        data = json.load(f)
    img_names = [asset['file_name'] for asset in data['assets']]
    all_colo = list(set(map(_name2colo, img_names)))

    r = {}
    for c in all_colo:
        r[c] = []
    for asset in data['assets']:
        r[_name2colo(asset['file_name'])].append(asset['file_name'])

    train_imgs, test_imgs = [], []
    for key in r:
        list_imgs = r[key]
        random.Random(0).shuffle(list_imgs)
        frontier = round(len(list_imgs) * train_ratio)
        train_imgs.extend(list_imgs[:frontier])
        test_imgs.extend(list_imgs[frontier:])

    for asset in data['assets']:
        if asset['file_name'] in train_imgs:
            asset['set'] = 'train'
        elif asset['file_name'] in test_imgs:
            asset['set'] = 'test'
        else:
            raise ValueError('file name not found in train nor test')

    with open(save_path, 'w') as f:
        json.dump(data, f)


################################################################################
# utils
################################################################################


def poly2mask(rows, cols, shape):
    """from polygone vertex coords output binary mask
    rows : 1darray or list
        vertex x coordinates
    cols : 1darray or list
        vertex y coordinates
    shape : tuple
        shape of the mask canvas
    """
    x, y = draw.polygon(rows, cols, shape)
    mask = np.zeros(shape, dtype=np.bool)
    mask[x, y] = True
    return mask


def mask2rle(binary_mask):
    rle = {'counts': [], 'size': binary_mask.shape}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel())):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def rle2mask(rle):
    values = [i % 2 for i in range(len(rle['counts']))]
    mask = []
    [mask.extend([x] * y) for x,y in zip(values, rle['counts'])]
    return np.array(mask, bool).reshape(rle['size'])


if __name__ == '__main__':
    pass
