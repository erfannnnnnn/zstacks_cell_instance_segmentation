import os
import json
import numpy as np
import imageio
from tqdm import tqdm

from ..tools import dsutils
from ..shared import volumeio, evaluation
from . import segment


def dataset_generator(data_json_file, img_folder, stack_folder, use_stack):
    """create a dataset generator object

    Parameters:
    -----------
    data_json_file : str
        path of the json file containing the annotations
    img_folder : str
        path of the folder containing the images.
    stack_folder : str
        path of the folder containing the z-stacks
    use_stack : boolean
        if True, uses the whole z-stack, otherwise use only image input.
    """
    with open(data_json_file, 'r') as f:
        data = json.load(f)
    assets = [a for a in data['assets'] if a['set'] == 'test']
    for asset in assets:
        img_name = asset['file_name']
        img = imageio.imread(os.path.join(img_folder, img_name))
        objects = asset['annotations'][0]['data']
        gt = np.stack([dsutils.rle2mask(o['mask']) for o in objects], axis=0)
        if use_stack:
            img_name_noext = os.path.splitext(img_name)[0]
            stack_path = os.path.join(stack_folder, img_name_noext)
            stack = volumeio.load(stack_path)
        else:
            stack = None
        yield img_name, img, gt, stack


def eval(data_json_file, img_folder, stack_folder, use_stack=True):
    """perform evaluation on CISD test set and print the results

    Parameters:
    -----------
    data_json_file : str
        path of the json file containing the annotations
    img_folder : str
        path of the folder containing the images / z-stacks
    use_stack : boolean, optional
        if True, uses the whole z-stack, otherwise use only image input. Default
        is True.
    """
    data_generator = dataset_generator(
        data_json_file, img_folder, stack_folder, use_stack)

    results = {
        'iou07' : [],
        'ap07'  : [],
        'ar07'  : [],
        'fnr07' : [],
    }

    for _, img, gt, stack in tqdm(data_generator, total=978):
        cytoplasms = segment.segment(img, stack=stack, alpha=2.0, beta=20)
        if len(cytoplasms) > 0:
            seg = np.stack([cytoplasms[d] for d in cytoplasms], axis=0)
        else:
            seg = np.zeros_like(gt)
        iou07, ap07, ar07, fnr07 = evaluation.eval_on_image(
            gt, seg, iou_ratio=0.7)

        results['iou07'].append(iou07)
        results['ap07'].append(ap07)
        results['ar07'].append(ar07)
        results['fnr07'].append(fnr07)

    print('IoU@07', np.nanmean(results['iou07']))
    print('AP@07', np.mean(results['ap07']))
    print('AR@07', np.mean(results['ar07']))
    print('FNR@07', np.mean(results['fnr07']))


if __name__ == '__main__':
    pass
