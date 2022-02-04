import torch
from itertools import product as product
import math
import numpy as np


class Anchors(object):
    """return anchors of type (cy, cx, h, w) normalized w.r.t the original image
    size
    """
    def __init__(self, img_height, img_width):
        super(Anchors, self).__init__()
        # min_scale = np.array([0.04, 0.1 , 0.26, 0.42])
        # max_scale = np.array([0.1 , 0.26, 0.42, 0.58])
        self.min_scale = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype='float')
        self.max_scale = np.array([0.3, 0.5, 0.7, 0.9, 1.1], dtype='float')

        self.img_size = np.array([img_height, img_width]) # array([512, 640])
        # self.pyramid_levels = [3, 4, 5, 6]
        self.pyramid_levels = [4, 5, 6, 7, 8]
        self.feat_shapes = [(self.img_size + 2 ** x - 1) // (2 ** x) for x in
                            self.pyramid_levels]
        # new feat_shapes: (32x32, 16x16, 8x8, 4x4, 2x2)
        self.steps = [2 ** x for x in self.pyramid_levels]  # [8, 16, 32, 64] : receptive field
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2]]

    def forward(self):
        coords = []
        for k, f in enumerate(self.feat_shapes):
            for i,j in product(range(f[0]), range(f[1])):
                cy = (i + 0.5) * self.steps[k] / self.img_size[0]
                cx = (j + 0.5) * self.steps[k] / self.img_size[1]

                s_k = self.min_scale[k]
                coords.append([cy, cx, s_k, s_k])

                s_k_prime = math.sqrt(s_k * self.max_scale[k])
                coords.append([cy, cx, s_k_prime, s_k_prime])

                for a_r in self.aspect_ratios[k]:
                    h = s_k / math.sqrt(a_r)
                    w = s_k * math.sqrt(a_r)
                    coords.append([cy, cx, h, w])
                    coords.append([cy, cx, w, h])

        output = torch.Tensor(coords).view(-1,4) # [n_anchors, 4]
        torch.clamp(output, min=0., max=1.)
        return output
