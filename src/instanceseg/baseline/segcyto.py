# -*- coding: utf-8 -*-

import numpy as np
from skimage import color, measure, transform, draw, morphology, img_as_ubyte, \
                    img_as_float64
from scipy import ndimage as ndi
from scipy import signal
import PIL

INFO_ISBI = {'alpha' : 2.25,
             'beta'  : 18}

INFO_GITHUB = {'alpha' : 1.75,
               'beta'  : 20}

def segment_cytoplasm(image, contour_size, contour_area, clumps, stack=None):
    """segment cytoplasm

    Parameters:
    -----------
    image : 2darray
        image to be segmented.
    contour_size : dict
        contain the number of pixels in every detected nucleus.
        keys go from 1 to n_nucleus.
    contour_area : dict
        contain boolean mask of every detected nucleus.
        keys go from 1 to n_nucleus.
    clumps : dict
        contain boolean mask of every detected clumps.
        keys go from 1 to n_clumps.
    stack : ndarray, optional
        associated z-stack.
    """
    img = image.copy()
    alpha = INFO_GITHUB['alpha']
    beta = INFO_GITHUB['beta']

    if img.ndim == 3: # RGB 2 grayscale
        img = color.rgb2gray(img)
    img = img_as_ubyte(img)

    grid_width = 8 # approx grid
    nucleus_gs_ratio = 0.1 # nucleus in grid case if covers more than 0.1 * case
    # include_detected_nucleus[n] == k means nucleus n is in clump k
    include_detected_nucleus = np.zeros(len(contour_area), dtype='int')
    for i in range(1, len(include_detected_nucleus) + 1):
        for j in range(1, len(clumps) + 1):
            if np.sum(contour_area[i] * clumps[j]) >= 0.7 * contour_size[i]:
                include_detected_nucleus[i-1] = j
                break

    square_area = grid_width * grid_width
    cytoplasms = {}
    canvas = np.zeros(clumps[1].shape, dtype=bool)

    for c in clumps: # first cytoplasm approximation
        nuclei_in_clump = (include_detected_nucleus == c).nonzero()[0] + 1 # nuclei in clump c
        nb_nuclei = nuclei_in_clump.size
        if nb_nuclei == 0: # skip clump if does not contains nucleus
            continue
        prop = measure.regionprops(clumps[c].astype('int'))
        # minus and plus 1 to match matlab bbox
        W1 = max(prop[0].bbox[0] - 1 - grid_width, 0) # min row
        H1 = max(prop[0].bbox[1] - 1 - grid_width, 0) # min col
        W2 = min(prop[0].bbox[2] + 1 + grid_width, canvas.shape[0])
        H2 = min(prop[0].bbox[3] + 1 + grid_width, canvas.shape[1])

        if W1 == 0:
            W2 = W2 - ((W2 - W1) % grid_width)
        else:
            W1 = W1 + ((W2 - W1) % grid_width)

        if H1 == 0:
            H2 = H2 - ((H2 - H1) % grid_width)
        else:
            H1 = H1 + ((H2 - H1) % grid_width)

        nuclei_effect = np.zeros(((W2 - W1) // grid_width,
                                  (H2 - H1) // grid_width,
                                  nb_nuclei))

        if stack is not None: # only is stack is available
            grid_focus = np.zeros((stack.shape[0],
                                   (W2 - W1) // grid_width,
                                   (H2 - H1) // grid_width))
            vec = np.zeros(stack.shape[0])
            for i in range((W2 - W1) // grid_width):
                for j in range((H2 - H1) // grid_width):
                    for k in range(stack.shape[0]):
                        x1 = i * grid_width + W1
                        x2 = (i+1) * grid_width + W1
                        y1 = j * grid_width + H1
                        y2 = (j+1) * grid_width + H1
                        tmp_gs = img_as_float64(stack[k, x1:x2, y1:y2])
                        vec[k] = tmp_gs.std()
                    if vec.sum() != 0: # norm between 0 and 1
                        vec = (vec - vec.min()) / (vec.max() - vec.min())
                    grid_focus[:, i, j] = vec

        nucleus_gridsquare = {}
        nucleus_center_coord = {}
        nucleus_center_coord_origin = {}

        for n in range(nb_nuclei):
            ncl_in_clump_area = contour_area[nuclei_in_clump[n]][W1:W2, H1:H2] # nucleus mask in clump bbox
            n_props = measure.regionprops(ncl_in_clump_area.astype(np.uint8)) # nucleus properties
            c0 = int(np.round(n_props[0].centroid[0])) # nucleus centroid coords
            c1 = int(np.round(n_props[0].centroid[1]))
            nucleus_center_coord_origin[nuclei_in_clump[n]] = (c0, c1)
            c0 = int(np.ceil(n_props[0].centroid[0] // grid_width)) # ncl centroid in grid ref
            c1 = int(np.ceil(n_props[0].centroid[1] // grid_width))
            nucleus_center_coord[nuclei_in_clump[n]] = (c0, c1)

            # looking for grid cases containing nucleus n:
            p =  0
            nucleus_gridsquare[nuclei_in_clump[n]] = {}
            for i in range((W2 - W1) // grid_width):
                for j in range((H2 - H1) // grid_width): # iter through grid cases
                    x1, x2 = i * grid_width, (i+1) * grid_width
                    y1, y2 = j * grid_width, (j+1) * grid_width
                    area = np.sum(ncl_in_clump_area[x1:x2, y1:y2])
                    if area > nucleus_gs_ratio * square_area:
                        nucleus_gridsquare[nuclei_in_clump[n]][p] = [i, j]
                        p += 1
            if len(nucleus_gridsquare[nuclei_in_clump[n]]) == 0: # at least one case should be found
                raise ValueError('ERROR')

            # for all cases, compute likelihood to belong to nucleus n cytoplasm
            for i in range((W2 - W1) // grid_width):
                for j in range((H2 - H1) // grid_width): # iter through grid cases
                    for k in nucleus_gridsquare[nuclei_in_clump[n]]: # iter through cases with bit of nucleus n
                        t1 = i-nucleus_gridsquare[nuclei_in_clump[n]][k][0]
                        t2 = j-nucleus_gridsquare[nuclei_in_clump[n]][k][1]
                        add = np.linalg.norm([t1, t2], 2)**2 / (2 * alpha**2)
                        add = np.exp(-add)
                        if stack is not None: # when stack is given
                            ink = nucleus_gridsquare[nuclei_in_clump[n]][k][0]
                            jnk = nucleus_gridsquare[nuclei_in_clump[n]][k][1]
                            dif = (grid_focus[:,i,j] - grid_focus[:,ink,jnk])**2
                            stack_sim = np.sqrt(np.sum(dif))**2 / (2 * alpha**2)
                            stack_sim = np.exp(-stack_sim)
                            add = add * stack_sim
                        nuclei_effect[i,j,n] = nuclei_effect[i,j,n] + add
                    norm = len(nucleus_gridsquare[nuclei_in_clump[n]])
                    nuclei_effect[i,j,n] = nuclei_effect[i,j,n] / norm

        sum_nuclei_effect = np.sum(nuclei_effect, 2)

        pil_img = PIL.Image.fromarray(clumps[c][W1:W2, H1:H2])
        size = ((H2 - H1) // grid_width, (W2 - W1) // grid_width) # inverted in PIL
        shrinked_clump = np.array(pil_img.resize(size,
                                                 resample=PIL.Image.BICUBIC),
                                  dtype=bool)

        for n in range(nb_nuclei):
            xn = nucleus_center_coord[nuclei_in_clump[n]][0] # nuclei center x
            yn = nucleus_center_coord[nuclei_in_clump[n]][1] # nuclei center y
            this_nuclei_effect = beta * nuclei_effect[:,:,n] - sum_nuclei_effect
            cellblobs = ndi.morphology.binary_fill_holes(this_nuclei_effect > 0)
            cellblobs = morphology.binary_opening(cellblobs & shrinked_clump,
                                                  selem=morphology.diamond(1))

            if not cellblobs[xn, yn]: # needs to be True
                cellblobs[xn, yn] = True

            need_continue = True
            while need_continue:
                need_continue = False
                pix_x, pix_y = cellblobs.nonzero() # all cyto coord
                for i in range(len(pix_x)): # for every case, compute direct line toward ncl centroid
                    path_x, path_y = draw.line(pix_x[i], pix_y[i], xn, yn)
                    if not np.all(cellblobs[path_x,path_y]): # if direct line outside cyto, delete case
                        cellblobs[pix_x[i], pix_y[i]] = 0
                        need_continue = True

            # Le cytoplasme associé a ce noyau est sauvegardé.
            cytoplasms[nuclei_in_clump[n]] = np.copy(canvas)
            pil_img = PIL.Image.fromarray(cellblobs)
            size = (cellblobs.shape[1]*grid_width,cellblobs.shape[0]*grid_width) # inverted in PIL
            cyto = np.array(pil_img.resize(size,
                                           resample=PIL.Image.BICUBIC),
                            dtype=bool)
            cyto = cyto & clumps[c][W1:W2, H1:H2]

            label_cellblobs = measure.label(cyto, connectivity=1)
            cyto = label_cellblobs == label_cellblobs[nucleus_center_coord_origin[nuclei_in_clump[n]][0],
                                                      nucleus_center_coord_origin[nuclei_in_clump[n]][1]]
            cytoplasms[nuclei_in_clump[n]][W1:W2, H1:H2] = cyto
            # if label_cellblobs[nucleus_center_coord_origin[nuclei_in_clump[n]][0],
            #                    nucleus_center_coord_origin[nuclei_in_clump[n]][1]] == 0:
                # raise ValueError('Error : one nucleus has no cytoplasm',
                #                  'try to decrease grid_width.')

    cytoplasms = _postprocess(img,
                              cytoplasms,
                              contour_area,
                              include_detected_nucleus,
                              clumps)
    return cytoplasms


def _postprocess(img, cytoplasms, contour_area, include_detected_nucleus, clumps):
    """second cytoplasm boundaries refinement

    Parameters:
    -----------
    image : 2darray
        image to be segmented.
    cytoplasms : dict
        dict of the first cytoplasms prediction
    contour_area : dict
        contain boolean mask of every detected nucleus.
        keys go from 1 to n_nucleus.
    include_detected_nucleus : 1d array
        table indicating in what clump a nucleus belong.
    clumps : dict
        contain boolean mask of every detected clumps.
        keys go from 1 to n_clumps.
    """
    keep_keys = np.array(list(contour_area.keys()))[include_detected_nucleus!=0] # nucleus in a clump
    cytoplasms = {k: cytoplasms[k] for k in keep_keys}
    contour_area = {k: contour_area[k] for k in keep_keys}
    include_detected_nucleus = include_detected_nucleus[include_detected_nucleus!=0]

    # meshgrid creation
    d, angle_sin = np.meshgrid(np.arange(0, 301),
                               np.sin(np.arange(0, 2*np.pi, 2*np.pi/360)))
    x_inc = np.round(d * angle_sin).astype('int')
    d, angle_cos = np.meshgrid(np.arange(0, 301),
                               np.cos(np.arange(0, 2*np.pi, 2*np.pi/360)))
    y_inc = np.round(d * angle_cos).astype('int')

    cyto_changing = np.ones(len(cytoplasms), dtype=bool) # True is cyto evolve
    for X in range(20): # multiple refinement iter
        if not np.any(cyto_changing): # break no cyto left evolving
            break
        if X == 0: # first iter
            for i in contour_area:
                expanded = morphology.binary_dilation(contour_area[i],
                                                      selem=morphology.disk(7))
                expanded[contour_area[i]] = False
                mask = morphology.binary_dilation(contour_area[i],
                                                  selem=morphology.disk(1))
                img[mask] = np.round(img[expanded].mean())

            tab = np.stack([contour_area[idx] for idx in contour_area], axis=-1)
            label_cellblobs = measure.label(np.sum(tab, axis=-1),
                                            connectivity=1)
            props = measure.regionprops(label_cellblobs, img)
            minv = int(min([props[idx].mean_intensity for idx in range(len(props))]))
            img = np.maximum(img, minv)

            img = img_as_float64(img) # float64 for wiener filtering
            img = img_as_ubyte(signal.wiener(img, (5,5)))
            # switch btw float64 to uint8 for filter and similarity with matlab
            img = img_as_float64(img)
            if img.max() > 1: img[img > 1] = 1
            if img.min() < 0: img[img < 0] = 0

            tab = np.stack([clumps[idx] for idx in clumps], axis=-1)
            foreground = np.sum(tab, axis=-1).astype(bool)
            img[~foreground] = 1 # background is set constant to 1

        a = 10
        c = .5

        for i, key in enumerate(cytoplasms):
            if not cyto_changing[i]:
                continue
            cytoprops = measure.regionprops(cytoplasms[key].astype(int))
            nucleusprops = measure.regionprops(contour_area[key].astype(int))

            ncl_x = int(np.round(nucleusprops[0].centroid[0])) # nucleus centroid
            ncl_y = int(np.round(nucleusprops[0].centroid[1])) # nucleus centroid

            x1 = cytoprops[0].bbox[0] # window containing the cytoplasm
            y1 = cytoprops[0].bbox[1]
            x2 = cytoprops[0].bbox[2] - 1
            y2 = cytoprops[0].bbox[3] - 1

            all_x = ncl_x - x_inc
            all_y = ncl_y + y_inc

            in_x = (all_x < img.shape[0]) & (all_x >= 0)
            in_y = (all_y < img.shape[1]) & (all_y >= 0)
            in_image = in_x & in_y
            in_cyto = (all_x <= x2) & (all_x >= x1) & (all_y <= y2) & (all_y >= y1)
            inrange_idx = np.ravel_multi_index(np.vstack([all_x[in_cyto],
                                                          all_y[in_cyto]]),
                                               img.shape)
            values = np.vstack([np.argwhere(cytoplasms[key])[:,0],
                                np.argwhere(cytoplasms[key])[:,1]])
            pixels = np.ravel_multi_index(values, cytoplasms[key].shape)
            in_cyto[in_cyto] = np.in1d(inrange_idx, pixels)

            all_boundary_pts = np.zeros((360, 2), dtype=int) # cyto bound in centroid referential
            previous_boundary_pts = np.zeros((360, 2), dtype=int)
            include_boundary_pts = np.ones(360, dtype=bool) # points to include
            changed_boundary_pts = np.ones(360, dtype=bool) # modified points

            for r in range(360): # for every angle, look where is cyto boundary
                first_pix_out = np.nonzero(~in_cyto[r, :])[0] # first pixel out of cyto along r
                if first_pix_out.size > 0:
                    first_pix_out = first_pix_out[0]
                else:
                    first_pix_out = in_cyto.shape[1]
                last_pix_cyto = first_pix_out - 1 # last pixel in cyto
                first_pix_out_img = np.nonzero(~in_image[r, :])[0]
                if first_pix_out_img.size > 0:
                    first_pix_out_img = first_pix_out_img[0]

                if X == 0:
                    ray_length = min(2 * (last_pix_cyto + 1), all_x.shape[1])
                    if first_pix_out_img.size > 0:
                        ray_length = min(ray_length, first_pix_out_img)
                    ray_weight = abs(np.arange(1, ray_length + 1) - (last_pix_cyto + 1))
                    ray_weight = 1 - ray_weight / (last_pix_cyto + 1)

                else:
                    ray_length = last_pix_cyto + 1
                    if first_pix_out_img.size > 0:
                        ray_length = min(ray_length, first_pix_out_img)
                    ray_weight = np.arange(1, ray_length + 1) / ray_length

                ray_weight = 1 / (1 + np.exp(-a * (ray_weight - c)))

                ray_x = all_x[r, np.arange(ray_length)] # ray x coord
                ray_y = all_y[r, np.arange(ray_length)] # ray y coord

                previous_boundary_pts[r, 0] = all_x[r, last_pix_cyto] # cyt bound before transform
                previous_boundary_pts[r, 1] = all_y[r, last_pix_cyto] # cyt bound before transform

                ray_pixels = img[ray_x.astype(int), ray_y.astype(int)] # pixels intensity along ray

                tmp = [0] + list(ray_pixels) + [1] # diff between neighbors pixels
                diff = np.zeros(len(ray_pixels))
                for idx in range(1, len(tmp) - 1):
                    diff[idx - 1] = tmp[idx - 1] - tmp[idx + 1] # why sum is needed here ???

                pos = np.argmin(ray_weight * diff) # position of pixel with highest intensity variation

                if ray_length - pos <= 5:
                    changed_boundary_pts[r] = False

                all_boundary_pts[r, :] = ray_x.astype(int)[pos], ray_y.astype(int)[pos]

            all_boundary_pts = all_boundary_pts[include_boundary_pts]

            poly_order = 3
            fit_w_width = 2 * (all_boundary_pts.shape[0] // 8) + 1 # window for transform
            fit_margin = fit_w_width // 2 # transform margin

            conc_x = np.concatenate((all_boundary_pts[-fit_margin:, 0],
                                     all_boundary_pts[:, 0],
                                     all_boundary_pts[:fit_margin, 0]))
            conc_y = np.concatenate((all_boundary_pts[-fit_margin:, 1],
                                     all_boundary_pts[:, 1],
                                     all_boundary_pts[:fit_margin, 1]))
            # smooth cyto bounds
            smooth_x = signal.savgol_filter(conc_x, fit_w_width, poly_order)
            smooth_y = signal.savgol_filter(conc_y, fit_w_width, poly_order)
            smooth_x = smooth_x[fit_margin:-fit_margin]
            smooth_y = smooth_y[fit_margin:-fit_margin]

            dist_smoothbound = np.stack([smooth_x, smooth_y], axis=1)
            dist_smoothbound = dist_smoothbound - all_boundary_pts
            dist_val = np.sqrt(np.sum(dist_smoothbound**2, axis=1))

            new_bound_points = all_boundary_pts[dist_val <= 10] # points that moved only a little

            fit_w_width = 2 * (new_bound_points.shape[0] // 16) + 1 # window for transform
            fit_margin = fit_w_width // 2 # transform margin

            conc_x = np.concatenate((new_bound_points[-fit_margin:, 0],
                                     new_bound_points[:, 0],
                                     new_bound_points[:fit_margin, 0]))
            conc_y = np.concatenate((new_bound_points[-fit_margin:, 1],
                                     new_bound_points[:, 1],
                                     new_bound_points[:fit_margin, 1]))
            smooth_x = signal.savgol_filter(conc_x, fit_w_width, poly_order)
            smooth_y = signal.savgol_filter(conc_y, fit_w_width, poly_order)
            smooth_x = smooth_x[fit_margin:-fit_margin]
            smooth_y = smooth_y[fit_margin:-fit_margin]

            new_cyto = poly2mask(smooth_x, smooth_y, img.shape)

            add_x = np.round(smooth_x).astype(int)
            add_y = np.round(smooth_y).astype(int)
            add_x = np.minimum(add_x, new_cyto.shape[0]-1)
            add_y = np.minimum(add_y, new_cyto.shape[1]-1)
            add_x = np.maximum(add_x, 0)
            add_y = np.maximum(add_y, 0)
            new_cyto[add_x, add_y] = True

            new_cyto = new_cyto & clumps[include_detected_nucleus[i]]
            new_cyto = contour_area[key] | new_cyto
            new_pre_diff = (new_cyto & ~cytoplasms[key]) | (~new_cyto & cytoplasms[key])

            if (np.sum(new_pre_diff) / np.sum(cytoplasms[key]) < 0.01):
                cyto_changing[i] = False
            cytoplasms[key] = new_cyto

    return cytoplasms


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


if __name__ == '__main__':
    pass
