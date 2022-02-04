import torch
import torch.nn as nn
import torchvision
import scipy.ndimage as ndi


class Postprocess(object):

    def __init__(self, n_class, top_k, conf_thresh, nms_thresh):
        self.n_class = n_class
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

    def __call__(self, pos_masks, pos_scores, pos_labels):
        """apply postprocessing

        Parameters:
        -----------
        pos_masks : torch.Tensor
            tensor of size: (n_instances, h, w), containing detected instance
            masks in one sample.
        pos_scores : torch.Tensor
            tensor of size: (n_instances), containing detected instances
            confidence score for one sample.
        pos_labels : torch.Tensor
            tensor of size: (n_instances), containing detected instances class
            in one sample.
        """
        final_masks = []
        final_scores = []
        final_labels = []

        # filter instances based on a confidence threshold
        keep_idx = pos_scores.gt(self.conf_thresh)
        pos_masks = pos_masks[keep_idx]
        pos_scores = pos_scores[keep_idx]
        pos_labels = pos_labels[keep_idx]

        # filter out empty masks
        masks_area = torch.sum(pos_masks, dim=(1, 2)) # size (n_instances)
        keep_idx = masks_area.gt(0)
        pos_masks = pos_masks[keep_idx]
        pos_scores = pos_scores[keep_idx]
        pos_labels = pos_labels[keep_idx]

        # iterate through classes:
        for cl in range(1, self.n_class):
            keep_idx = pos_labels == cl
            cl_pos_masks = pos_masks[keep_idx]
            cl_pos_scores = pos_scores[keep_idx]
            cl_pos_labels = pos_labels[keep_idx]

            # sort instances by conf score and keep only top_k most confident
            _, keep_idx = torch.sort(cl_pos_scores, descending=True)
            keep_idx = keep_idx[:self.top_k]
            cl_pos_masks = cl_pos_masks[keep_idx]
            cl_pos_scores = cl_pos_scores[keep_idx]
            cl_pos_labels = cl_pos_labels[keep_idx]

            # apply non-maximum suppression
            keep_idx = nms(cl_pos_masks, cl_pos_scores, self.nms_thresh)
            cl_pos_masks = cl_pos_masks[keep_idx] # size (n_instances, h, w)
            cl_pos_scores = cl_pos_scores[keep_idx] # size (n_instances)
            cl_pos_labels = cl_pos_labels[keep_idx] # size (n_instances)

            final_masks.append(cl_pos_masks)
            final_scores.append(cl_pos_scores)
            final_labels.append(cl_pos_labels)

        final_masks = torch.cat(final_masks, dim=0)
        final_scores = torch.cat(final_scores, dim=0)
        final_labels = torch.cat(final_labels, dim=0)

        final_masks = final_masks.cpu().numpy()
        final_scores = final_scores.cpu().numpy()
        final_labels = final_labels.cpu().numpy()

        final_masks = binary_filtering(final_masks)

        return final_masks, final_scores, final_labels


def nms(masks, scores, iou_threshold):
    """
    Parameters:
    -----------
    masks : torch.Tensor
        tensor of size (n_instances, h, w), containing detected instances masks
        in one sample and for one class. Masks are sorted by score confidence.
    masks : torch.Tensor
        tensor of size (n_instances), containing detected instances scores for
        one sample and for one class.
    iou_threshold : float
        threshold on IoU for non-maximum suppression.
    """
    bboxes = masks2bboxes(masks)
    bboxes = bboxes.type(torch.float)
    keep = torchvision.ops.nms(boxes=bboxes,
                               scores=scores,
                               iou_threshold=iou_threshold)
    return keep


def masks2bboxes(masks):
    """function to convert masks into bounding boxes

    Parameters:
    -----------
    masks : torch.Tensor
        tensor of size (n_instances, h, w), containing detected instances masks
        in one sample and for one class. Masks are sorted by score confidence.
    """
    n_instances = masks.size(0)
    inst_nnz, y_nnz, x_nnz = torch.nonzero(masks, as_tuple=True)
    bboxes = torch.zeros(n_instances, 4, device=masks.device)
    for idx, n in enumerate(range(n_instances)):
        filt_idx = inst_nnz == n
        y_nnz_filtered = y_nnz[filt_idx]
        x_nnz_filtered = x_nnz[filt_idx]
        y1 = torch.min(y_nnz_filtered)
        y2 = torch.max(y_nnz_filtered)
        x1 = torch.min(x_nnz_filtered)
        x2 = torch.max(x_nnz_filtered)
        bboxes[idx] = torch.stack([y1, x1, y2, x2])
    return bboxes


def binary_filtering(pred):
    """post-process network prediction

    Parameters:
    -----------
    pred : ndarray
        array of shape (n_dets, h, w)

    Returns:
    --------
    out : ndarray
        array of shape (n_dets, h, w) after postprocessing
    """
    n_dets = pred.shape[0]
    for i in range(n_dets):
        pred[i, ...] = ndi.binary_fill_holes(pred[i, ...])
        # remove small objects and smooth contours
        pred[i, ...] = ndi.binary_opening(pred[i, ...], iterations=10)
    return pred


if __name__ == '__main__':
    pass
