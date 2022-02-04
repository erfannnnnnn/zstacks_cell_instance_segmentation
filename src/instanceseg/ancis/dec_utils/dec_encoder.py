import torch


def encode(match_boxes, priors, variances):
    """encode matched gt boxes with anchors

    Parameters:
    -----------
    match_boxes : torch.Tensor of shape (n_anchors, 4)
        boxes represent corners (y1, x1, y2, x2)
    priors : torch.Tensor of shape (n_anchors, 4)
        anchors represent corners (cy, cx, h, w)
    variances : list of float
        variances to normalize cy, cx, h and w distributions
    """
    c_yx = (match_boxes[:, :2] + match_boxes[:, 2:]).float() / 2
    c_yx = c_yx - priors[:, :2]
    c_yx = c_yx.float() / (variances[0] * priors[:,2:])

    hw = (match_boxes[:, 2:] - match_boxes[:, :2]).float() / priors[:,2:]
    hw = torch.log(hw.float())/variances[1]

    return torch.cat([c_yx, hw], 1)


def split_to_box(priors):
    """take anchor of shape (cy, cx, h, w) and return box (y1, x1, y2, x2)"""
    return torch.cat([priors[:, :2] - priors[:, 2:] / 2,
                      priors[:, :2] + priors[:, 2:] / 2], 1)


def intersect(boxes_a, boxes_b):
    """compute intersection between bboxes

    Parameters:
    -----------
    boxes_a : torch.Tensor of shape (num_a, 4)
        bbox represent corners : (y1, x1, y2, x2)
    boxes_b : torch.Tensor of shape (num_b, 4) (y1, x1, y2, x2)
        bbox represent corners : (y1, x1, y2, x2)

    Return:
    -------
    inter : torch.Tensor of shape (num_a, num_b)
        tensor of intersection area between bboxes a and b.
    """
    num_a = boxes_a.size(0)
    num_b = boxes_b.size(0)
    max_xy = torch.minimum(boxes_a[:,2:].unsqueeze(1).expand(num_a,num_b,2),
                           boxes_b[:,2:].unsqueeze(0).expand(num_a,num_b,2))
    min_xy = torch.maximum(boxes_a[:,:2].unsqueeze(1).expand(num_a,num_b,2),
                           boxes_b[:,:2].unsqueeze(0).expand(num_a,num_b,2))

    inter = torch.clamp((max_xy - min_xy), min=0.)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(boxes_a, boxes_b):
    """compute IoU between boxes_a and boxes_b

    Parameters:
    -----------
    boxes_a : torch.Tensor of shape (n_box_a, 4)
        bbox represent corners : (y1, x1, y2, x2)
    boxes_b : torch.Tensor of shape (n_box_b, 4)
        bbox represent corners : (y1, x1, y2, x2)

    Return:
    -------
    IoU (aka Jaccard index) : torch.Tensor of shape (n_box_a, n_box_b)
    """
    inter = intersect(boxes_a, boxes_b) # size (n_box_a, n_box_b)
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1]))
    area_a = area_a.unsqueeze(1).expand_as(inter)
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1]))
    area_b = area_b.unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union


def encode_gt_to_anchors(gt_boxes, gt_label, anchors, match_thresh, variances):
    """encode ground truth as objective w.r.t anchors and anchors' labels.

    Parameters:
    -----------
    gt_boxes : torch.FloatTensor of shape (n_gt_box, 4)
        boxes represent bbox corners: (y1, x1, y2, x2).
    gt_label : torch.Tensor of shape (n_gt_box)
        label of the class in every gt bbox.
    anchors : torch.Tensor of shape (n_anchors, 4)
        anchors are of type: (center_y, center_x, h, w).
    match_tresh : float
        Jaccard index min value to consider a anchors a match with a gt box.
    variances : list of float
        variances to normalize cy, cx, h and w distributions

    Return:
    -------
    encoded_anchor_bboxes : torch.Tensor of shape (n_anchors, 4)
        bbox of type (cy', cx', h', w') with:
            cy' = (cy_gt - cy_anchor) / h_anchor / sigma_xy
            cx' = (cx_gt - cx_anchor) / w_anchor / sigma_xy
            h' = log(h_gt / h_anchor) / sigma_hw
            w' = log(w_gt / w_anchor) / sigma_hw
    encoded_anchor_labels : torch.Tensor of shape (n_anchors)
        label of each anchor (0 if background).
    """
    anchors_box = split_to_box(anchors)
    overlaps = jaccard(gt_boxes, anchors_box) # size (n_gt_box, n_anchors)

    best_gt, best_gt_idx = overlaps.max(0, keepdim=True)
    best_gt.squeeze_(0) # size (n_anchors)
    best_gt_idx.squeeze_(0) # size (n_anchors)

    best_anchor, best_anchor_idx = overlaps.max(1, keepdim=True)
    best_anchor.squeeze_(1) # size (n_gt_box)
    best_anchor_idx.squeeze_(1) # size (n_gt_box)

    # IoU of 1 for the anchors matching best the gt boxes
    # to prevent issues with large match_tresh and poor default anchors
    best_gt.index_fill_(0, best_anchor_idx, 1)
    # ensure each gt box is at least the best for one anchor
    for j in range(best_anchor_idx.size(0)): # iterate n_gt_box
        best_gt_idx[best_anchor_idx[j]] = j

    match_boxes = gt_boxes[best_gt_idx] # size (n_anchors, 4)
    encoded_anchor_labels = gt_label[best_gt_idx] # size (n_anchors)
    encoded_anchor_labels[best_gt < match_thresh] = 0.
    encoded_anchor_bboxes = encode(match_boxes, anchors, variances)

    return encoded_anchor_bboxes, encoded_anchor_labels.unsqueeze(1)


if __name__ == '__main__':
    pass
