import torch.nn.functional as F
import torch


def decode(locs, priors, variances):
    """
    Parameters:
    -----------
    locs : torch.Tensor
        size (n_anchors, 4)
    priors : torch.Tensor
        size (n_anchors, 4) [cy, cx, h, w]
    variances : torch.Tensor
        variances used for normalization

    Return:
    -------
    boxes : torch.Tensor
        tensor of predicted bbox locations, of size (n_anchors, 4) with boxes
        [y1, x1, y2, x2].
    """
    boxes = torch.cat([priors[:, :2] + locs[:, :2] * variances[0] * priors[:, 2:],
                       priors[:, 2:] * torch.exp(locs[:, 2:] * variances[1])], 1)
    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
    boxes[:, 3] = boxes[:, 3] + boxes[:, 1]
    return boxes

def nms(boxes, scores, nms_thresh, top_k):
    """non maximum suppression

    Parameters:
    -----------
    boxes : torch.Tensor
        tensor of size (n_pred, 4) of detected bboxes [y1, x1, y2, x2]
    scores : torch.Tensor
        tensor of size (n_pred) of confidence scores associated to each bbox
    nms_tresh : float
        non maximum suppression threshold
    top_k : int
        number max of instances considered for one class
    """
    keep = scores.new(scores.size(0)).zero_().long() # size (n_pred)
    count = 0
    if boxes.numel() == 0: # return the total elements number in boxes
        return keep, count
    y1, x1, y2, x2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3] # size (n_pred)
    area = torch.mul(y2-y1, x2-x1) # size (n_pred)
    _, idx = scores.sort(0) # size (n_pred)
    idx = idx[-top_k:] # size (top_k)

    while idx.numel() > 0:
        i = idx[-1] # index of the bbox with highest confidence
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]

        # select the remaining boxes
        yy1 = torch.index_select(y1, dim=0, index=idx) # size (idx)
        xx1 = torch.index_select(x1, dim=0, index=idx)
        yy2 = torch.index_select(y2, dim=0, index=idx)
        xx2 = torch.index_select(x2, dim=0, index=idx)

        # calculate the inter boxes clamp with box i
        yy1 = torch.clamp(yy1, min=y1[i]) # size (idx)
        xx1 = torch.clamp(xx1, min=x1[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        xx2 = torch.clamp(xx2, max=x2[i])

        w = torch.clamp(xx2 - xx1, min=0.0)
        h = torch.clamp(yy2 - yy1, min=0.0)
        inter = w * h

        rem_areas = torch.index_select(area, dim=0, index=idx)
        union = (rem_areas - inter) + area[i]
        iou = inter / union # size (idx)
        # keep only remaining bbox with iou < nms_thresh for next iterations
        idx = idx[iou.le(nms_thresh)]
    return keep, count


class Detect(object):
    def __init__(self, n_class, top_k, conf_thresh, nms_thresh, variance):
        self.n_class = n_class
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.output = torch.zeros(1, self.n_class, self.top_k, 5)

    def __call__(self, locs, confs, anchors):
        """
        Parameters:
        -----------
        locs : torch.Tensor (lives on cuda)
            tensor of bbox regression, of size (batch, n_anchors, 4).
        confs : torch.Tensor (lives on cuda)
            tensor of bbox confidence, of size (batch, n_anchors, n_class)
        anchors : torch.Tensor (lives on cpu)
            tensor of default anchors, of size (n_anchors, 4) as [cy, cx, h, w]

        Returns:
        --------
        output : torch.Tensor (cpu)
            tensor of size (batch, n_class, top_k, 5) containing the detected
            boxes [y1, x1, y2, x2] for every class after nms.
        """
        confs = F.softmax(confs, dim=2)
        num_batch = locs.size(0)

        locs  = locs.data.cpu()
        confs = confs.data.cpu()
        # should be possible to reduce output size with self.n_class - 1 -> skip background class
        output = torch.zeros(num_batch, self.n_class, self.top_k, 5) # size (batch, n_class, top_k, 5)

        # Decoding...
        for i in range(num_batch):
            decoded_boxes_i = decode(locs[i], anchors, torch.Tensor(self.variance)) # size (n_anchors, 4)
            p_conf_i = confs[i] # size (n_anchors, n_class)
            for cl in range(1, self.n_class):
                cl_mask = p_conf_i[:, cl].gt(self.conf_thresh) # gt = greater than
                p_conf_i_cl = p_conf_i[:, cl][cl_mask]
                if p_conf_i_cl.shape[0] == 0:
                    continue
                loc_mask = cl_mask.unsqueeze(1).expand_as(decoded_boxes_i)
                p_boxes_i_cl = decoded_boxes_i[loc_mask].view(-1,4)
                ids, count = nms(boxes=p_boxes_i_cl,
                                 scores=p_conf_i_cl,
                                 nms_thresh=self.nms_thresh,
                                 top_k=self.top_k)
                output[i, cl, :count] = torch.cat(
                    (p_conf_i_cl[ids[:count]].unsqueeze(1),
                     p_boxes_i_cl[ids[:count]]), 1)

        return output
