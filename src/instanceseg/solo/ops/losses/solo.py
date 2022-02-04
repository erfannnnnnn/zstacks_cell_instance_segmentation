import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, reduction=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        """Compute the focal loss between `logits` and the ground truth `labels`.
            Focal loss = -alpha_t * (1-pt) ^ gamma * log(pt)
         where pt is the probability of being classified to the true class.
          pt = p(if true class), otherwise pt = 1 - p. p = sigmoid(logit).
           Args:
                labels: A float tensor of size[batch, num_classes].
                logits: A float tensor of size[batch, num_classes].
                alpha: A float tensor of size[batch_size]
                 specifying per-example weight for balanced cross entropy.
                gamma: A float scalar modulating loss from hard and easy examples.
            Returns:
                focal_loss: A float32 scalar representing normalized total loss
        """
        bce = F.binary_cross_entropy_with_logits(
            input=input, target=target, reduction='none')

        if self.gamma == 0.0:
            modulator = 1.0
        else:
            modulator = torch.exp(-self.gamma * target * input - \
                self.gamma * torch.log(1 + torch.exp(-1.0 * input)))

        loss = modulator * bce
        return loss


class DiceLoss(nn.Module):

    def __init__(self, reduction=None, smooth=1.):
        super(DiceLoss, self).__init__()
        self.reduction = reduction
        self.smooth = smooth

    def forward(self, output_masks, target_masks):
        """Computes the dice loss

        Args:
            output_masks (torch.Tensor): output masks with shape
                (N, h, w) or (N, h, w, 1)
            target_masks (torch.Tensor): target masks with shape
                (N, h, w) or (N, h, w, 1)

        Returns:
            torch.Tensor: dice loss
        """
        num_masks, h, w = output_masks.shape
        output_masks = output_masks.view(num_masks, -1)
        target_masks = target_masks.view(num_masks, -1)
        num = 2 * (output_masks * target_masks).sum(dim=-1) + self.smooth
        den = output_masks.sum(dim=-1) + target_masks.sum(dim=-1) + self.smooth

        dice_loss = 1 - num / den
        if self.reduction == 'mean':
            dice_loss = dice_loss.mean()
        elif self.reduction == 'sum':
            dice_loss = dice_loss.sum()
        return dice_loss


################################################################################
# modified SOLO loss
################################################################################


class SOLOLoss(nn.Module):

    def __init__(self,
                 num_classes,
                 pyramid_levels,
                 scale_ranges=((0, 0.75), (0.25, 1)),
                 epsilon=0.2,
                 lambda_mask=3,
                 gamma=0):
        """init SOLO loss

        Parameters:
        -----------
        num_classes : int
            number of classes, including background class.
        pyramid_levels : list of int
            list of the FPN pyramid levels used for prediction with solo head.
            Example: pyramid_levels=[0, 1] if the two first FPN levels are used.
        scale_range : tuple of tuple
            For each pyramid level, the scale ranges of the instances to be
            encoded. Scale are given in ratio w.r.t the original image size.
        epsilon : float, optional
            Default is 0.5.
        lambda_mask : int, optional
            Default is 3.
        gamma : int / float ?
            Default is 0.
        """
        super(SOLOLoss, self).__init__()
        assert len(pyramid_levels) == len(scale_ranges)
        self.num_classes = num_classes
        self.pyramid_levels = pyramid_levels
        self.scale_ranges = scale_ranges
        self.epsilon = epsilon
        self.focal_loss = FocalLoss(reduction=None, gamma=gamma)
        self.dice_loss = DiceLoss()
        self.lambda_mask = lambda_mask

    def forward(self, output, gt_bboxes_raw, gt_labels_raw, gt_masks_raw):
        """forward pass for SOLO Loss

        Parameters:
        -----------
        output : tuple of tensors
        target : list of tuple
            contains (bboxes, labels and masks)
        """
        pred_cat, pred_ins = output
        gt_cat, gt_ins = self.get_gt(
            output, gt_bboxes_raw, gt_labels_raw, gt_masks_raw)

        # Count all grid  cell whose label is different than 0 (background)
        masks_loss = []
        cat_loss = []

        for pyramid_level in self.pyramid_levels: # loop over pyramid levels
            gt_cat_lvl = gt_cat[pyramid_level]
            pred_cat_lvl = pred_cat[pyramid_level]
            gt_masks_lvl = gt_ins[pyramid_level]
            pred_masks_lvl = pred_ins[pyramid_level]

            # Reshape gt categories (N, 1, S, S) -> (N*S**2)
            gt_cat_lvl = gt_cat_lvl.flatten()
            pos_ind = gt_cat_lvl > 0
            gt_cat_lvl = F.one_hot(gt_cat_lvl, num_classes=self.num_classes) # new shape (N*S**2, n_class)

            # reshape pred categories (N, n_class, S, S) -> (N*S**2, n_class)
            pred_cat_lvl = pred_cat_lvl.permute(0, 2, 3, 1)
            pred_cat_lvl = pred_cat_lvl.reshape((-1, pred_cat_lvl.size(-1)))

            # Compute category loss
            cat_loss.append(self.focal_loss(pred_cat_lvl, gt_cat_lvl.float()))

            # Reshape gt and pred masks (N, S**2, h', w') --> (N*S**2, h', w')
            _, _, mask_h, mask_w = gt_masks_lvl.shape
            gt_masks_lvl = gt_masks_lvl.view(-1, mask_h, mask_w)
            pred_masks_lvl = pred_masks_lvl.view(-1, mask_h, mask_w)

            # Filtered pos gt instances and computes mask loss
            num_pos = len(pos_ind.nonzero().flatten())
            if num_pos > 0:
                gt_pos_masks = gt_masks_lvl[pos_ind]
                pred_pos_masks = pred_masks_lvl[pos_ind]
                masks_loss.append(
                    self.dice_loss(pred_pos_masks, gt_pos_masks))

        # aggregate losses and return total loss
        cat_loss = torch.cat(cat_loss).mean()
        masks_loss = self.lambda_mask * torch.cat(masks_loss).mean()
        total_loss = cat_loss + masks_loss

        return total_loss, cat_loss, masks_loss


    def compute_areas(self, bboxes):
        # Compute side of each object
        bboxes_h = bboxes[:, 2] - bboxes[:, 0]
        bboxes_w = bboxes[:, 3] - bboxes[:, 1]
        areas = torch.sqrt(bboxes_h * bboxes_w)
        return areas

    def get_gt(self, output, gt_bboxes_raw, gt_labels_raw, gt_masks_raw):
        """encode ground truth masks and labels to match SOLO paradigm"""
        # define output:
        dict_level_cate_labels = {lvl: [] for lvl in self.pyramid_levels}
        dict_level_ins_labels = {lvl: [] for lvl in self.pyramid_levels}

        cats, masks = output
        device = masks[0].device
        # cats -> contains vector of size [batch, n_class, num_grid, num_grid]
        # masks -> contains vectors of size [batch, num_grid**2, h', w']
        batch_size = len(gt_bboxes_raw)
        for i in range(batch_size): # loop over batch size
            gt_bboxes_batch = gt_bboxes_raw[i] # tensor of size [n_instance, 4]
            gt_labels_batch = gt_labels_raw[i] # tensor of size [n_instance]
            gt_masks_batch = gt_masks_raw[i] # tensor of size [n_instance, H, W]
            gt_areas = self.compute_areas(gt_bboxes_batch)

            for pyramid_level in self.pyramid_levels: # loop over pyramid levels
                lower_bound, upper_bound = self.scale_ranges[pyramid_level]
                hit_indices = ((gt_areas >= lower_bound) &
                    (gt_areas <= upper_bound)).nonzero().flatten()
                # number of instance to be encoded at this level:
                num_instance = len(hit_indices)

                mask_h, mask_w = masks[pyramid_level].size()[-2:]
                num_grid = cats[pyramid_level].size()[-1]

                ins_label = torch.zeros([1, num_grid, num_grid, mask_h, mask_w],
                                        dtype=torch.uint8, device=device)
                cate_label = torch.zeros([1, 1, num_grid, num_grid],
                                         dtype=torch.int64, device=device)

                if num_instance > 0:
                    gt_bboxes = gt_bboxes_batch[hit_indices, ...]
                    gt_masks = gt_masks_batch[hit_indices, ...]
                    gt_labels = gt_labels_batch[hit_indices]

                    # Compute center regions of instances
                    yc = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2
                    xc = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2
                    half_h = 0.5*(gt_bboxes[:,2]-gt_bboxes[:,0])*self.epsilon
                    half_w = 0.5*(gt_bboxes[:,3]-gt_bboxes[:,1])*self.epsilon

                    tops = torch.floor((yc - half_h) * num_grid)
                    tops = torch.clamp(tops, min=0, max=num_grid-1)
                    bots = torch.floor((yc + half_h) * num_grid)
                    bots = torch.clamp(bots, min=0, max=num_grid-1)
                    lefts = torch.floor((xc - half_w) * num_grid)
                    lefts = torch.clamp(lefts, min=0, max=num_grid-1)
                    rights = torch.floor((xc + half_w) * num_grid)
                    rights = torch.clamp(rights, min=0, max=num_grid-1)

                    for j in range(num_instance): # loop over instances
                        top = tops[j].type(torch.long)
                        bot = bots[j].type(torch.long)
                        left = lefts[j].type(torch.long)
                        right = rights[j].type(torch.long)

                        cate_label[..., top:bot+1, left:right+1] = gt_labels[j] # size (1, 1, 32, 32)

                        ins_mask = F.interpolate(
                            gt_masks[j].unsqueeze(0).unsqueeze(0),
                            size=(mask_h, mask_w),
                            mode='bilinear',
                            align_corners=True)
                        ins_mask = ins_mask.squeeze(0).squeeze(0)
                        ins_label[:, top:bot+1, left:right+1, ...] = ins_mask

                # Reshape instance masks (1, S, S, h', w') -> (1, S**2, h', w')
                ins_label = torch.reshape(ins_label, (1, -1, mask_h, mask_w))

                dict_level_cate_labels[pyramid_level].append(cate_label)
                dict_level_ins_labels[pyramid_level].append(ins_label)

        # Concat along batch dimension by pyramid levels
        encoded_gt_cate_labels = []
        encoded_gt_ins_labels = []
        for pyramid_level in self.pyramid_levels:
            encoded_gt_cate_labels.append(
                torch.cat(dict_level_cate_labels[pyramid_level], dim=0))
            encoded_gt_ins_labels.append(
                torch.cat(dict_level_ins_labels[pyramid_level], dim=0))

        return tuple(encoded_gt_cate_labels), tuple(encoded_gt_ins_labels)


if __name__ == "__main__":
    pass
