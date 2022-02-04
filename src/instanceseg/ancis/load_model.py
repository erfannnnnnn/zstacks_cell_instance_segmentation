import torch

from .models import dec_net, dec_net_seg


def load_dec_model(z_depth):
    kwargs = {'pretrained' : False,
              'num_classes' : 2,
              'z_depth' : z_depth,
              'with_fpn' : False,
              'with_ancis_fusion' : False,
              'use_deform_conv' : False,
              'se_block' : False}
    return dec_net.resnetssd18(**kwargs)


def load_dec_seg_model(z_depth):
    kwargs = {'pretrained' : False,
              'num_classes' : 2,
              'z_depth' : z_depth,
              'with_fpn' : False,
              'with_ancis_fusion' : False,
              'use_deform_conv' : False,
              'se_block' : False}
    return dec_net_seg.resnetssd18(**kwargs)


def load_dec_weights(dec_model, dec_weights):
    dec_dict = torch.load(dec_weights)
    dec_dict_update = {}
    for k in dec_dict:
        if k.startswith('module') and not k.startswith('module_list'):
            dec_dict_update[k[7:]] = dec_dict[k]
        else:
            dec_dict_update[k] = dec_dict[k]
    dec_model.load_state_dict(dec_dict_update, strict=True)
    return dec_model


if __name__ == '__main__':
    pass
