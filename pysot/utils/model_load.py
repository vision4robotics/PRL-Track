from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging

import torch

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)
script_name = os.path.splitext(os.path.basename(__file__))[0]
log_filename = os.path.join(log_dir, script_name + '.txt')
logging.basicConfig(filename=log_filename, level=logging.INFO, 
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s', datefmt='%a, %d %b %Y %H:%M:%S')
logger = logging.getLogger(__name__)


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    missing_keys = [x for x in missing_keys if not x.endswith("num_batches_tracked")]

    if len(missing_keys) > 0:
        logger.warning("[Warning] missing keys: {}".format(missing_keys))
    logger.info("missing keys:{}".format(len(missing_keys)))
    if len(unused_pretrained_keys) > 0:
        logger.warning(
            "[Warning] unused_pretrained_keys: {}".format(unused_pretrained_keys)
        )
    logger.info("unused checkpoint keys:{}".format(len(unused_pretrained_keys)))

    logger.info("used keys:{}".format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, "load NONE from pretrained checkpoint"
    return True


def remove_prefix(state_dict, prefix):
    """Old style model is stored with all names of parameters
    share common prefix 'module.'"""
    logger.info("remove prefix '{}'".format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_pretrain(model, pretrained_path):
    logger.info("load pretrained model from {}".format(pretrained_path))
    device = torch.cuda.current_device()
    pretrained_dict = torch.load(
        pretrained_path, map_location=lambda storage, loc: storage.cuda(device)
    )
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict["state_dict"], "module.")
    else:
        pretrained_dict = remove_prefix(pretrained_dict, "module.")

    try:
        check_keys(model, pretrained_dict)
    except:
        logger.info(
            '[Warning]: using pretrain as features.\
                Adding "features." as prefix'
        )
        new_dict = {}
        for k, v in pretrained_dict.items():
            k = "features." + k
            new_dict[k] = v
        pretrained_dict = new_dict
        check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def load_pretrain2(model, pretrained_path):
    ckpt = torch.load(pretrained_path, map_location="cpu")["model"]
    new_dict = {}
    for k, v in ckpt.items():
        if "pos_embed" not in k and "mask_token" not in k:  # use fixed pos embed
            new_dict[k] = v
    model.load_state_dict(new_dict, strict=False)
    return model


def restore_from(model, optimizer, ckpt_path):
    device = torch.cuda.current_device()
    ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda(device))
    print(ckpt)
    epoch = ckpt["epoch"]

    ckpt_model_dict = remove_prefix(ckpt["state_dict"], "module.")
    check_keys(model, ckpt_model_dict)
    model.load_state_dict(ckpt_model_dict, strict=False)

    # check_keys(optimizer, ckpt['optimizer'])
    # optimizer.load_state_dict(ckpt['optimizer'])

    # return model, optimizer, epoch
    return model, 0, epoch
