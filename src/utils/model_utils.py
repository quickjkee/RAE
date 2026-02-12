import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from stage1 import RAE
import torch.nn as nn
from omegaconf import OmegaConf
import yaml
import torch
import os
import torchvision.transforms as transforms
from utils.train_utils import center_crop_arr, np_chw_to_pil
from yt_tools.utils import instantiate_from_config as instantiate_from_config2

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config) -> object:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    model = get_obj_from_str(config["target"])(**config.get("params", dict()))
    ckpt_path = config.get("ckpt", None)
    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location="cpu")
        # see if it's a ckpt from training by checking for "model"
        if "ema" in state_dict:
            state_dict = state_dict["ema"]
        elif "model" in state_dict:
            raise NotImplementedError("Loading from 'model' key not implemented yet.")
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict, strict=True)
        print(f'target {config["target"]} loaded from {ckpt_path}')
    return model


def create_dataloader(dataloader_config_path: str, batch_size: int, skip_rows=0):
    with open(dataloader_config_path) as f:
        dataloader_config = OmegaConf.create(yaml.load(f, Loader=yaml.SafeLoader))
    dataloader_config["params"]["batch_size"] = batch_size
    return instantiate_from_config2(dataloader_config, skip_rows=skip_rows)


def unpack_batch(batch, args):
    if os.path.exists(args.data_path):
        x, y = batch
    else:
        transform_train = transforms.Compose([
                            transforms.Lambda(np_chw_to_pil),
                            transforms.Lambda(lambda img: center_crop_arr(img, args.image_size)),
                            transforms.RandomHorizontalFlip(),
                            transforms.PILToTensor()
                        ])
        x = torch.stack([transform_train(img) for img in batch['image']])
        y = torch.tensor(batch['label'])
    return x, y

